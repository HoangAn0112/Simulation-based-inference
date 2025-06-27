import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

import scipy
from scipy.integrate import odeint

import keras

import bayesflow as bf

import os
from numpy import genfromtxt
from millard_ode.tools import ssr_error
from millard_ode.Millard_dicts import variable_standard_deviations_dict
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from millard_ode.deriv_equations_Millard import deriv_Millard
from millard_ode.Millard_dicts import ode_parameters_dict,ode_parameter_ranges_dict, ode_parameter_log_ranges_dict

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

##########
# Initial conditions used by Millard
GLC_1_0 = 12.89999655
ACE_env_1_0 = 0.9200020244
X_1_0 = 0.06999993881

#Eyeball estimated initial conditions
ACCOA_1_0 = 0.27305
ACP_1_0 = 0.063
ACE_cell_1_0 = 1.035

# Initial conditions vector
y_1_0 = [GLC_1_0, ACE_env_1_0, X_1_0, ACCOA_1_0, ACP_1_0, ACE_cell_1_0] 

DATA_FILE = "./data/"

# Load experimental data
data_1mM = genfromtxt(os.path.join(DATA_FILE,'data_1mM.csv'), delimiter=',')
data_t_1mM = data_1mM[1:, 0]  

observables = ["GLC","ACE_env","X"]
variable_data = {"GLC": data_1mM[1:, 3], "ACE_env": data_1mM[1:, 1], "X":data_1mM[1:, 2]}
variable_no_data  = {"ACCOA":None,"ACP":None,"ACE_cell":None}
data_t = data_t_1mM
new_ode_parameters = ode_parameters_dict.copy()

def prior_millard():
     new_ode_parameters = ode_parameters_dict.copy()
     for p, (low, high) in ode_parameter_log_ranges_dict.items():
#          new_ode_parameters[p]= 10 ** np.random.uniform(np.log10(low), np.log10(high)
            new_ode_parameters[p]= 10 ** np.random.uniform(low, high)
     return dict(new_ode_parameters = new_ode_parameters)

def solver_millard(new_ode_parameters, N=None):
    """    
    Args:
        new_ode_parameters: Either a single parameter dict (when N=None) 
                           or a list/array of parameter dicts (when N is specified)
        N: Optional batch size (inferred automatically if None)
    """
    # Handle both single and batch cases
    if N is None:
        param_list = [new_ode_parameters] if isinstance(new_ode_parameters, dict) else new_ode_parameters
        N = len(param_list)
    else:
        param_list = new_ode_parameters
    
    # Initialize storage
    results = dict(
#        t =  data_t,
        GLC = np.array([]),
        ACE_env=np.array([]),
        X=np.array([]),
        ACCOA=np.array([]),
        ACP=np.array([]),
        ACE_cell=np.array([])
    )
    
    for i in range(N):
        new_ode_parameter =  {k: v[i] for k, v in param_list[0].items()}
        try:
            res = solve_ivp(fun=deriv_Millard,
                            t_span=(0, 4.25),
                            y0=np.array(y_1_0),  # Ensure 1D array
                            method='LSODA',
                            args=(new_ode_parameter,),
                            t_eval=data_t)
            
            GLC, ACE_env, X, ACCOA, ACP, ACE_cell = res.y

            results['GLC'] = np.append(results['GLC'], GLC)
            results['ACE_env'] = np.append(results['ACE_env'], ACE_env)
            results['X'] = np.append(results['X'], X)
            results['ACCOA'] = np.append(results['ACCOA'], ACCOA)
            results['ACP'] = np.append(results['ACP'], ACP)
            results['ACE_cell'] = np.append(results['ACE_cell'], ACE_cell)
        except:
            results['GLC'] = np.append(results['GLC'], np.nan)
            results['ACE_env'] = np.append(results['ACE_env'], np.nan)
            results['X'] = np.append(results['X'], np.nan)
            results['ACCOA'] = np.append(results['ACCOA'], np.nan)
            results['ACP'] = np.append(results['ACP'], np.nan)
            results['ACE_cell'] = np.append(results['ACE_cell'], np.nan)

    
    return results

simulator = bf.simulators.make_simulator([prior_millard, solver_millard])
res = simulator.sample(2000)
print("ODE parameter list:")
for key, value in res['new_ode_parameters'].items():
    print(f"{key} : {value.shape}")
print(f"GLC shape {res['GLC'].shape}")

