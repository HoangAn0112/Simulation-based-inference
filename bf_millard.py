import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

from scipy.integrate import odeint


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


def prior():
    sampled_parameters = {}
    for p, (low, high) in ode_parameter_log_ranges_dict.items():
        sampled_parameters[p] = 10 ** np.random.uniform(low, high)
    return dict(sampled_parameters = sampled_parameters)

def solver(sampled_parameters):
    """    
    Args:
        sampled_parameters: Either a single parameter dict (when N=None) 
                           or a list/array of parameter dicts (when N is specified)
        N: Optional batch size (inferred automatically if None)
    """
    # Handle both single and batch cases
    if isinstance(sampled_parameters, dict):
        param_list = [sampled_parameters]
    else:
        param_list = sampled_parameters

    N = len(param_list)

    
    # Initialize storage
    results = dict(
#        t =  data_t,
        GLC = np.array([]),
        ACE_env=np.array([]),
        X=np.array([]),
        # ACCOA=np.array([]),
        # ACP=np.array([]),
        # ACE_cell=np.array([])
    )
    
    for i in range(N):
        new_ode_parameter = ode_parameters_dict.copy()
        new_sample_parameter =  {k: v[i] for k, v in param_list[0].items()}
        new_ode_parameter.update(new_sample_parameter)
        try:
            res = solve_ivp(fun=deriv_Millard,
                            t_span=(0, 4.25),
                            y0=np.array(y_1_0),  # Ensure 1D array
                            method='LSODA',
                            args=(new_ode_parameter,),
                            t_eval=data_t)
            
            GLC, ACE_env, X, _, _, _ = res.y

            results['GLC'] = np.append(results['GLC'], GLC)
            results['ACE_env'] = np.append(results['ACE_env'], ACE_env)
            results['X'] = np.append(results['X'], X)
            # results['ACCOA'] = np.append(results['ACCOA'], ACCOA)
            # results['ACP'] = np.append(results['ACP'], ACP)
            # results['ACE_cell'] = np.append(results['ACE_cell'], ACE_cell)
        except:
            results['GLC'] = np.append(results['GLC'], np.nan)
            results['ACE_env'] = np.append(results['ACE_env'], np.nan)
            results['X'] = np.append(results['X'], np.nan)
            # results['ACCOA'] = np.append(results['ACCOA'], np.nan)
            # results['ACP'] = np.append(results['ACP'], np.nan)
            # results['ACE_cell'] = np.append(results['ACE_cell'], np.nan)

    
    return results

simulator = bf.simulators.make_simulator([prior, solver])
simulation_sample = simulator.sample(100)
# print("ODE parameter list:")
# for key, value in res['sampled_parameters'].items():
#     print(f"{key} : {value.shape}")
# print(f"GLC shape {res['GLC'].shape}")
# print(res)



adapter = (
    bf.Adapter()
#    .broadcast("N", to="GLC")
#    .as_set(["x", "y"])
#    .constrain("sigma", lower=0)
#    .sqrt("N")
    .convert_dtype("float64", "float32")
    .concatenate(["sampled_parameters"], into="inference_variables")
    .concatenate(["GLC", "ACE_env","X"], into="summary_variables")
#    .rename("N", "inference_conditions")
)

processed_draws = adapter(simulation_sample)
print(processed_draws["summary_variables"].shape)
#print(processed_draws["inference_conditions"].shape)
print(processed_draws["inference_variables"].shape)