import os
import numpy as np
import bayesflow as bf
from millard_ode.Millard_dicts import *
from scipy.integrate import solve_ivp
from millard_ode.deriv_equations_Millard import deriv_Millard
from numpy import genfromtxt

##########
DATA_FILE = "./data/"

# Load experimental data
data_1mM = genfromtxt(os.path.join(DATA_FILE,'data_1mM.csv'), delimiter=',')
data_t= data_1mM[1:, 0] 

def prior():
    sampled_parameters = {}
    for p, (low, high) in ode_parameter_log_ranges_dict.items():
        sampled_parameters[p] = np.random.uniform(low, high)
    return sampled_parameters


def solver_log(**kwargs):
    """    
    Args:
        kwargs: Either a single parameter dict (when N=None) 
               or a list/array of parameter dicts (when N is specified)
        N: Optional batch size (inferred automatically if None)
    """
    nan_arr = np.full_like(data_t, 0)
    results = dict(
        GLC=np.array([]),
        ACE_env=np.array([]),
        X=np.array([]),
    )

    new_ode_parameter = ode_parameters_dict.copy()
    new_sample_parameter = {k: 10 ** v for k, v in kwargs.items()}
    new_ode_parameter.update(new_sample_parameter)
    
    try:
        res = solve_ivp(
            fun=deriv_Millard,
            t_span=(0, 4.25),
            y0=np.array(y_1_0),  
            method='BDF',
            args=(new_ode_parameter,),
            t_eval=data_t
        )
        
        GLC, ACE_env, X, _, _, _ = res.y
        for arr, name in zip([GLC, ACE_env, X], ['GLC', 'ACE_env', 'X']):
            if len(arr) != len(data_t):
                print(f"{name} length mismatch: expected {len(data_t)}, got {len(arr)}, {arr}")
                results[name] = np.append(results[name], nan_arr)
            else:
                arr_np = np.array(arr)
                arr_np[arr_np <= 0] = 1 
                log_values = np.log(arr_np)
                results[name] = np.append(results[name], log_values)

    except Exception as e:
        print(f"solver failed: {e}")
        results['GLC'] = np.append(results['GLC'], nan_arr)
        results['ACE_env'] = np.append(results['ACE_env'], nan_arr)
        results['X'] = np.append(results['X'], nan_arr)
    
    return results

def solver(**kwargs):
    """    
    Args:
        kwargs: Either a single parameter dict (when N=None) 
               or a list/array of parameter dicts (when N is specified)
        N: Optional batch size (inferred automatically if None)
    """
    nan_arr = np.full_like(data_t, np.nan)
    results = dict(
        GLC=np.array([]),
        ACE_env=np.array([]),
        X=np.array([]),
        ACCOA=np.array([]),
        ACP=np.array([]),
        ACE_cell=np.array([]),
    )

    new_ode_parameter = ode_parameters_dict.copy()
    new_sample_parameter = {k: 10 ** v for k, v in kwargs.items()}
    new_ode_parameter.update(new_sample_parameter)
    
    try:
        res = solve_ivp(
            fun=deriv_Millard,
            t_span=(0, 4.25),
            y0=np.array(y_1_0),  
            method='BDF',
            args=(new_ode_parameter,),
            t_eval=data_t
        )
        
        GLC, ACE_env, X, ACCOA, ACP, ACE_cell = res.y
        for arr, name in zip([GLC, ACE_env, X,ACCOA, ACP, ACE_cell],['GLC','ACE_env','X','ACCOA','ACP','ACE_cell']):
            if len(arr) != len(data_t):
                print(f"{name} length mismatch: expected {len(data_t)}, got {len(arr)}, {arr}")
                results[name] = np.append(results[name], nan_arr)
            else:
                results[name] = np.append(results[name], arr)

    except Exception as e:
        print(f"solver failed: {e}")
    
    return results