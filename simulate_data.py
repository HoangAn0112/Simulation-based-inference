import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
from scipy.integrate import odeint
from functools import partial
import keras
import bayesflow as bf
import os
import torch
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
parameters_name = ode_parameter_log_ranges_dict.keys()

def prior():
    sampled_parameters = {}
    for p, (low, high) in ode_parameter_log_ranges_dict.items():
        sampled_parameters[p] = np.random.uniform(low, high)
    return sampled_parameters

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
    )

    new_ode_parameter = ode_parameters_dict.copy()
    new_sample_parameter = {k: 10**v for k, v in kwargs.items()}
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
                results[name] = np.append(results[name], arr)

    except Exception as e:
        print(f"solver failed: {e}")
        results['GLC'] = np.append(results['GLC'], nan_arr)
        results['ACE_env'] = np.append(results['ACE_env'], nan_arr)
        results['X'] = np.append(results['X'], nan_arr)
    
    return results

def remove_nan_rows(data_dict, length = None):
    """
    Removes all values at positions (indices) where any key in the dictionary
    has a NaN value. Assumes all values in the dictionary are lists of equal length.
    
    Args:
        data_dict (dict): Dictionary with list values.
        
    Returns:
        dict: A new dictionary with NaN-containing rows removed across all keys.
    """
    # Convert everything to np.array for consistency
    data_dict = {k: np.atleast_2d(np.array(v)) for k, v in data_dict.items()}

    # Infer length (rows)
    if length is None:
        length = next(iter(data_dict.values())).shape[0]

    # Build a mask of valid (non-NaN) rows
    valid_mask = np.ones(length, dtype=bool)
    for key, array in data_dict.items():
        if array.shape[0] != length:
            raise ValueError(f"Key '{key}' has inconsistent length: {array.shape[0]} != {length}")
        valid_mask &= ~np.isnan(array).any(axis=1)

    # Apply mask and ensure 2D shape
    cleaned_dict = {key: array[valid_mask] for key, array in data_dict.items()}

    return cleaned_dict

def check_for_nan_inf(data):
    flat_data = keras.tree.flatten(data)
    for i, tensor in enumerate(flat_data):
        arr = keras.ops.convert_to_numpy(tensor)
        if np.isnan(arr).any():
            print(f"Tensor {i} has NaNs.")
        if np.isinf(arr).any():
            print(f"Tensor {i} has Infs.")
    print("No INF")

simulator = bf.simulators.make_simulator([prior, solver])

# size = 10000
# sampling = simulator.sample(size)
# sampling = remove_nan_rows(sampling,size)

sizes = [100]

for size in sizes:
    # Sample and clean
    sampling = simulator.sample(size)
    sampling = remove_nan_rows(sampling, size)
    
    # Print shape for confirmation
    print(f"Size {size}:")
    print(keras.tree.map_structure(keras.ops.shape, sampling))
    # print(sampling.dtype)
    
    # Save each dataset to its own file
    filename = f"data/sampled_dataset_{size}.pth"
    torch.save(sampling, filename)

    print(f"Saved to {filename}")

# print(keras.tree.map_structure(keras.ops.shape, sampling))

    # Convert time-series data to numpy arrays
    time_series = {
        'GLC': keras.ops.convert_to_numpy(sampling['GLC']),
        'ACE_env': keras.ops.convert_to_numpy(sampling['ACE_env']),
        'X': keras.ops.convert_to_numpy(sampling['X'])
    }

    # Identify scalar parameters (exclude time-series keys)
    param_keys = [k for k in sampling.keys() if k not in ['GLC', 'ACE_env', 'X']]

    # Create the figure with two columns
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # Left: Time series (1/3), Right: Histograms (2/3)

    # --- Left Column: Time Series ---
    left_gs = gs[0].subgridspec(3, 1)  # 3 rows for GLC, ACE_env, X
    axes_left = [fig.add_subplot(left_gs[i]) for i in range(3)]

    time_points = np.arange(12)  # Time steps (adjust if needed)

    for ax, (var_name, data) in zip(axes_left, time_series.items()):
        mean = np.mean(data, axis=0).flatten()
        std = np.std(data, axis=0).flatten()
        
        ax.plot(time_points, mean, 'o-', label='Mean', color='blue')
        ax.fill_between(
            time_points, mean - std, mean + std,
            alpha=0.3, color='blue', label='±1 SD'
        )
        ax.set_title(f'{var_name} over Time', fontsize=10)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True)

    # --- Right Column: Parameter Histograms ---
    n_cols = 2  # Adjust columns per row as needed
    n_rows = (len(param_keys) // n_cols) + (1 if len(param_keys) % n_cols else 0) 
    right_gs = gs[1].subgridspec(n_rows, n_cols)
    axes_right = [fig.add_subplot(right_gs[i]) for i in range(n_rows * n_cols)]

    for i, key in enumerate(param_keys):
        data = keras.ops.convert_to_numpy(sampling[key]).flatten()
        sns.histplot(data, ax=axes_right[i], kde=True, bins=20, color='green')
        axes_right[i].set_title(f'{key}', fontsize=9)
        axes_right[i].set_xlabel('Value', fontsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes_right)):
        axes_right[j].set_visible(False)

    plt.tight_layout()
    # plt.show()
    filename = f"data/sampled_dataset_{size}.png"
    plt.savefig(filename)