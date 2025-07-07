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
from numpy import genfromtxt
from millard_ode.tools import ssr_error
from millard_ode.Millard_dicts import variable_standard_deviations_dict
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from millard_ode.deriv_equations_Millard import deriv_Millard
from millard_ode.Millard_dicts import ode_parameters_dict,ode_parameter_ranges_dict, ode_parameter_log_ranges_dict

# TO DOs: input model in log scale 

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
# simulator.sample(100)
adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .as_time_series(["GLC", "ACE_env","X"])
    .concatenate(parameters_name, into="inference_variables")
    .concatenate(["GLC", "ACE_env","X"], into="summary_variables")
)


class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(8)
        
    def call(self, time_series, **kwargs):
        """Compresses time_series of shape (batch_size, T, 1) into summaries of shape (batch_size, 8)."""

        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary
    

summary_net = GRU()

point_inference_network = bf.networks.PointInferenceNetwork(
    scores=dict(
        mean=bf.scores.MeanScore(),
        quantiles=bf.scores.QuantileScore(data_t),
    ),
)

# Configure optimizer with gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    global_clipnorm=1.0
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=point_inference_network,
    summary_network=summary_net,
    optimizer=optimizer,
    standardize=None
)

training_size = 1000
validation_size = 100
training_data = workflow.simulate(training_size)
training_data = remove_nan_rows(training_data,training_size)
validation_data = workflow.simulate(validation_size)
validation_data = remove_nan_rows(validation_data,validation_size)
check_for_nan_inf(training_data)
check_for_nan_inf(validation_data)

print(f'training - {keras.tree.map_structure(keras.ops.shape, training_data)}')
print(validation_data)

history = workflow.fit_offline(
    training_data,
    epochs=50, 
    batch_size=64, 
    validation_data=validation_data,
)
f = bf.diagnostics.plots.loss(history)

plt.show()