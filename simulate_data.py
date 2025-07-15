import keras
import bayesflow as bf
import os
from numpy import genfromtxt
import torch
from solver import prior, solver_log
from millard_ode.Millard_dicts import *
from tools import *
# TO DOs: replug into solver

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

##########
DATA_FILE = "./data/"

# Load experimental data
data_1mM = genfromtxt(os.path.join(DATA_FILE,'data_1mM.csv'), delimiter=',')
data_t= data_1mM[1:, 0]  
variable_data = {"GLC": data_1mM[1:, 3], "ACE_env": data_1mM[1:, 1], "X":data_1mM[1:, 2]}
variable_no_data  = {"ACCOA":None,"ACP":None,"ACE_cell":None}
new_ode_parameters = ode_parameters_dict.copy()
parameters_name = ode_parameter_log_ranges_dict.keys()


simulator = bf.simulators.make_simulator([prior, solver_log])

# size = 10000
# sampling = simulator.sample(size)
# sampling = remove_nan_rows(sampling,size)

sizes = [50000]

for size in sizes:
    # Sample and clean
    sampling = simulator.sample(size)
    sampling = remove_nan_rows(sampling, size)
    
    # Print shape for confirmation
    print(f"Size {size}:")
    print(keras.tree.map_structure(keras.ops.shape, sampling))
    check_for_nan_inf(sampling)
    
    # Save each dataset to its own file
    filename = f"data/sampled_dataset_{size}.pth"
    torch.save(sampling, filename)

    print(f"Saved to {filename}")

# print(keras.tree.map_structure(keras.ops.shape, sampling))

    plot_sampling_results(sampling, size = size)