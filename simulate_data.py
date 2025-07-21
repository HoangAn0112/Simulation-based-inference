import keras
import bayesflow as bf
import os
import torch
from solver import prior, solver_log
from config import *
from millard_ode.Millard_dicts import *
from tools import *

def simulate_data(sizes=[100]):
    """
    Simulate and save datasets of specified sizes
    
    Args:
        sizes (list): List of dataset sizes to simulate (e.g., [100, 500, 1000])
    """
    simulator = bf.simulators.make_simulator([prior, solver_log])
    
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

        plot_sampling_results(sampling, size = size)