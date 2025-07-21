import os
from numpy import genfromtxt
from tools import load_sampled_data
from millard_ode.Millard_dicts import *


# Environment settings
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"

# Path configurations
DATA_FILE = "./data/"
MODEL_FILE = "model/CoupleFlow_{size}.keras"
SAMPLE_FILE = "data/sampled_dataset_{size}.pth"
VALIDATION_FILE = "data/sampled_dataset_valid_100.pth"

# Simulation parameters
PARAMETERS_NAME = list(ode_parameter_log_ranges_dict.keys()) 


def load_experimental_data(data_path):
    """Load experimental data from CSV file"""
    data = genfromtxt(data_path, delimiter=',')
    data_t = data[1:, 0]
    variable_data = {
        "GLC": data[1:, 3],
        "ACE_env": data[1:, 1],
        "X": data[1:, 2]
    }
    return data_t, variable_data

def load_training_data(size=1000):
    """Load training and validation data"""
    try:
        training_data = load_sampled_data(SAMPLE_FILE.format(size=size))
        validation_data = load_sampled_data(VALIDATION_FILE)
        return training_data, validation_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None