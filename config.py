import os
import yaml
import argparse
from numpy import genfromtxt
from tools import load_sampled_data
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from millard_ode.Millard_dicts import ode_parameter_log_ranges_dict

def _setup_environment(config: Dict[str, Any]) -> None:
    """Internal function to set up environment variables"""
    if "KERAS_BACKEND" not in os.environ:
        os.environ["KERAS_BACKEND"] = config["environment"]["keras_backend"]

class Config:
    _instance = None

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Load config first
            with open(config_path, "r") as f:
                cls._instance._config = yaml.safe_load(f)
            # Set up environment BEFORE anything else
            _setup_environment(cls._instance._config)
        return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if not hasattr(self, '_initialized'):  # Prevent re-initialization
            self._initialized = True
            self.mode = self._parse_mode()
            self.SAMPLE_FILE = self._config["paths"]["sample"]
            self.VALIDATION_FILE = self._config["paths"]["validation"]

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _parse_mode(self) -> str:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "mode",
            choices=["simulate", "train", "infer"],
            help="Run mode: simulate/train/infer"
        )
        return parser.parse_args().mode

    @property
    def settings(self) -> Dict[str, Any]:
        """Returns settings for current mode"""
        return self._config["modes"][self.mode]

    @property
    def data_dir(self) -> Path:
        return Path(self._config["paths"]["data"])

    def model_path(self, size: int = None) -> Path:
        path = self._config["paths"]["model"]
        if size is not None:
            path = path.format(size=size)
        return Path(path)

    def sample_path(self, size: int = None) -> Path:
        path = self.SAMPLE_FILE
        if size is not None:
            path = path.format(size=size)
        return Path(path)

    @property
    def validation_path(self) -> Path:
        return Path(self.VALIDATION_FILE)
    
    
    def load_training_data(self, size: int = None) -> Tuple[Optional[Any], Optional[Any]]:
        """Load training and validation data"""
        if size is None:
            size = self.settings["size"]  # Direct access since we know it exists
            
        try:
            return (
                load_sampled_data(self.sample_path(size)),
                load_sampled_data(self.validation_path)
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None


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
