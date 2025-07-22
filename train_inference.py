
import numpy as np
import keras
from config import *
from model import GRUSummaryNetwork

def train_model(workflow, cfg: Config,):
    """
    Train the model and save the approximator using configuration
    
    Args:
        workflow: The BayesFlow workflow to train
        cfg: Config object containing all paths and settings
        size: Optional override for training size
        epochs: Optional override for epoch count
        batch_size: Optional override for batch size
    """
    # Get settings from config if not overridden
    settings = cfg.settings
    size = settings["size"]
    epochs = settings["epoch"]
    batch_size = settings["batch_size"]
    
    # Load data using config
    training_data, validation_data = cfg.load_training_data(size)
    if training_data is None or validation_data is None:
        raise ValueError("Failed to load training data")
    
    # Train the model
    history = workflow.fit_offline(
        training_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
    )
    
    # Save the model using config path
    model_path = cfg.model_path(size=size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    workflow.approximator.save(model_path, save_format="keras_v3")
    
    print(f"Model saved to {model_path}")
    return history


def inference(workflow, variable_data, num_samples=5):
    """Pure inference function"""
    log_variable_data = {k: np.log(v) for k, v in variable_data.items()}
    samples = workflow.sample(conditions=log_variable_data, num_samples=num_samples)
    
    # Process results
    from solver import solver
    return [solver(**{k: v[0][i][0] for k,v in samples.items()}) 
            for i in range(num_samples)]


def load_saved_model(model_path):
    """Load a saved Keras model with custom components."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Register custom objects used in the model
    custom_objects = {
        "GRUSummaryNetwork": GRUSummaryNetwork,
        "CustomModels>GRUSummaryNetwork": GRUSummaryNetwork  # Keras 3.0+ format
    }

    try:
        model = keras.saving.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # Disable compilation if not training
        )
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")