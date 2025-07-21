
import numpy as np
import keras
from config import *
from model import GRUSummaryNetwork

def train_model(workflow, size=1000, epochs=100, batch_size=64):
    """Train the model and save the approximator"""
    training_data, validation_data = load_training_data(size)
    
    history = workflow.fit_offline(
        training_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
    )
    
    # Save the approximator (not the whole workflow)
    model_path = MODEL_FILE.format(size=size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save using Keras 3.0 format
    workflow.approximator.save(model_path, save_format="keras_v3")
    
    # Optionally save the full workflow state (if needed)
    # torch.save(workflow.state_dict(), model_path.replace('.keras', '.pth'))
    
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