from config import *
cfg = Config()

import bayesflow as bf
import matplotlib.pyplot as plt
from model import *
from train_inference import *
from simulate_data import *
from solver import prior, solver_log
from tools import plot_results_grid


def run_training():
    cfg = Config()
    
    # Load data
    train_data, valid_data = cfg.load_training_data()
    if train_data is None:
        raise RuntimeError("Failed to load training data")

    # Training setup
    simulator = bf.simulators.make_simulator([prior, solver_log])
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series(["GLC", "ACE_env", "X"])
        .concatenate(PARAMETERS_NAME, into="inference_variables")
        .concatenate(["GLC", "ACE_env", "X"], into="summary_variables")
    )
    
    train_model(
        create_workflow(simulator, adapter),
        cfg
    )


def run_inference(model_size, num_samples):
    model_path = f"model/CoupleFlow_{model_size}.keras"
    approximator = load_saved_model(model_path)

    data_1mM = np.genfromtxt("./data/data_1mM.csv", delimiter=',')
    variable_data = {
        "GLC": data_1mM[1:, 3],
        "ACE_env": data_1mM[1:, 1],
        "X": data_1mM[1:, 2]
    }

    log_data = {k: np.log(v) for k, v in variable_data.items()}
    samples = approximator.sample(conditions=log_data, num_samples=num_samples)

    from solver import solver
    results = [solver(**{k: v[0][i][0] for k, v in samples.items()}) 
               for i in range(num_samples)]
    plot_results_grid(results, data_1mM)
    plt.show()
    
    return results

def main():
    cfg = Config()
    settings = cfg.settings
    if cfg.mode == "simulate":
        sizes = settings["sizes"]
        print(f"Simulating datasets with sizes: {sizes}")
        simulate_data(sizes=sizes)

    if cfg.mode == 'train':
        print(f"Training with config: {settings}")
        run_training()
    
    elif cfg.mode == 'infer':
        print(f"Inference with model size: {cfg.settings['model_size']}")
        run_inference(cfg.settings['model_size'], cfg.settings["num_samples"])

if __name__ == "__main__":
    main()