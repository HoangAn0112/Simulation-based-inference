import os
import bayesflow as bf
import keras
import argparse
import matplotlib.pyplot as plt
from config import *
from model import *
from train_inference import *
from solver import prior, solver_log
from tools import plot_results_grid


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bayesian parameter estimation workflow")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--size', type=int, default=1000, help='Training dataset size')

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference with saved model')
    infer_parser.add_argument('--model-size', type=int, default=1000, help='Size of model to load')
    infer_parser.add_argument('--num-samples', type=int, default=5, help='Number of posterior samples')

    return parser.parse_args()


def run_training(size):
    """Execute full training pipeline"""
    simulator = bf.simulators.make_simulator([prior, solver_log])
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series(["GLC", "ACE_env", "X"])
        .concatenate(PARAMETERS_NAME, into="inference_variables")
        .concatenate(["GLC", "ACE_env", "X"], into="summary_variables")
    )
    workflow = create_workflow(simulator, adapter)
    train_model(workflow, size=size)
    print(f"Training completed. Model saved to {MODEL_FILE.format(size=size)}")


def run_inference(model_size=1000, num_samples=5):
    # 1. Load the pre-trained model
    model_path = f"model/CoupleFlow_{model_size}.keras"
    approximator = load_saved_model(model_path)

    # 2. Load your experimental data
    data_1mM = np.genfromtxt("./data/data_1mM.csv", delimiter=',')
    variable_data = {
        "GLC": data_1mM[1:, 3],
        "ACE_env": data_1mM[1:, 1],
        "X": data_1mM[1:, 2]
    }

    # 3. Run inference
    log_data = {k: np.log(v) for k, v in variable_data.items()}
    samples = approximator.sample(conditions=log_data, num_samples=num_samples)

    # 4. Process results (using your existing solver)
    from solver import solver
    results = [solver(**{k: v[0][i][0] for k, v in samples.items()}) 
               for i in range(num_samples)]
    plot_results_grid(results, data_1mM)
    plt.show()
    
    return results

def main():
    args = parse_arguments()
    
    if args.command == 'train':
        run_training(args.size)
    elif args.command == 'infer':
        run_inference(args.model_size, args.num_samples)

if __name__ == "__main__":
    main()