import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch" #tensorflow is faster than torch

import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf
from numpy import genfromtxt

from millard_ode.Millard_dicts import *
from millard_ode.tools import ssr_error
from tools import *
from solver import prior, solver_log, solver

##########
# TO - Dos: 
# - checking others 3 variables => not meaningful
# - test save and reload torch_model
# - more modular: create a main script
# - testing handmade statistics + point estimation 
# - others architechture: compare time and training data_1mM
# - running for other dataset
# - active learning for NPE

##########
DATA_FILE = "./data/"

data_1mM = genfromtxt(os.path.join(DATA_FILE,'data_1mM.csv'), delimiter=',')
data_t= data_1mM[1:, 0]  
variable_data = {"GLC": data_1mM[1:, 3], "ACE_env": data_1mM[1:, 1], "X":data_1mM[1:, 2]}
variable_no_data  = {"ACCOA":None,"ACP":None,"ACE_cell":None}
new_ode_parameters = ode_parameters_dict.copy()
parameters_name = list(ode_parameter_log_ranges_dict.keys())


simulator = bf.simulators.make_simulator([prior, solver_log])
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
inference_net = bf.networks.CouplingFlow()

point_inference_network = bf.networks.PointInferenceNetwork(
    scores=dict(
        mean=bf.scores.MeanScore(),
        # quantiles=bf.scores.QuantileScore(data_t),
    ),
)

# Configure optimizer with gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    global_clipnorm=1.0
)

# Configure learning rate scheduler for stability
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
    optimizer=optimizer,
    standardize=True,
    callbacks = lr_scheduler
)


size = 1000
number_of_results = 5
sample_file_name = f"data/sampled_dataset_{size}.pth"
plot_params = False

try:
    print("load pre data")
    training_data = load_sampled_data(sample_file_name)
    validation_data = load_sampled_data("data/sampled_dataset_valid_100.pth")
    
    print(f"Loaded {size} training samples")
    print(f"Loaded validation samples")

except Exception as e:
    print(f"Error loading data: {e}")

print(f'training - {keras.tree.map_structure(keras.ops.shape, training_data)}')

history = workflow.fit_offline(
    training_data,
    epochs=100, 
    batch_size=64, 
    validation_data=validation_data,
)
f = bf.diagnostics.plots.loss(history)
plt.show()

####################
# Save model
from pathlib import Path
filepath = Path("model") / f"CoupleFlow_{size}.keras"
filepath.parent.mkdir(exist_ok=True)
workflow.approximator.save(filepath=filepath)

###################
# # reload workflow
# approximator = keras.saving.load_model(filepath)
# Inference
log_variable_data = {k: np.log(v) for k, v in variable_data.items()}
samples = workflow.sample(conditions = log_variable_data, num_samples=number_of_results )

millard_parameters = {k: np.log10(ode_parameters_dict[k]) for k in parameters_name}

all_results = []
for i in range(number_of_results):
    result = {k: v[0][i][0] for k,v in samples.items()}
    result_object = solver(**result)
    all_results.append(result_object)
    print(result_object)
    print(result)

    if plot_params:
        plt.scatter(millard_parameters.values(), result.values())
        plt.xlabel('millard')
        plt.ylabel('sbi')
        plt.title('compare parameters')
        min_val = min(min(millard_parameters.values()), min(result.values()))
        max_val = max(max(millard_parameters.values()), max(result.values()))
        plt.axline((min_val, min_val), (max_val, max_val), color='red', linestyle='--')
        plt.show()


plot_results_grid(all_results, data_1mM)



