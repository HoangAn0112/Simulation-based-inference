import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt

from millard_ode.Millard_dicts import *
from millard_ode.tools import ssr_error
from tools import *
from solver import prior, solver_log, solver

# TO DOs: replug into solver

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

##########
DATA_FILE = "./data/"

print("import data")
data_1mM = genfromtxt(os.path.join(DATA_FILE,'data_1mM.csv'), delimiter=',')
data_t= data_1mM[1:, 0]  
variable_data = {"GLC": data_1mM[1:, 3], "ACE_env": data_1mM[1:, 1], "X":data_1mM[1:, 2]}
variable_no_data  = {"ACCOA":None,"ACP":None,"ACE_cell":None}
new_ode_parameters = ode_parameters_dict.copy()
parameters_name = ode_parameter_log_ranges_dict.keys()


simulator = bf.simulators.make_simulator([prior, solver_log])
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


size = 10000
number_of_results = 10
sample_file_name = f"data/sampled_dataset_{size}.pth"
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

# plt.show()
####################
# # SAve model
# from pathlib import Path
# filepath = Path("model") / f"CoupleFlow_{size}.keras"
# filepath.parent.mkdir(exist_ok=True)
# workflow.approximator.save(filepath=filepath)

###################
# Inference
log_variable_data = {k: np.log(v) for k, v in variable_data.items()}
samples = workflow.sample(conditions = log_variable_data, num_samples=number_of_results )
# samples = workflow.approximator.estimate(conditions=log_variable_data)
# print(f' keras {keras.tree.map_structure(keras.ops.shape, samples)}')
# print(samples)
# result = {k: v['mean'][0][0] for k,v in samples.items()}
# # Convert into a nice format 2D data frame
# # samples_frame = workflow.samples_to_data_frame(samples)

# # print(samples_frame)
# dict_try = result
# result_object = solver(**dict_try)
# print(result_object)
# plot_results(result_object, data_1mM)

# print(workflow.samples_to_data_frame(samples))
# for i in range(number_of_results):
#     result = {k: v[0][i][0] for k,v in samples.items()}
#     result_object = solver(**result)
#     print(result)
    # print(ssr_error(variable_standard_deviations_dict,
    #       observables=observables,
    #       variable_data=variable_data,
    #       variable_res=result_time_serie_dict,
    #      ))
    # plot_results(result_object, data_1mM)

millard_parameters = {k: np.log(ode_parameters_dict[k]) for k in parameters_name}
all_results = []
for i in range(number_of_results):
    result = {k: v[0][i][0] for k,v in samples.items()}
    result_object = solver(**result)
    all_results.append(result_object)
    print(result)

    plt.scatter(millard_parameters.values(), result.values())
    plt.xlabel('millard')
    plt.ylabel('sbi')
    plt.title('compare parameters')
    min_val = min(min(millard_parameters.values()), min(result.values()))
    max_val = max(max(millard_parameters.values()), max(result.values()))
    plt.axline((min_val, min_val), (max_val, max_val), color='red', linestyle='--')

    plt.show()


plot_results_grid(all_results, data_1mM)



