import keras
from bayesflow import networks
import bayesflow as bf

import keras
from bayesflow import networks

@keras.saving.register_keras_serializable(package="CustomModels")
class GRUSummaryNetwork(networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(8)
        
    def call(self, time_series, **kwargs):
        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        return self.summary_stats(summary)
    
    def get_config(self):
        return super().get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_workflow(simulator, adapter):
    """Create and configure the BayesFlow workflow"""
    summary_net = GRUSummaryNetwork()
    inference_net = networks.CouplingFlow()
    
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        global_clipnorm=1.0
    )
    
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
        optimizer=optimizer,
        standardize=True,
        callbacks=lr_scheduler
    )