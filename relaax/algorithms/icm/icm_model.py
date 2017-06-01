from __future__ import absolute_import

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from .lib import icm_graph
from . import icm_config


class Network(subgraph.Subgraph):
    def build_graph(self):
        conv_layer = dict(type=layer.Convolution, activation=layer.Activation.Elu,
                          n_filters=32, filter_size=[3, 3], stride=[1, 1],
                          border=layer.Border.Same)
        icm_conv = [dict(conv_layer)] * 4
        input = layer.Input(icm_config.config.input, descs=icm_conv)

        sizes = icm_config.config.hidden_sizes
        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size,
                                     activation=layer.Activation.Relu) for size in sizes])

        lstm = layer.LSTM(graph.Expand(dense, 0), size=sizes[-1])
        head = graph.Reshape(lstm, [-1, sizes[-1]])

        actor = layer.Actor(head, icm_config.config.output)
        critic = layer.Dense(head, 1)

        self.ph_state = input.ph_state

        self.ph_lstm_state = lstm.ph_state
        self.ph_lstm_step = lstm.ph_step
        self.lstm_zero_state = lstm.zero_state
        self.lstm_state = lstm.state

        self.actor = actor
        self.critic = graph.Flatten(critic)

        layers = [input, dense, actor, critic, lstm]
        self.weights = layer.Weights(*layers)
