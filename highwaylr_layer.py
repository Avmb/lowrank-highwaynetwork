#!/usr/bin/env python
# -*- coding: utf-8 -*-

# by Antonio Valerio Miceli Barone

from deepy.layers import NeuralLayer
from deepy.utils import build_activation
import theano.tensor as T

class HighwayLayerLR(NeuralLayer):
    """
    Highway network layer.
    See http://arxiv.org/abs/1505.00387.
    """

    def __init__(self, activation='relu', init=None, gate_bias=-5, projection_dim=10):
        super(HighwayLayerLR, self).__init__("highwayLR")
        self.activation = activation
        self.init = init
        self.gate_bias = gate_bias
        self.projection_dim = projection_dim

    def setup(self):
        self.output_dim = self.input_dim
        self._act = build_activation(self.activation)
        self.W_hl = self.create_weight(self.input_dim, self.projection_dim, "hl", initializer=self.init)
        self.W_tl = self.create_weight(self.input_dim, self.projection_dim, "tl", initializer=self.init)
        self.W_hr = self.create_weight(self.projection_dim, self.input_dim, "hr", initializer=self.init)
        self.W_tr = self.create_weight(self.projection_dim, self.input_dim, "tr", initializer=self.init)
        self.B_h = self.create_bias(self.input_dim, "h")
        self.B_t = self.create_bias(self.input_dim, "t", value=self.gate_bias)

        self.register_parameters(self.W_hl, self.B_h, self.W_tl, self.B_t, self.W_hr, self.W_tr)

    def output(self, x):
        t = self._act(T.dot(x, self.W_tl).dot(self.W_tr) + self.B_t)
        h = self._act(T.dot(x, self.W_hl).dot(self.W_hr) + self.B_h)
        return h * t + x * (1 - t)

