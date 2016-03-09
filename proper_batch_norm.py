#!/usr/bin/env python
# -*- coding: utf-8 -*-

# by Antonio Valerio Miceli Barone

import numpy as np
import theano.tensor as T

from deepy.layers import NeuralLayer
from deepy.utils import FLOATX

class BatchNormalization(NeuralLayer):
    """
    Batch normalization.
    http://arxiv.org/pdf/1502.03167v3.pdf
    """
    def __init__(self, with_scale=True, with_bias=True, epsilon=1e-4, tau=0.1):
        super(BatchNormalization,self).__init__("norm")
        self.with_scale = with_scale
        self.with_bias = with_bias
        self.epsilon = epsilon
        self.tau = tau

    def setup(self):
        if self.with_scale:
            self.S = self.create_vector(self.input_dim, "S")
            self.S.set_value(np.ones(self.input_dim, dtype=FLOATX))
            self.register_parameters(self.S)
        if self.with_bias:
            self.B = self.create_bias(self.input_dim, "B")
            self.register_parameters(self.B)
        self.Mean = self.create_vector(self.input_dim, "Mean")
        self.Std = self.create_vector(self.input_dim, "Std")

    def output(self, x):
        x_mean = T.mean(x, axis=0)
        x_std = T.std(x, axis=0)
        rv = (x - x_mean) / (x_std + self.epsilon)
        if self.with_scale:
            rv = rv * self.S
        if self.with_bias:
            rv = rv + self.B
        
        new_mean = self.tau * x_mean + (1.0 - self.tau) * self.Mean
        new_std = self.tau * x_std + (1.0 - self.tau) * self.Std
        self.register_training_updates((self.Mean, new_mean),
                                       (self.Std, new_std))
        
        return rv
    
    def test_output(self, x):
        rv = (x - self.Mean) / (self.Std + self.epsilon)
        if self.with_scale:
            rv = rv * self.S
        if self.with_bias:
            rv = rv + self.B
        return rv
