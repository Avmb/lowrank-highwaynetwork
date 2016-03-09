#!/usr/bin/env python
# -*- coding: utf-8 -*-

# by Antonio Valerio Miceli Barone

from deepy.networks import NeuralNetwork
from deepy.utils import FLOATX, EPSILON, CrossEntropyCost
import theano.tensor as T

class L2HingeNeuralClassifier(NeuralNetwork):
    """
    Classifier network with squared hinge loss.
    http://arxiv.org/abs/1306.0239
    """

    def __init__(self, input_dim, config=None, input_tensor=None, last_layer_l2_regularization = 0.0):
        super(L2HingeNeuralClassifier, self).__init__(input_dim, config=config, input_tensor=input_tensor)
        self.last_layer_l2_regularization = last_layer_l2_regularization

    def setup_variables(self):
        super(L2HingeNeuralClassifier, self).setup_variables()

        self.k = T.ivector('k')
        self.target_variables.append(self.k)

    def _cost_func(self, y):
        #y = T.clip(y, EPSILON, 1.0 - EPSILON)
        #return CrossEntropyCost(y, self.k).get()
        
        k_onehot = T.eye(y.shape[1])[self.k]
        k_centered = 2.0 * k_onehot - 1.0
        loss = T.mean(T.sqr(T.maximum(0.0, 1.0 - y*k_centered)))
        return loss

    def _error_func(self, y):
        return 100 * T.mean(T.neq(T.argmax(y, axis=1), self.k))

    @property
    def cost(self):
        last_layer_W = self.layers[-1].W
        regularization_cost = 0.0 if (self.last_layer_l2_regularization == 0.0) else self.last_layer_l2_regularization * last_layer_W.norm(2) / last_layer_W.size
        return self._cost_func(self.output) + regularization_cost

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)

    def prepare_training(self):
        self.training_monitors.append(("err", self._error_func(self.output)))
        self.testing_monitors.append(("err", self._error_func(self.test_output)))
        super(L2HingeNeuralClassifier, self).prepare_training()

    def predict(self, x):
        return self.compute(x).argmax(axis=1)

