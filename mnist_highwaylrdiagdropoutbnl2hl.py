#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
#from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, Dropout, Activation
from deepy.trainers import MomentumTrainer, LearningRateAnnealer, ScheduledLearningRateAnnealer, AdamTrainer
from deepy.conf import TrainerConfig
from deepy.utils import XavierGlorotInitializer, OrthogonalInitializer
from highwaylrdropoutbn_layer import HighwayLayerLRDropoutBatchNorm
from highwaylrdiagdropoutbn_layer import HighwayLayerLRDiagDropoutBatchNorm
from proper_batch_norm import BatchNormalization
from l2hinge_classifier import L2HingeNeuralClassifier

import sys
sys.path.append(os.path.dirname(__file__))
from mnist_small_validation import MnistDatasetSmallValid
from incremental_annealer import IncrementalLearningRateAnnealer

#model_path = os.path.join(os.path.dirname(__file__), "models", "highwaylrdropoutbnl2hl-l2reg1e-5-n1024-d256-T5-p02-03-03-05-b100-a3e-3-autoannealing.gz")
model_path = os.path.join(os.path.dirname(__file__), "models", "highwaylrdiagdropoutbnl2hl-l2reg1e-5-n1024-d256-T5-p02-03-03-05-b100-a3e-3-autoannealing.gz")

if __name__ == '__main__':
    dropout_p_0 = 0.2
    dropout_p_h_0 = 0.3
    dropout_p_h_1 = 0.3
    dropout_p_2 = 0.5
    T = 5
    n = 1024
    d = 256
    gate_bias = -1.0
    activation = 'relu'
    #l2_reg = 0.001
    l2_reg = 1e-5
    init = XavierGlorotInitializer()
    model = L2HingeNeuralClassifier(input_dim=28*28, last_layer_l2_regularization = l2_reg)
    model.stack(Dropout(p=dropout_p_0), Dense(n, init=init, disable_bias=True), BatchNormalization(), Activation(activation))
    #model.stack(Dropout(p=dropout_p_0), BatchNormalization())

    for _ in range(T):
        #model.stack(HighwayLayerLRDropoutBatchNorm(activation=activation, gate_bias=gate_bias, projection_dim=d, d_p_0 = dropout_p_h_0, d_p_1 = dropout_p_h_1, init=init))
        model.stack(HighwayLayerLRDiagDropoutBatchNorm(activation=activation, gate_bias=gate_bias, projection_dim=d, d_p_0 = dropout_p_h_0, d_p_1 = dropout_p_h_1, init=init, quasi_ortho_init=True))
    #model.stack(BatchNormalization(),Dropout(p=dropout_p_2), Dense(10, init=init))
    model.stack(Dropout(p=dropout_p_2), Dense(10, init=init))

    
    learning_rate_start  = 3e-3
    #learning_rate_target = 3e-7
    #learning_rate_epochs = 100
    #learning_rate_decay  = (learning_rate_target / learning_rate_start) ** (1.0 / learning_rate_epochs)
    conf = TrainerConfig()
    conf.learning_rate = LearningRateAnnealer.learning_rate(learning_rate_start)
    #conf.gradient_clipping = 1
    conf.patience = 20
    #conf.gradient_tolerance = 5
    conf.avoid_nan = True
    conf.min_improvement = 1e-10

    #trainer = MomentumTrainer(model)
    trainer = AdamTrainer(model, conf)

    mnist = MiniBatches(MnistDataset(), batch_size=100)
    #mnist = MiniBatches(MnistDatasetSmallValid(), batch_size=100)

    #trainer.run(mnist, controllers=[IncrementalLearningRateAnnealer(trainer, 0, learning_rate_decay)])
    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer, 3, 14)])
    logging.info('Setting best parameters for testing.')
    trainer.set_params(*trainer.best_params)
    trainer._run_test(-1, mnist.test_set())

    model.save_params(model_path)
