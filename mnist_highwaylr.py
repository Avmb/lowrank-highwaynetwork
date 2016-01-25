#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This experiment setting is described in following paper:
http://arxiv.org/abs/1505.00387 .

With highway network layers, Very deep networks (20 layers here) can be trained properly.
"""

import logging, os
logging.basicConfig(level=logging.INFO)

from deepy.dataset import MnistDataset, MiniBatches
from deepy.networks import NeuralClassifier
from deepy.layers import Dense, Softmax, Dropout
from deepy.trainers import MomentumTrainer, LearningRateAnnealer, AdamTrainer
from deepy.conf import TrainerConfig
from highwaylr_layer import HighwayLayerLR

model_path = os.path.join(os.path.dirname(__file__), "models", "highwaylr1.gz")

if __name__ == '__main__':
    dropout_p = 0.3
    model = NeuralClassifier(input_dim=28*28)
    model.stack(Dense(1000, 'relu'), Dropout(p=dropout_p))
    for _ in range(10):
        model.stack(HighwayLayerLR(activation='relu', gate_bias=-1.5, projection_dim=5))
    model.stack(Dropout(p=dropout_p), Dense(10, 'linear'),
                Softmax())

    
    conf = TrainerConfig()
    conf.learning_rate = LearningRateAnnealer.learning_rate(1e-5)
    conf.gradient_clipping = 1
    conf.patience = 50
    #conf.gradient_tolerance = 5
    conf.avoid_nan = True
    conf.min_improvement = 1e-7
    conf.weight_decay = 1e-5

    #trainer = MomentumTrainer(model)
    trainer = AdamTrainer(model, conf)

    mnist = MiniBatches(MnistDataset(), batch_size=20)

    trainer.run(mnist, controllers=[LearningRateAnnealer(trainer)])

    model.save_params(model_path)
