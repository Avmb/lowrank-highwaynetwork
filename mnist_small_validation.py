#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import tempfile
import os
import gzip
import urllib
import cPickle

from deepy.dataset import Dataset
import numpy as np

logging = logging.getLogger(__name__)

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

class MnistDatasetSmallValid(Dataset):

    def __init__(self):
        self._target_size = 10
        logging.info("loading minst data")
        path = os.path.join(tempfile.gettempdir(), "mnist.pkl.gz")
        if not os.path.exists(path):
            logging.info("downloading minst data")
            urllib.urlretrieve (MNIST_URL, path)
        self._train_set, self._valid_set, self._test_set = cPickle.load(gzip.open(path, 'rb'))
        
        # Moving validation examples to training set, leaving 1000
        train_set_x = np.vstack((self._train_set[0], self._valid_set[0][:-1000]))
        train_set_y = np.hstack((self._train_set[1], self._valid_set[1][:-1000]))
        valid_set_x = self._valid_set[0][-1000:]
        valid_set_y = self._valid_set[1][-1000:]
        self._train_set = (train_set_x, train_set_y)
        self._valid_set = (valid_set_x, valid_set_y)
        
        logging.info("[mnist small validation] training data size: %d" % len(self._train_set[0]))
        logging.info("[mnist small validation] valid data size: %d" % len(self._valid_set[0]))
        logging.info("[mnist small validation] test data size: %d" % len(self._test_set[0]))

    def train_set(self):
        data, target = self._train_set
        return zip(data,  np.array(target))

    def valid_set(self):
        data, target = self._valid_set
        return zip(data,  np.array(target))

    def test_set(self):
        data, target = self._test_set
        return zip(data,  np.array(target))
