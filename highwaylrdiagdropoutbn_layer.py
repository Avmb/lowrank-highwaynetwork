#!/usr/bin/env python
# -*- coding: utf-8 -*-

# by Antonio Valerio Miceli Barone

import numpy as np
from deepy.layers import NeuralLayer
from deepy.utils import build_activation, global_theano_rand, OrthogonalInitializer, FLOATX

import theano.tensor as T

class HighwayLayerLRDiagDropoutBatchNorm(NeuralLayer):
    """
    Low-rank plus diagonal Highway network layer with internal dropout and batch normalization.
    
    Extends http://arxiv.org/abs/1505.00387.
    """

    def __init__(self, activation='relu', init=None, gate_bias=-5, projection_dim=10, d_p_0 = 0.3, d_p_1 = 0.3, epsilon=1e-4, tau=0.1, diag_init_val=1e-2, quasi_ortho_init=False):
        super(HighwayLayerLRDiagDropoutBatchNorm, self).__init__("highwayLRDiagDropoutBatchNorm")
        self.activation = activation
        self.init = init
        self.gate_bias = gate_bias
        self.projection_dim = projection_dim
        self.d_p_0 = d_p_0
        self.d_p_1 = d_p_1
        self.epsilon = epsilon
        self.tau = tau
        self.diag_init_val = diag_init_val
        self.quasi_ortho_init = quasi_ortho_init

    def setup(self):
        self.output_dim = self.input_dim
        self._act = build_activation(self.activation)
        self.W_hl = self.create_weight(self.input_dim, self.projection_dim, "hl", initializer=self.init)
        self.W_tl = self.create_weight(self.input_dim, self.projection_dim, "tl", initializer=self.init)
        self.W_hr = self.create_weight(self.projection_dim, self.input_dim, "hr", initializer=self.init)
        self.W_tr = self.create_weight(self.projection_dim, self.input_dim, "tr", initializer=self.init)
        self.B_h = self.create_bias(self.input_dim, "h")
        self.B_t = self.create_bias(self.input_dim, "t", value=self.gate_bias)
        self.D_h = self.create_vector(self.input_dim, "D_h")
        self.D_t = self.create_vector(self.input_dim, "D_t")
        self.D_h.set_value(np.ones(self.input_dim, dtype=FLOATX) * self.diag_init_val)
        self.D_t.set_value(np.ones(self.input_dim, dtype=FLOATX) * self.diag_init_val)
        
        self.S_h = self.create_vector(self.input_dim, "S_h")
        self.S_t = self.create_vector(self.input_dim, "S_t")
        self.S_h.set_value(np.ones(self.input_dim, dtype=FLOATX))
        self.S_t.set_value(np.ones(self.input_dim, dtype=FLOATX))

        self.register_parameters(self.W_hl, self.B_h, self.W_tl, self.B_t, self.W_hr, self.W_tr, self.D_h, self.D_t, self.S_h, self.S_t)

        self.Mean_hl = self.create_vector(self.projection_dim, "Mean_hl")
        self.Mean_tl = self.create_vector(self.projection_dim, "Mean_tl")
        self.Mean_hr = self.create_vector(self.input_dim, "Mean_hr")
        self.Mean_tr = self.create_vector(self.input_dim, "Mean_tr")
        self.Std_hl = self.create_vector(self.projection_dim, "Std_hl")
        self.Std_tl = self.create_vector(self.projection_dim, "Std_tl")
        self.Std_hr = self.create_vector(self.input_dim, "Std_hr")
        self.Std_tr = self.create_vector(self.input_dim, "Std_tr")
        self.Std_hl.set_value(np.ones(self.projection_dim, dtype=FLOATX))
        self.Std_tl.set_value(np.ones(self.projection_dim, dtype=FLOATX))
        self.Std_hr.set_value(np.ones(self.input_dim, dtype=FLOATX))
        self.Std_tr.set_value(np.ones(self.input_dim, dtype=FLOATX))
        
        self.register_free_parameters(self.Mean_hl, self.Mean_tl, self.Mean_hr, self.Mean_tr, self.Std_hl, self.Std_tl, self.Std_hr, self.Std_tr)
        
        if self.quasi_ortho_init:
        	self.setup_quasi_ortho_init()
    
    def setup_quasi_ortho_init(self):
    	ortho_init = OrthogonalInitializer()
    	ortho_init.rand = self.init.rand
    	
    	# Initialize low-rank decomposition matrices
    	w_h = ortho_init.sample((self.input_dim, self.input_dim))
    	w_t = ortho_init.sample((self.input_dim, self.input_dim))
    	w_h_u, w_h_sv, w_h_v = np.linalg.svd(w_h)
    	w_t_u, w_t_sv, w_t_v = np.linalg.svd(w_t)
    	h_sqsv_truncated = np.diag(np.sqrt(w_h_sv[:self.projection_dim]))
    	t_sqsv_truncated = np.diag(np.sqrt(w_t_sv[:self.projection_dim]))
    	w_h_u_truncated = w_h_u[:, :self.projection_dim]
    	w_t_u_truncated = w_t_u[:, :self.projection_dim]
    	w_h_v_truncated = w_h_v[:self.projection_dim, :]
    	w_t_v_truncated = w_t_v[:self.projection_dim, :]
    	w_hl = np.dot(w_h_u_truncated, h_sqsv_truncated)
    	w_tl = np.dot(w_t_u_truncated, t_sqsv_truncated)
    	w_hr = np.dot(h_sqsv_truncated, w_h_v_truncated)
    	w_tr = np.dot(t_sqsv_truncated, w_t_v_truncated)
    	
    	# Initialize diagonal matrix
    	test_vec = self.init.rand.normal(0.0, 1.0, (self.input_dim,))
    	err_h = np.dot(test_vec, w_h) - np.dot(test_vec, w_hl).dot(w_hr)
    	err_t = np.dot(test_vec, w_t) - np.dot(test_vec, w_tl).dot(w_tr)
    	d_h = err_h / (test_vec + self.epsilon)
    	d_t = err_t / (test_vec + self.epsilon)
    	
    	# Correct for dropout
    	w_hl /= (1.0 - self.d_p_0)
    	w_tl /= (1.0 - self.d_p_0)
    	d_h  /= (1.0 - self.d_p_0)
    	d_t  /= (1.0 - self.d_p_0)
    	w_hr /= (1.0 - self.d_p_1)
    	w_tr /= (1.0 - self.d_p_1)
    	
    	# Set values
    	self.W_hl.set_value(w_hl.astype(FLOATX))
    	self.W_tl.set_value(w_tl.astype(FLOATX))
    	self.W_hr.set_value(w_hr.astype(FLOATX))
    	self.W_tr.set_value(w_tr.astype(FLOATX))
    	self.D_h.set_value(d_h.astype(FLOATX))
    	self.D_t.set_value(d_t.astype(FLOATX))

    def output(self, x):
        d_0 = global_theano_rand.binomial(x.shape, p=1-self.d_p_0, dtype=FLOATX)
        d_1 = global_theano_rand.binomial((x.shape[0], self.projection_dim), p=1-self.d_p_1, dtype=FLOATX)
        
        tl_raw = T.dot(x * d_0, self.W_tl)
        hl_raw = T.dot(x * d_0, self.W_hl)
        tl_mean = T.mean(tl_raw, axis=0)
        hl_mean = T.mean(hl_raw, axis=0)
        tl_std = T.std(tl_raw, axis=0)
        hl_std = T.std(hl_raw, axis=0)
        tl = (tl_raw - tl_mean) / (tl_std + self.epsilon)
        hl = (hl_raw - hl_mean) / (hl_std + self.epsilon)
        new_Mean_tl = self.tau * tl_mean + (1.0 - self.tau) * self.Mean_tl
        new_Mean_hl = self.tau * hl_mean + (1.0 - self.tau) * self.Mean_hl
        new_Std_tl = self.tau * tl_std + (1.0 - self.tau) * self.Std_tl
        new_Std_hl = self.tau * hl_std + (1.0 - self.tau) * self.Std_hl
        
        tr_raw = (tl * d_1).dot(self.W_tr) + (x * d_0 * self.D_h)
        hr_raw = (hl * d_1).dot(self.W_hr) + (x * d_0 * self.D_t)
        tr_mean = T.mean(tr_raw, axis=0)
        hr_mean = T.mean(hr_raw, axis=0)
        tr_std = T.std(tr_raw, axis=0)
        hr_std = T.std(hr_raw, axis=0)
        tr = (tr_raw - tr_mean) / (tr_std + self.epsilon)
        hr = (hr_raw - hr_mean) / (hr_std + self.epsilon)
        new_Mean_tr = self.tau * tr_mean + (1.0 - self.tau) * self.Mean_tr
        new_Mean_hr = self.tau * hr_mean + (1.0 - self.tau) * self.Mean_hr
        new_Std_tr = self.tau * tr_std + (1.0 - self.tau) * self.Std_tr
        new_Std_hr = self.tau * hr_std + (1.0 - self.tau) * self.Std_hr
        
        t  = T.nnet.sigmoid(tr * self.S_t + self.B_t)
        h  = self._act(hr * self.S_h + self.B_h)
        rv = h * t + x * (1 - t)
        
        self.register_training_updates((self.Mean_tl, new_Mean_tl), 
                                       (self.Mean_hl, new_Mean_hl), 
                                       (self.Mean_tr, new_Mean_tr), 
                                       (self.Mean_hr, new_Mean_hr),
                                       (self.Std_tl, new_Std_tl), 
                                       (self.Std_hl, new_Std_hl), 
                                       (self.Std_tr, new_Std_tr), 
                                       (self.Std_hr, new_Std_hr))
        
        return rv
    
    def test_output(self, x):
        d_0 = 1.0 - self.d_p_0
        d_1 = 1.0 - self.d_p_1
        
        tl_raw = T.dot(x * d_0, self.W_tl)
        hl_raw = T.dot(x * d_0, self.W_hl)
        tl = (tl_raw - self.Mean_tl) / (self.Std_tl + self.epsilon)
        hl = (hl_raw - self.Mean_hl) / (self.Std_hl + self.epsilon)
        
        tr_raw = (tl * d_1).dot(self.W_tr) + (x * d_0 * self.D_h)
        hr_raw = (hl * d_1).dot(self.W_hr) + (x * d_0 * self.D_t)
        tr = (tr_raw - self.Mean_tr) / (self.Std_tr + self.epsilon)
        hr = (hr_raw - self.Mean_hr) / (self.Std_hr + self.epsilon)
        
        t  = T.nnet.sigmoid(tr * self.S_t + self.B_t)
        h  = self._act(hr * self.S_h + self.B_h)
        rv = h * t + x * (1 - t)
        
        return rv

