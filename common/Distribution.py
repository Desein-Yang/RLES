#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   distribution.py
@Time    :   2019/08/09 20:25:26
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   define probtype and basic functions about distribution(sample,kl and entropy)
'''

# here put the import slib

import numpy as np
#from tensorflow import tnsor
#import theano.tensor as T, theano
'''此处后端使用的是theano，改成TensorflowT需要对照修改,也可以不修改直接用
'''
from keras import backend as K

K.set_floatx(np.float64)
K.set_epsilon(1e-7)


def categorical_sample(prob_nk):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_nk = np.asarray(prob_nk)
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)# return cumlative sum of prob_nk[1]
    return np.argmax(csprob_nk > np.random.rand(N,1), axis=1) # return the index of maxvalue

TINY = np.finfo(np.float64).tiny

def categorical_kl(p_nk, q_nk):
    """
    caculate categorical KL divigence of p_nk,q_nk
    """
    p_nk = np.asarray(p_nk,dtype=np.float64)
    q_nk = np.asarray(q_nk,dtype=np.float64)
    ratio_nk = p_nk / (q_nk+TINY) # so we don't get warnings
    # next two lines give us zero when p_nk==q_nk==0 but inf when q_nk==0 
    ratio_nk[p_nk==0] = 1
    ratio_nk[(q_nk==0) & (p_nk!=0)] = np.inf
    return (p_nk * np.log(ratio_nk)).sum(axis=1)

def categorical_entropy(p_nk):
    """
    caculate categorical entropy of p_nk
    """
    p_nk = np.asarray(p_nk,dtype=np.float64)
    p_nk = p_nk.copy()
    p_nk[p_nk == 0] = 1
    return (-p_nk * np.log(p_nk)).sum(axis=1)

class ProbType(object):
    """
    define father class to represent different probability distribution type
    """
    def sampled_variable(self):
        raise NotImplementedError
    def prob_variable(self):
        raise NotImplementedError
    def likelihood(self, a, prob):
        raise NotImplementedError
    def loglikelihood(self, a, prob):
        raise NotImplementedError
    def kl(self, prob0, prob1):
        raise NotImplementedError
    def entropy(self, prob):
        raise NotImplementedError
    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return K.placeholder(name='a',dtype='int32', ndim=1) # creates one Variable int32x1 with name 'a'
        # return 
    def prob_variable(self):
        # return T.matrix('prob') # creates one variable with name prob
        return K.placeholder(name='prob')
    def likelihood(self, a, prob):
        return prob[K.arange(prob.shape[0]), a]
    def loglikelihood(self, a, prob):
        return K.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * K.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * K.log(prob0)).sum(axis=1)
    def sample(self, prob):
        return categorical_sample(prob)
    def maxprob(self, prob):
        return prob.argmax(axis=1)

class CategoricalOneHot(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return K.placeholder('a')
    def prob_variable(self):
        return K.placeholder('prob')
    def likelihood(self, a, prob):
        return (a * prob).sum(axis=1)
    def loglikelihood(self, a, prob):
        return K.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * K.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * K.log(prob0)).sum(axis=1)
    def sample(self, prob):
        assert prob.ndim == 2
        inds = categorical_sample(prob)
        out = np.zeros_like(prob)
        out[np.arange(prob.shape[0]), inds] = 1
        return out
    def maxprob(self, prob):
        out = np.zeros_like(prob)
        out[prob.argmax(axis=1)] = 1

class DiagGauss(ProbType):
    """
    covariance matrix of Gaussian distribution is diagnonal
    """
    def __init__(self, d):
        self.d = d
    def sampled_variable(self):
        return K.placeholder('a')
    def prob_variable(self):
        return K.placeholder('prob')
    def loglikelihood(self, a, prob):
        mean0 = prob[:,:self.d]
        std0 = prob[:, self.d:]
        # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
        return - 0.5 * K.square((a - mean0) / std0).sum(axis=1) - 0.5 * K.log(2.0 * np.pi) * self.d - K.log(std0).sum(axis=1)
    def likelihood(self, a, prob):
        return K.exp(self.loglikelihood(a, prob))
    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return K.log(std1 / std0).sum(axis=1) + ((K.square(std0) + K.square(mean0 - mean1)) / (2.0 * K.square(std1))).sum(axis=1) - 0.5 * self.d
    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return K.log(std_nd).sum(axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d
    def sample(self, prob):
        mean_nd = prob[:, :self.d] 
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(np.float64) * std_nd + mean_nd
        # casting random from real to float
    def maxprob(self, prob):
        return prob[:, :self.d]


