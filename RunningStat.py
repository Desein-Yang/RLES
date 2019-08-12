#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   RunningStat.py
@Time    :   2019/08/09 20:30:12
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   Define class of running and controling state and episodes
'''

# here put the import lib
import numpy as np

class RunningStat(object):
    '''
    Define the markov state 
    attribute:sum and sumsq of reward,count of rounds,mean,std
    method:init, increment, set_from_init
    '''
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count
