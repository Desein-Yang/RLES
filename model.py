#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   network.py
@Time    :   2019/08/10 21:43:16
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   construct network by keras
'''

# here put the import lib
from keras.models import Sequential
from keras.layers import Dense, Activation
from gym.spaces import Box, Discrete
from gym import Env
from common.Distribution import Categorical,DiagGauss

def MultiLayerPerc(cfg):
    '''
    construct a multilayer perception network with 64 hidden units to make policy
    # Argument: cfg
    # Return: model
    '''
    model=Sequential()
    hid_sizes=cfg["hid_sizes"]
    activation=cfg["activation"]
    # add two 64 hidden unit separated by tanh
    for (i, layeroutsize) in enumerate(hid_sizes):
        if i==0:
            inshp = dict(input_shape=ob_space.shape) 
        else:
            inshp={}
        model.add(Dense(layeroutsize, activation, **inshp))
        
    if isinstance(ac_space, Box):
        outdim=ac_space.shape[0]
        model.add(Dense(outdim))
        model.add(ConcatFixedStd())
    elif ifinstance(ac_space,Discrete):
        outdim=ac_space.n
        model.add(Dense(outdim, activation="softmax"))
    return model

def decision(ob_space,ac_space,model):
    '''
    Input:\n observation_space and action_space\n model\n
    Output:\n policy
    '''




