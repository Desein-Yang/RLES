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

def MultiLayerPerc():
    '''
    construct a multilayer perception network with 64 hidden units to make policy
    # Argument: cfg
    # 
    '''
    # if value is BOX it means continious else Discrete
    ifinstance(ac_space,Box): 

    ifinstance(ac_space,Discrete):

    model=Sequential()
    hid_sizes=cfg["hid_sizes"]
    activation=cfg["activation"]
    for (i, layeroutsize) in enumerate(hid_sizes):
        if i==0 
        inshp = dict(input_shape=ob_space.shape) 
        else 
        inshp={}
        model.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))

