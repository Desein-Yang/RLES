#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2019/08/10 21:43:16
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   construct network by keras\n has been tested successfully
'''

# here put the import lib
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Activation
from keras import backend as K
from keras import initializers as ini
from gym.spaces import Box, Discrete
from gym import Env
import gym

import argparse
from gym import wrappers,logger
import pip
import sys

# some fields
conv_args = {
    "padding": "VALID",
    #"bias_initializer": None,
    "activation": None,
    "kernel_initializer": ini.RandomNormal(0,0.05)
}
dense_args = {
    #"bias_initializer": None,
    "activation": None,
    "kernel_initializer": ini.RandomNormal(0,0.05)
}

bn_args = {
    #"decay": 0.,
    #"center": True,
    #"epsilon": 1e-8,
    #"scale"
    # "activation_fn": 'elu',
    #"is_training": False
}


def Network(ob_space,action_space):
    '''
    construct a multilayer perception network with 64 hidden units to make policy
    # Argument: cfg
    # Return: model
    '''
    model=Sequential()
    # get input shape
    

    # get output dimension
    outdim=0
    if isinstance(action_space, Box):
        outdim=action_space.shape[0]
    elif isinstance(action_space,Discrete):
        outdim=action_space.n
    assert outdim != 0 

    # construct models layer by layer
    # network architecture is DQN (activiation is modified to elu)
    model.add(Conv2D(filters=32,kernel_size=8,strides=4,**conv_args))
    model.add(Activation('elu',**bn_args))

    model.add(Conv2D(filters=64,kernel_size=4,strides=2,**conv_args))
    model.add(Activation('elu',**bn_args))

    model.add(Conv2D(filters=64,kernel_size=3,strides=1,**conv_args))
    model.add(Activation('elu',**bn_args))

    model.add(Dense(units=512,**dense_args))
    model.add(Activation('elu',**bn_args))

    model.add(Dense(units=outdim,**dense_args)) 
    model.add(Activation('softmax',**bn_args)) 
    return model  
    
    
    
    



# here is a test 
# args structure:env_id
if __name__ == "__main__":

    logger.set_level(logger.WARN)

    env=gym.make('SpaceInvaders-v0') 
 
    ob_space=env.reset()
    action_space=env.action_space
    model=Network(ob_space,action_space)
    for i in range(10):
        print(model.get_layer(index=i))
    



