#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
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

import argparse
from gym import wrappers,logger
import gym

cfg={
        "hid_sizes":64,
        "activation":'tanh'
    }

def MultiLayerPerc(cfg,ob_space,action_space):
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
        
    if isinstance(action_space, Box):
        outdim=action_space.shape[0]
        model.add(Dense(outdim))
        model.add(ConcatFixedStd())
    elif isinstance(action_space,Discrete):
        outdim=action_space.n
        model.add(Dense(outdim, activation="softmax"))
    return model

# here is a test 
# args structure:env_id
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=None)# command line parser into python
    parser.add_argument('env_id',nargs='?',default='SpaceInvaders-v0',help='select the environment')# add help informations to add argument
    args=parser.parse_args()

    logger.set_level(logger.INFO)

    env=gym.make(args.env_id) 
    outdir='./tmp/results'
    env=wrappers.Monitor(env,directory=outdir,force=True)
 
    ob_space=env.ob_space
    action_space=env.action_space
    model=MultiLayerPerc(cfg,ob_space,action_space)
    print(model)
    



