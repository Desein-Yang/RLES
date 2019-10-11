#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/10/10 22:28:59
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   None
'''

# here put the import lib
import parser
import gym

def main():
    #read configure file or parse arguments
    args=parser.ParseArgs()

    #save configure parameter #args:env,multiprocess,hyperparameter
    cpu=args.cpu # Bool type
    env_id=args.env_id
    
    #create envirnonment
    env=gym.make(env_id)

    #create policy

    #get policy parameter
    
    #create optimizer

    #set hyperparameters
    
    #log information to debug
    logger.set_level(logger.INFO)
    
    while True:
        #create episodes (length,reward,noise index)
        
        for every cpu:
            #get parameter from optimizer
            #set parameter of policy
            #get length and reward from environment
            #save episode in buffer
            #evaluate mean and max reward
            
        #flatten episode information
        #update parameter of optimizer
        #count step num
        
        #write log of parameter
        #visualize by replay information


