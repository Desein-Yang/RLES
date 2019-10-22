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
import agent as ag
import optimizer as op
import model
import logging
import gym
from gym import wrappers,logger,vector


def main():
    #read configure file or parse arguments
    args=parser.ParseArgs()
    cfg={
        "learning_rate":0.1,
        "activiation":None,
    }

    #save configure parameter #args:env,multiprocess,hyperparameter
    cpu=args.cpu # Bool type
    env_id=args.env_id # name-version

    #create envirnonment and recorder
    env=vector.make(env_id,1,asynchronous=False,wrappers=None)
    env=wrappers.Monitor(env,directory='/tmp/results',video_callble=None,force=False,resume=False,write_upon_reset=False,uid=None,mode=None)
    env.seed(0)
    
    #create policy/agent
    agent=ag.RandomAgent(env.action_space)
    
    #create optimizer and set hyperparameters
    model=model.MultiLayerPerc(cfg,env.ob_space,env.action_space)
    optimizer=op.ces(cfg)

    #set level of log information
    logger.set_level(logger.INFO)

    #initialize
    episode_count=100
    reward=0
    render=False
    done=False

    # rollout
    for i in range(episode_count):
        ob=env.reset()
        while True:
            action=agent.act(ob,reward,done)
            ob,reward,done=env.step(action)
            cum_reward+=reward
            if render and i%10==0:
                env.render()
            if done:
                break

    env.close()
