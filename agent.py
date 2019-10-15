#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   agent.py
@Time    :   2019/08/09 20:29:05
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   Define classes of agent with different policy
'''

# here put the import lib
import argparse
import pip
import sys
import gym
from gym import wrappers,logger

# here is a base class
class Agent():
    '''
    Define agent without policy
    act: output action 
    get_parameter: get parameter now
    set_parameter: set parameter 
    '''
    def __init__(self,action_space):
        self.action_space=action_space

    def act(self):
        '''input:observation,reward,doneflag\noutput:action i action_space'''
        raise NotImplementedError

    def get_parameter(self):
        return self.par

    def set_parameter(self,newpar):
        self.par=newpar

#here is a random agent
class RandomAgent(Agent):
    '''create  a random agent'''
    def __init__(self,action_space):
        self.action_space=action_space

    def act(self,observation,reward,done):
        return self.action_space.sample()

# write other agent
class CESAgent(Agent):
    def __init__(self,model,action_space):
        self.model=model
        self.action_space=action_space

    def act(self,observation,reward,done):
        return self.action_space.sample()

# here is a test 
# args structure:env_id
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=None)# command line parser into python
    parser.add_argument('env_id',nargs='?',default='SpaceInvaders-v0',help='select the environment')# add help informations to add argument
    args=parser.parse_args()

    logger.set_level(logger.INFO)

    env=gym.make(args.env_id) 
    outdir='./tmp/results'
    env=wrappers.Monitor(env,directory=outdir,force=True)# run environment
    env.seed(0)
    agent=RandomAgent(env.action_space)#configure agent

    # config
    episode_count=100
    reward=0
    done=False

    for i in range(episode_count):
        ob=env.reset()#observe environment
        while True:
            action=agent.act(ob,reward,done)#make action
            ob,reward,done,_=env.step(action)
            env.render('rgb_array')
            if done:
                break
            
    env.close()

