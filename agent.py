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
import sys
import gym
import numpy as np
from gym import wrappers,logger,envs
from keras.optimizers import SGD,Adam
import model 

test_envs={'algorithm':'Copy-v0',
           'toy_text':'FrozenLake-v0',#not successful
           'control':'CartPole-v0', # OK
           'atari':'SpaceInvaders-v0',# OK
          'mujoco':'Humanoid-v1',     # not successful
          'box2d':'LunarLander-v2' }  # not successful

# print env_id list
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
print(env_ids)

# here is a base class
class Agent():
    '''
    Define agent without policy
    act: output action 
    get_parameter: get parameter now
    set_parameter: set parameter 
    '''
    def __init__(self,env):
        self.env=env
        self.episode_count=10000

        self.ac_space=env.action_space
        self.ob_space=env.ob_space
        

    def act(self):
        '''input:observation,reward,doneflag\noutput:action i action_space'''
        raise NotImplementedError

    def get_parameter(self):
        pass

    def set_parameter(self,parameters):
        pass

    def rollout(self,render=False):
        '''run the agent and provide cumulative reward and iteration times'''
        ob = self.env.reset()
        ob = np.asarray(ob)
        reward=0
        done=False
        times = 0
        cum_rew=0
        for _ in range(self.episode_count):
            action = self.act(ob,reward,done)
            ob,reward,done,_ = self.env.step(np.argmax(action))
            ob = np.asarray(ob)
            cum_rew += reward
            times += 1
            if render:
                self.env.render('rgb_array')
            if done:
                break

        return cum_rew,times

#here is a random agent
class RandomAgent(Agent):
    '''create  a random agent'''
    def __init__(self,env):
        self.env=env
        self.episode_count=10000
        self.ac_space=env.action_space

    def act(self,observation):
        return self.ac_space.sample()

class EsAgent(Agent):
    def __init__(self,env,optimizer):
        self.env=env
        self.episode_count=10000

        self.ac_space=env.action_space
        self.ob_space=env.ob_space

        self.model=model.Network(self.ob_space,self.ac_space)
        self.optimizer=optimizer


    def act(self,observation):
        '''input:observation,reward,doneflag\noutput:action i action_space'''
        return self.policy.act(observation)

    def compile(self,optimizer,metrics):
        '''rewrite model.compile() to perform reinforcement learning'''



# here is a test 
# args structure:env_id
if __name__ == "__main__":
 
    logger.set_level(logger.INFO)

    env=gym.make('SpaceInvaders-v0') 


    agent=RandomAgent(env)

    optimizer=SGD(0.1,0)

    reward,times=agent.rollout(render=False)

    print(reward)
    print(times)
    env.close()

