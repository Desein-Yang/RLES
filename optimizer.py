#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2019/10/10 22:12:35
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   rewrite the optimizer class of keras by Canonical Evolution Strategy (CES) and Natural Evolution Strategy (NES)
'''

# here put the import lib
#from torch import optim
from keras import optimizers
import logging 
import numpy as np
from numpy.random import RandomState,random_integers,randint
import math

# pytorch optimizer reference
# https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
# keras optimizer reference
# https://keras.io/optimizers/


class optimizer(object):
    def __init__(self,params,cfg):
        self.params=params
        self.lr=cfg["learning_rate"]

    #def set_weight(self):

    #def get_weight(self):

    # Sample gaussian noise to perturb params
    def sample(self):
        raise NotImplementedError

    # Compute policy gradient
    def get_grad(self):
        raise NotImplementedError

    # Update optimizer based on reward
    def get_update(self):
        #self.par=self.par+grad*self.rate
        raise NotImplementedError

    # get config of optimizer
    def get_config(self):
        raise NotImplementedError

    # Log basic and update infomation in optimization
    def logBasic(self,logger):
        raise NotImplementedError
    def logNorm(self,logger):
        raise NotImplementedError


class CES(optimizer):
    '''
    an optimizer based on canonical evolution strategy
    '''
    def __init__(self,params,cfg):
        self.params = params
        self.n = len(params)
        self.lr = cfg["learning rate"]
        self.mu = cfg["child popsize"]
        # self.lam = cfg["parents_popsize"] may be useless
        self.sigma = cfg["Mutation step"]
        self.discount = cfg["Discount"]
        # assert ( self.mu <= self.lam )

        # create shared nosie table
        self.noise_table = RandomState(123).randn(int(5e8)).astype('float32')

        # initialize weight of every parents as algorithm
        self.w = np.array([math.log(self.mu + 0.5) - math.log(i) for i in range(1, self.mu + 1)])
        self.w /= np.sum(self.w)
    
    
    def sample(self):
        '''return random start id'''
        return randint(0, len(self.noise_table)-self.n)

    def perturb(self , params):
        '''return pertubed params=(theta + sigma * noise)''' 
        rand_id = self.sample()
        epsilon = self.noise_table[rand_id:rand_id + self.n]
        params = self.params + self.lr * epsilon
        
        return params , rand_id

    def get_update(self,reward,randid):
        '''update params by reward\n no return,return an updated self.params '''
        step = np.zeros(len(self.params))
        # numbers of best n childs in reward
        best = np.array(reward).argsort()[::-1][:self.mu]

        for i in range(self.mu):
            rand_id = randid[best[i]]
            epsilon = self.noise_table[rand_id:rand_id + self.n]
            step += self.w * epsilon

        self.params += self.lr * step

    def logBasic(self):
        '''log some basic in log.txt header'''
        logger.info('=============Basic information===========')
        logger.info(msg='Dimension'.ljust(25) + '%f' % self.n)
        logger.info(msg='Learning rate'.ljust(25) + '%f' % self.lr)
        logger.info(msg='Poplation size'.ljust(25) + '%f' % self.mu)
        logger.info(msg='Optimizer'.ljust(25)+'CES')
        logger.info('=========================================')

    def log(self):
        '''log informations in every iteration'''
        logger.info(self.params)


    
# A simple reward function for test
def fun1(params):
    reward = 0
    for i in range(len(params)):
        if i%2 == 0:
            reward += np.sin(params[i])
        else:
            reward += np.cos(params[i])
    return reward

# here is a test function
if __name__ == "__main__":
    cfg={
        "learning_rate":0.1,
        "child_popsize":10,
        "parents_popsize":10,
        "discount":0.99
    }

    # create an instance of logger by module
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     filename='log.txt',
    #     filemode='w',
    #     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #     datefmt = '%Y-%m-%d  %H:%M:%S %a'
    # )
    logger=logging.getLogger(__name__)
    logger.setLevel('INFO')
    
    # make a 10 dim vector
    params=np.array(np.arange(10)).astype('float32')
    optimizer = NES(params,cfg)
    cumreward = 0
    iter_times = 0



    while (iter_times < 100):
        sample_episode=100
        reward = [0] * sample_episode
        randid = [0] * sample_episode
        for i in range(sample_episode):
            params , id = optimizer.perturb(params)
            reward[i] = fun1(params)
            randid[i] = id
        
        optimizer.get_update(reward,randid)
        iter_times += 1
        cumreward += fun1(params)
        logger.info(msg='Iteration'.ljust(25) + '%f' % iter_times)
        logger.info(msg='CumulativeReward'.ljust(25) + '%f' % cumreward)



    

    

class NES(optimizer):
    '''
    an optimizer based on natural evolution strategy
    '''
    def __init__(self,params,cfg):
        self.params = params
        self.n= len(params)
        self.lr = cfg["learning_rate"]
        self.lam = cfg["population_size"]
        self.sigma = cfg["Mutation size"]
        self.grad = np.zeros(param.shape)
        assert self.lam % 2 == 0,'Population config error'

        # create shared nosie table
        self.noise_table = RandomState(123).randn(int(5e8)).astype('float32')


    def sample(self):
        '''return random start id'''
        return randint(0, len(self.noise_table)-self.n)

    def perturb(self,lamb):
        '''return a mirrored sample'''
        rand_id = self.sample()
        epsilon = self.noise_table[rand_id:rand_id + self.n]
        if lamb % 2 == 0:
            params = self.params + self.lr * epsilon
        else:
            params = self.params - self.lr * epsilon
        return params,rand_id
        
    @staticmethod
    def rank(array):
        '''return an lam*1 array of rank of fitness\nhave tested'''
        # assert array.shape[0]==1,'Shape setting error'
        rank=np.ones(array.shape,int)# rank as [1,1,1,1,1]

        rank[array.argsort()]=np.arange(len(array))
        rank=(rank.astype(float)/(len(array)-1))-0.5    
        return rank

    # get estimated natural gradient (rather than calculated gradient)
    def get_grad(self,randid,rank):
        '''
        return natural gradient
        '''
        for i in range(self.lam):
            rand = randid[i]
            epsilon = self.noise_table[rand:rand + self.n]
            self.grad += rank[i] * epsilon
    def get_update(self):
        '''
        Input:params:paramter vector in t iteration
        grad:natural gradient
        Output:update:updates in t iteration
        '''
        pass

    def get_config(self):
        pass
