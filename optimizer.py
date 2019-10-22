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
from numpy.random import RandomState,random_integers

# pytorch optimizer reference
# https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
# keras optimizer reference
# https://keras.io/optimizers/



# create an instance of logger by module
logging.basicConfig(
    level=logging.DEBUG,
    filename='log.txt',
    filemode='w',
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt = '%Y-%m-%d  %H:%M:%S %a'
)
logger=logging.getLogger(__name__)
logger.setLevel('INFO')


class optimizer(object):
    def __init__(self,params,cfg):
        self.params=params
        self.lr=cfg["learning_rate"]

    #def set_weight(self):

    #def get_weight(self):

    #sample gaussian noise to perturb params
    def sample(self):
        noise_table = RandomState(123).randn(int(5e8)).astype('float32')
        return random_integers(0, len(noise_table)-len(self.params))

    def get_grad(self):
        raise NotImplementedError

    def get_update(self):
        #self.par=self.par+grad*self.rate
        raise NotImplementedError

    def get_config(self):
        '''get config parameters of this optimizer'''
        config ={
            "learning_rate":self.lr
        }
        return dict(list(config.items())


class CES(optimizer):
    pass

class CES(optimizer):
    '''
    an optimizer based on canonical evolution strategy
    '''
    def __init__(self,params,cfg):
        self.params=params
        self.lr=cfg["learning_rate"]
        self.mu=cfg["child_popsize"]
        self.lam=cfg["parents_popsize"]
        assert(self.mu <= self.lam)

        self.w = np.array([np.log(self.u + 0.5) - np.log(i) for i in range(1, self.u + 1)])
        self.w /= np.sum(self.w)
    def _evaluation(self):
        
        return reward

    def get_update(self):
        step=np.zeros_like(len(self.params))
        for i in range(self.mu):
            step+=

class NES(optimizer):
    '''
    an optimizer based on natural evolution strategy
    '''
    def __init__(self,params,cfg):
        self.lr=cfg["learning_rate"]
        self.itertimes=cfg["itertimes"]
        self.popsize=cfg["population_size"]
        self.shape=params.shape()
    
    # sample gaussian noise to perturb params  
    # def _sample(self):
    #     noise=np.zeros(self.shape)
    #     coviance=np.identity(self.shape)
    #     mean=np.zeros(self.shape)
    #     noise=np.random.multivariate_normal(mean,coviance,(self.shape,1)) # narray
    #     return noise
    
    # evaluate the fitness to assign the credit in the next iteration
    def _evaluation(self,params):
        '''
        Input: PolicyFun(x):evalution policy function from games
                noise: noise vector, theta.shape*1
                sigma: learning step size
                params: parameters in t iteration
        Output:F(x+noise) 
        '''
        sigma=np.matrix(self.sigma)
        fitness=np.zeros(self.popsize)
        noiselist=[]
        assert self.popsize%2==0,'Population config error'
        for i in range(0,self.popsize):
            noise=self.Sampling()
            if (i%2==0):
                fitness[i]=PolicyFun(theta+noise*(sigma.T))
            else:
                fitness[i]=PolicyFun(theta-noise*sigma.T)
            noiselist.append(noise)
        noisearray=np.array(noiselist)# popsize*theta.shape
        return fitness,noisearray

    # rank evalution of fitness to avoid influence of the scale of value
    def _rank(self,array):
        '''
        Input: array:array of fitness\n
        Output: ranks: a array of rank of fitness\nhave tested
        '''
        assert array.shape[0]==1,'Shape setting error'
        rank=np.ones(array.shape,int)# rank as [1,1,1,1,1]
        rank[array.argsort()]=np.arange(len(array))
        rank=(rank.astype(float)/(len(array)-1))-0.5    
        return rank

    # get estimated natural gradient (rather than calculated gradient)
    def get_grad(self，noise,rank):
        '''
        Input: rank popsize*1 
        Output:grad:natural gradient\n
        '''
        cumgrad=np.zeros(noise.shape)
        for i in range(0,popsize-1):
            cumgrad=cumgrad+rank[i-1]*noise[i-1]
        grad=cumgrad/(self.sigma*self.popsize)
        return grad
        

    def get_update(self):
        '''
        Input:params:paramter vector in t iteration
        grad:natural gradient
        Output:update:updates in t iteration
        '''
        pass

    def get_config(self):
        pass