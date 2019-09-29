#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Evostra.py
@Time    :   2019/08/09 20:27:28
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   Define class of Evolution strategy Updater
'''

# here put the import lib
import numpy as np
#from keras.optimizers import SGD,adam
import multiprocessing as mp



class Evostra():
    '''
    Evolution strategy
    '''
    def __init__(self,learning_step_size,population,iter_times,theta_0):# turn to config later
        self.shape=len(theta_0)
        self.sigma=learning_step_size
        self.pop=population
        self.it=iter_times
        self.t0=theta_0

    # Sampling noise a shape tuple(3,4) 
    def Sampling(self):
        noise=np.zeros(self.shape)
        coviance=np.identity(self.shape)
        mean=np.zeros(self.shape)
        noise=np.random.multivariate_normal(mean,coviance,(self.shape,1)) # narray
        return noise

    # Evaluation F()
    def Evaluation(self,theta):
        '''
        Input: PolicyFun(x):evalution policy function from games
                noise: noise vector, theta.shape*1
                sigma: learning step size
                theta: theta in t iteration
        Output:F(x+noise) 
        '''
        sigma=np.matrix(self.sigma)
        fitness=np.zeros(self.pop)
        noiselist=[]
        assert self.pop%2==0,'Population config error'
        for i in range(0,self.pop):
            noise=self.Sampling()
            if (i%2==0):
                fitness[i]=PolicyFun(theta+noise*(sigma.T))
            else:
                fitness[i]=PolicyFun(theta-noise*sigma.T)
            noiselist.append(noise)
        noisearray=np.array(noiselist)# pop*theta.shape
        return fitness,noisearray

    # rank evalution of fitness
    def Rank(self,array):
        '''
        Input: array:array of fitness\n
        Output: ranks: a array of rank of fitness\n
        have tested
        '''
        assert array.shape[0]==1,'Shape setting error'
        rank=np.ones(array.shape,int)# rank as [1,1,1,1,1]
        # sorted_array=np.sort(array)
        rank[array.argsort()]=np.arange(len(array))
        # for index,i in enumerate(array):
            # for j in array:
                # if j>i:rank[index]=rank[index]+1
        rank=(rank.astype(float)/(len(array)-1))-0.5    
        return rank

    # Estimate gradient 
    def EstGrad(self,noise,rank):
        '''
        Input: rank pop*1 
        Output:grad:natural gradient\n
        '''
        cumgrad=np.zeros(noise.shape)
        for i in range(0,pop-1):
            cumgrad=cumgrad+rank[i-1]*noise[i-1]
        grad=cumgrad/(self.sigma*self.pop)
        return grad

    # Update parameter
    def Update(self,theta,grad):
        '''
        Input:theta:paramter vector in t iteration
        grad:natural gradient
        Output:theta:parameter vector in t iteration
        '''
        theta=theta+optimizer(grad)
        return theta

    def Run(self):
        '''
        Input:\n self.it itertimes\n 
        '''
        return self.sigma

def PolicyFun(x):
    '''
    policy function
    '''
    return x.cumsum()

def optimizer(x):
    '''
    optimizer function:function used to optimize gradient (SGD,Adam,...)in keras
    '''
    return x
    

if __name__ == "__main__":
    learning_step_size=[0.1,0.1]
    pop=10
    iter_times=3
    theta_0=[0.1,0.1]
    history=[]
    es=Evostra(learning_step_size,pop,iter_times,theta_0)

    t=1
    if t<iter_times:
        if t==1:theta=theta_0 
        fitness,noise=es.Evaluation(theta)
        rank=es.Rank(fitness)
        print(rank)
        print("nosie shape is",noise.shape)
        grad=es.EstGrad(noise,rank)
        print(grad)
        theta=es.Update(theta,grad)
        t+=1
        history.append(theta)
    print(history)
        
    
        

