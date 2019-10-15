#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2019/10/10 22:12:35
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   None
'''

# here put the import lib
from torch import optim


# pytorch optimizer reference
# https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/

class optimizer(object):
    def __init__(self,par,cfg):
        self.par=par
        self.rate=cfg["learning_rate"]

    def update(self,grad):
        #self.par=self.par+grad*self.rate
        raise NotImplementedError

class ces(optimizer):
    '''an optimizer based on canonical evolution strategy
    '''
    def __init__(self,cfg):
        self.rate=cfg["learning_rate"]

    def update(self):
        pass

class nes(optimizer):
    '''an optimizer based on natural evolution strategy
    '''
    def __init__(self,par,cfg):
        self.par=par
        self.rate=cfg["learning_rate"]

    def update(self):
        pass
