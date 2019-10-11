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

class optimizer(object):
    def __init__(self,par,cfg):
        self.par=par
        self.rate=cfg["learning_rate"]

    def update(self,grad):
        self.par=self.par+grad*rate

