#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Preprocess.py
@Time    :   2019/09/19 20:29:05
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   preprocess atari raw image into data
'''


# pytorch load and data process
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
import torch
import torchvision
import torchvision.transforms as transforms

# transform steps (mnih 2015 DQN)
# 1. maximize value
# 2. gray
# 3. resize 
# 4. tensor
Process=transforms.Compose(
    [transforms.Grayscale(),
    transforms.Resize((84,84)),
    transforms.ToTensor()
    ])




