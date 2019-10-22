#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parser.py
@Time    :   2019/10/11 11:27:11
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   It is a argument parser in command line to get args. 
This module have been tested and successful!
'''

# here put the import lib

import argparse

def ParseArgs():
    '''parse argument function'''
    # command line parser into python
    parser=argparse.ArgumentParser(description='Please input enviroment')
    # add help informations to add argument
    parser.add_argument('--env_id',nargs='?',default='CartPole-v0',help='select the environment')
    parser.add_argument('--cpu',nargs='?',default='True',help='select the hardware')
    parser.add_argument('--policy',nargs='?',default='Random',help='select the policy')
    
    # parse args
    args=parser.parse_args()
    return args

#test
if __name__ == "__main__":
    args=ParseArgs()
    print (args.env_id)
    print (args.cpu)
    print (args.policy)


