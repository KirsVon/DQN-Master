#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：test.py.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/4/2 9:49 PM 
'''
for i in range(1,10):
    for k in range(1, i):
        print("\t", end="")
    for j in range(i,10):
        print("%s*%s=%2s" % (i, j, i*j), end="\t")
    print("\n" )