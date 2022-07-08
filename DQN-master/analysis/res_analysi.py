#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：res_analysi.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/20 3:29 PM 
'''

import pandas as pd
import math

df = pd.DataFrame(columns=['len5', 'len10', 'len15', 'len20'])
len5 = pd.read_csv('/Users/lalala/Desktop/Apweb2022/exp_res/tw/len5/reward_10.csv', index_col=False)
len10 = pd.read_csv('/Users/lalala/Desktop/Apweb2022/exp_res/gama/exp/reward_10.csv', index_col=False)
len15 = pd.read_csv('/Users/lalala/Desktop/Apweb2022/exp_res/tw/len15/reward_10.csv', index_col=False)
len20 = pd.read_csv('/Users/lalala/Desktop/Apweb2022/exp_res/tw/len20/reward_10.csv', index_col=False)

res = 0
len_5_list = []
for index, row in len5.iterrows():
    res += row['reward']
    len_5_list.append(res)

res = 0
len_10_list = []
for index, row in len10.iterrows():
    res += row['reward']
    len_10_list.append(res)

res = 0
len_15_list = []
for index, row in len15.iterrows():
    res += row['reward']
    len_15_list.append(res)

res = 0
len_20_list = []
for index, row in len20.iterrows():
    res += row['reward']
    len_20_list.append(res)

df['len5'] = len_5_list
df['len10'] = len_10_list
df['len15'] = len_15_list
df['len20'] = len_20_list

df.to_csv('/Users/lalala/Desktop/Apweb2022/exp_res/tw/tw.csv',index=False)


