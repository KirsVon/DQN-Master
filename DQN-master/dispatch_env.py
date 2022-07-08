#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：dispatch_env.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/1 9:04 PM 
'''
import gym
import numpy as np
from gym import spaces, error
from gym import utils
from gym.utils import seeding


class DispatchEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)




