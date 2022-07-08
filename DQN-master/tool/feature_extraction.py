#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：feature_extraction.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/2/14 9:20 PM 
'''
import threading
import numpy as np

import pandas as pd


class FeatureExtraction:
    """
    时间窗内货物和司机的特征提取：
        1. 司机、货物的材料one-hot编码
        2. 司机、货物城的市one-hot编码
        3. 所在当天时间区间、所在星期位置
    """

    def __init__(self, ):