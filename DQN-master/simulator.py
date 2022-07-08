#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：simulator.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/19 3:25 PM 
'''
import math

import numpy as np
import pandas as pd
from random import random
from config import curr_config_class


# 车辆到达分布的模拟器
class Simulator:
    '''
        初始化分布， 基数以及稀疏度，车辆时间
    '''

    def __init__(self, cardinality, sparsity, flag: int):
        self.cardinality = cardinality
        self.sparsity = sparsity
        self.distribution = None
        self.time_list = []
        self.car_data = pd.DataFrame(columns=['city', 'arrive_time', 'commodity'])
        self.cargo_data = pd.DataFrame(columns=['city', 'arriving_time', 'commodity', 'load'])
        self.flag = flag
        self.mu = 12
        self.sigma = 4
        self.lam = 10
        if flag == 1:
            # normal distribution
            self.distribution = np.random.normal(loc=12, scale=1, size=int(self.cardinality * self.sparsity))
        elif flag == 2:
            # poisson distribution
            self.distribution = np.random.poisson(10, size=int(self.cardinality * self.sparsity))
        else:
            # square distribution
            self.distribution = np.random.uniform(0, 24, size=int(self.cardinality * self.sparsity))

    def func_to_time(self):
        self.time_list = []
        if self.flag == 1:
            # normal distribution
            self.distribution = np.random.normal(loc=12, scale=1, size=int(self.cardinality * self.sparsity))
        elif self.flag == 2:
            # poisson distribution
            self.distribution = np.random.poisson(10, size=int(self.cardinality * self.sparsity))
        else:
            # square distribution
            self.distribution = np.random.uniform(0, 24, size=int(self.cardinality * self.sparsity))
        self.distribution.sort()
        for i in range(len(self.distribution)):
            self.distribution[i] = self.distribution[i] * 10000
            seconds = self.distribution[i] % 100
            seconds = int(math.floor(60 * (seconds / 100)))
            minutes = (self.distribution[i] // 100) % 100
            minutes = int(math.floor(60 * (minutes / 100)))
            if seconds > 60 or minutes > 60:
                print(2)
                print(1)
            res = (self.distribution[i] // 10000) * 10000 + minutes * 100 + seconds
            self.distribution[i] = res
            if res >= 240000:
                res -= 240000
            print(res)
            self.time_list.append(abs(int(res)))
            self.time_list.sort()

    def generate_car_data(self):
        self.func_to_time()
        car_df = pd.read_csv(curr_config_class.CITY_HIS)
        commodity_df = pd.read_csv(curr_config_class.COMMODITY_HIS)
        city_list = list(car_df['city'])
        city_pro = list(car_df['probability'])
        commodity_list = list(commodity_df['commodity'])
        commodity_pro = list(commodity_df['probability'])
        cities = []
        commodities = []
        for i in range(len(self.distribution)):
            a = random()
            j = 0
            city = None
            while j < len(city_pro):
                if city_pro[j] >= a:
                    city = city_list[j]
                    cities.append(city)
                    break
                j += 1
            b = random()
            j = 0
            commodity = None
            while j < len(commodity_pro):
                if commodity_pro[j] >= b:
                    commodity = commodity_list[j]
                    commodities.append(commodity)
                    break
                j += 1
        self.car_data['city'] = cities
        self.car_data['arrive_time'] = self.time_list
        self.car_data['commodity'] = commodities
        self.car_data.to_csv(
            curr_config_class.SIMULATOR_CAR_DIRECTORY + 'car_' + str(int(self.cardinality * self.sparsity)) + '.csv',
            index=False)

    def generate_lp_data(self):
        self.func_to_time()
        city_df = pd.read_csv(curr_config_class.CARGO_CITY_HIS)
        commodity_df = pd.read_csv(curr_config_class.CARGO_COMMODITY_HIS)
        city_list = list(city_df['city'])
        city_pro = list(city_df['probability'])
        commodity_list = list(commodity_df['commodity'])
        commodity_pro = list(commodity_df['probability'])
        cities = []
        commodities = []
        load_list = []
        weight_pro = [0.02, 0.08, 0.24, 0.42, 0.65, 0.83, 0.92, 1]
        weight_cho = [29, 30, 31, 32, 33, 34, 35, 36]
        for i in range(len(self.distribution)):
            a = random()
            j = 0
            city = None
            while j < len(city_pro):
                if city_pro[j] >= a:
                    city = city_list[j]
                    cities.append(city)
                    break
                j += 1
            b = random()
            j = 0
            commodity = None
            while j < len(commodity_pro):
                if commodity_pro[j] >= b:
                    commodity = commodity_list[j]
                    commodities.append(commodity)
                    break
                j += 1
            c = random()
            j = 0
            load = None
            while j < len(weight_pro):
                if weight_pro[j] >= c:
                    load = weight_cho[j]
                    load_list.append(load)
                    break
                j += 1
        self.cargo_data['city'] = cities
        self.cargo_data['arriving_time'] = self.time_list
        self.cargo_data['commodity'] = commodities
        self.cargo_data['load'] = load_list
        self.cargo_data.to_csv(
            curr_config_class.SIMULATOR_LP_DIRECTORY + 'lp_' + str(int(self.cardinality * self.sparsity)) + '.csv',
            index=False)


if __name__ == '__main__':
    si = Simulator(1000, 10, 3)
    si.generate_car_data()
    si.generate_lp_data()

