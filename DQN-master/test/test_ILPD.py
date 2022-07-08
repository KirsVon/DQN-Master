#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：vehicle-cargo matching.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/24 4:31 PM 
'''
import math
from datetime import datetime
from config import curr_config_class
from entity.car import Car
from entity.load_plan import LoadPlan
from entity.time_window import Time_Window
from entity.cargo import Cargo
import pandas as pd
from copy import deepcopy
import time
import numpy as np
from tool.kuhn_munkras import kuhn_munkras

q_table_path = "/Users/lalala/Desktop/Apweb2022/exp_res/gama/exp/q_value_10.csv"

def load_car_list(car_file_path):
    car_list = []
    car_train_data = pd.read_csv(car_file_path)
    records_dict_list = car_train_data.to_dict(orient="records")
    for i in records_dict_list:
        temp_car = Car()
        temp_car.set_attr(i)
        temp_car.set_commodity_list()
        temp_car.set_district_list()
        temp_car.set_city_list()
        temp_car.city += '市'
        # temp_car.commodity = literal_eval(temp_car.commodity)[0]
        car_list.append(temp_car)
    return car_list

def load_lp_list(lp_file_path):
    lp_list = []
    lp_train_data = pd.read_csv(lp_file_path)
    records_dict_list = lp_train_data.to_dict(orient="records")
    for i in records_dict_list:
        car = Car()
        temp_lp = LoadPlan(car)
        temp_lp.set_attr(i)
        lp_list.append(temp_lp)
    return lp_list

def load_q_table(q_path):
    q_df = pd.read_csv(q_path)
    q_table = np.random.rand(31, 301, 12 ,11)
    value_list = list(q_df['q_value'])
    for i in range(31):
        for j in range(301):
            for k in range(12):
                for l in range(11):
                    q_table[i][j][k][l] = value_list[i * 301 * 12 * 11 + j * 12 * 11 + k * 11 + l]
    return q_table



sparsity = 1
cardinality = 1000
start_time = 0
car_path = curr_config_class.SIMULATOR_CAR_DIRECTORY + "car_" + str(int(cardinality * sparsity * 0.8)) + ".csv"
lp_path = curr_config_class.SIMULATOR_LP_DIRECTORY + "lp_" + str(int(cardinality * sparsity * 0.8)) + ".csv"
car_df = pd.read_csv(car_path)
lp_df = pd.read_csv(lp_path)
car_list = load_car_list(car_path)
lp_list = load_lp_list(lp_path)
j = 0
duration = 1000
i = 0
lp_pool = []
car_cache = []

weight = 0
q_table = load_q_table(q_table_path)
cnt = 0
for u in range(len(lp_list)):
    lp_list[u].id = cnt
    cnt += 1
def time_add(start_time:int):
    minute = (start_time // 100) % 100
    minute += 1
    hour = (start_time) // 10000
    if minute == 60:
        minute = 0
        hour += 1
    return hour * 10000 + minute * 100

def time_minus(start_time, end_time):
    st = str(start_time)
    while len(st) < 6:
        st = "0" + st
    ed = str(end_time)
    while len(ed) < 6:
        ed = "0" + ed
    st = datetime.strptime(st, "%H%M%S")
    ed = datetime.strptime(ed, "%H%M%S")
    delta = st - ed
    return delta.seconds

t = time.time()
while start_time <= 235959:
    start_time = time_add(start_time)
    if start_time == 240000:
        z = time.time()
        print((int(round((z - t) * 1000))))
        break
    print(start_time)
    k = 0
    while k < len(car_cache) and time_minus(start_time, car_cache[k].arrive_time) > 600:
        k += 1
    del car_cache[0:k]
    while i < len(lp_list) and lp_list[i].arriving_time < start_time:
        lp_pool.append(lp_list[i])
        i += 1
    while j < len(car_list) and car_list[j].arrive_time < start_time:
        car_cache.append(car_list[j])
        j += 1
    time_range = None
    if len(car_cache) == 0:
        time_range = 0
    else:
        st = str(start_time)
        while len(st) < 6:
            st = "0" + st
        ed = str(car_cache[0].arrive_time)
        while len(ed) < 6:
            ed = "0" + ed
        st = datetime.strptime(st, "%H%M%S")
        ed = datetime.strptime(ed, "%H%M%S")
        delta = st - ed
        time_range = int(math.ceil(delta.seconds / 60))
    state = np.random.randint(10, size=4)
    time_period = start_time // 10000
    state[0] = len(car_cache)
    state[1] = len(lp_pool)
    state[2] = time_period // 2
    state[3] = time_range
    l = q_table[state[0]][state[1]][state[2]][state[3]]
    if l <= time_range and state[0] or time_range >= 10:
        tw = Time_Window()
        tw.car_list = deepcopy(car_cache)
        tw.can_be_sent_load_plan = deepcopy(lp_pool)
        matcher = kuhn_munkras(tw)
        res, match = matcher.km()
        unbound_lp_list = tw.node_clear(match)
        id_list = []
        for u in unbound_lp_list:
            if u.id not in id_list:
                id_list.append(u.id)
        car_cache = deepcopy(tw.car_list)
        temp_lps = []
        for u in range(len(lp_pool)):
            if lp_pool[u].id not in id_list:
                temp_lps.append(deepcopy(lp_pool[u]))
        lp_pool = deepcopy(temp_lps)
        for u in unbound_lp_list:
            weight += u.load
        print("匹配时刻: ", start_time, " 目前重量: ", weight)




