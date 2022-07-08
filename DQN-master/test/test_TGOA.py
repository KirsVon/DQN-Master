#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：test_TGOA.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/25 7:14 PM 
'''
import math
import time

from config import curr_config_class
from entity.car import Car
from entity.load_plan import LoadPlan
from entity.time_window import Time_Window
from entity.cargo import Cargo
import pandas as pd
from copy import deepcopy
import numpy as np
from tool.kuhn_munkras import kuhn_munkras
from datetime import datetime


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
        # temp_car.city = literal_eval(temp_car.city)[0] + '市'
        # temp_car.commodity = literal_eval(temp_car.commodity)[0]
        lp_list.append(temp_lp)
    return lp_list


sparsity = 1
cardinality = 1000

start_time = 0
next_time = 100
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
res = 0


def get_driver_frequency():
    driver_frequency = {}
    dp_df = pd.read_csv(curr_config_class.DRIVER_FREQUENCY)
    timezone = list(dp_df['timezone'])
    frequency = list(dp_df['frequency'])
    for u in range(len(timezone)):
        timezone[u] = str(timezone[u])
        if len(timezone[u]) < 12:
            num = 12 - len(timezone[u])
            for l in range(num):
                timezone[u] = '0' + timezone[u]
        driver_frequency[str(timezone[u][6:10])] = frequency[u]
    return driver_frequency


driver_frequency = get_driver_frequency()


def get_time_zone(start_time: int):
    minute = (start_time // 100) % 100
    while (minute != 0 and minute != 20 and minute != 40):
        minute -= 1
    hour = (start_time) // 10000
    if minute < 10:
        minute = "0" + str(minute)
    if hour < 10:
        hour = "0" + str(hour)
    return str(hour) + str(minute)

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

timezone = None
cnt = None
t = time.time()
while start_time <= 235959:
    start_time = time_add(start_time)
    if start_time >= 240000:
        break
    print(start_time)
    next_time = time_add(next_time)
    temp_tz = get_time_zone(start_time)
    next_tz = get_time_zone(next_time)
    flag = False
    k = 0
    while k < len(car_cache) and time_minus(start_time, car_cache[k].arrive_time) > 600:
        max_weight = 0
        max_index = -1
        for u in range(len(lp_pool)):
            if lp_list[u].city == car_cache[k].city:
                if max_weight < lp_list[u].load:
                    max_weight = lp_list[u].load
                    max_index = u
        if max_index != -1:
            del lp_pool[max_index]
            res += max_weight
        k += 1
    del car_cache[0:k]
    if temp_tz != next_tz:
        cnt = math.ceil(driver_frequency[temp_tz])
        flag = True
    while i < len(lp_list) and lp_list[i].arriving_time < start_time:
        lp_pool.append(lp_list[i])
        i += 1
    while j < len(car_list) and car_list[j].arrive_time < start_time:
        car_cache.append(car_list[j])
        j += 1

    if cnt and not flag:
        k = 0
        while k < len(car_cache):
            max_index = -1
            max_weight = 0
            for l in range(len(lp_pool)):
                if lp_pool[l].city == car_cache[k].city and lp_pool[l].commodity == car_cache[k].commodity:
                    if lp_pool[l].load > max_weight:
                        max_weight = lp_pool[l].load
                        max_index = l
            if max_index != -1:
                res += max_weight
                car_cache.pop(k)
                del lp_pool[max_index]
                break
            k += 1
    else:
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
        if (flag or time_range >= 10) and len(car_cache) and len(lp_pool) :
            tw = Time_Window()
            tw.car_list = deepcopy(car_cache)
            tw.can_be_sent_load_plan = deepcopy(lp_pool)
            matcher = kuhn_munkras(tw)
            rew, match = matcher.km()
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
            # lp_pool = deepcopy(tw.can_be_sent_load_plan)
            for u in unbound_lp_list:
                res += u.load
z = time.time()
print((int(round((z - t) * 1000))))
print(res)
