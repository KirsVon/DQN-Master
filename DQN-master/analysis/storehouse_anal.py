#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：storehouse_anal.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/5/23 22:10 
'''
from config import curr_config_class
from entity.car import Car
from entity.load_plan import LoadPlan
from entity.cargo import Cargo
import pandas as pd
from datetime import datetime, timedelta
import time


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


def time_add(start_time: int):
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
    if start_time >= 240000:
        break
    print(start_time)
    k = 0
    while k < len(car_cache) and time_minus(start_time, car_cache[k].arrive_time) > 600:
        k += 1
    del car_cache[0:k]

    # k = 0
    # while k < len(lp_pool) and lp_pool[k].arriving_time + 1000 < start_time:
    #     k += 1
    # del lp_pool[0:k]
    while i < len(lp_list) and lp_list[i].arriving_time < start_time:
        lp_pool.append(lp_list[i])
        i += 1
    while j < len(car_list) and car_list[j].arrive_time < start_time:
        car_cache.append(car_list[j])
        j += 1
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
            del lp_pool[max_index]
        k += 1

print("res = ", res)
z = time.time()
print((int(round((z - t) * 1000))))


def datetime_to_str(time_zone: str):
    return datetime.strptime(time_zone, "%Y%m%D%H%M%S")


def str_to_datetime(timezone: datetime):
    return datetime.strftime(timezone, "%Y%m%D%H%M%S")


class gm:
    def __init__(self, start_date: str, end_date: str):
        self.cargo_list = []
        self.time_stamp = None
        self.current_car = None
        self.time_zone = start_date

    def load_cargo(self):

