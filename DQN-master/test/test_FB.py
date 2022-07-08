#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：test_FB.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/25 3:18 PM 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：test_ga.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/24 11:10 AM 
'''
from config import curr_config_class
from entity.car import Car
from datetime import datetime
from entity.load_plan import LoadPlan
from entity.time_window import Time_Window
from entity.cargo import Cargo
import pandas as pd
from copy import deepcopy
import numpy as np
from tool.kuhn_munkras import kuhn_munkras
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

car_path = curr_config_class.SIMULATOR_CAR_DIRECTORY + "car_" + str(int(cardinality * sparsity * 0.4)) + ".csv"
lp_path = curr_config_class.SIMULATOR_LP_DIRECTORY + "lp_" + str(int(cardinality * sparsity * 0.4)) + ".csv"
car_df = pd.read_csv(car_path)
lp_df = pd.read_csv(lp_path)
car_list = load_car_list(car_path)
lp_list = load_lp_list(lp_path)
j = 0
duration = 1000
i = 0
lp_pool = []
car_cache = []
cnt = 0
for u in range(len(lp_list)):
    lp_list[u].id = cnt
    cnt += 1

res = 0
weight = 0

def time_add(start_time:int):
    minute = (start_time // 100) % 100
    minute += 10
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
    # k = 0
    # while k < len(car_cache):
    #     max_index = -1
    #     max_weight = 0
    #     for l in range(len(lp_pool)):
    #         if lp_pool[l].city == car_cache[k].city and lp_pool[l].commodity == car_cache[k].commodity:
    #             if lp_pool[l].load > max_weight:
    #                 max_weight = lp_pool[l].load
    #                 max_index = l
    #     if max_index != -1:
    #         res += max_weight
    #         del lp_pool[max_index]
    #     k += 1
    if len(car_cache) != 0 and len(lp_pool) != 0:
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
        # lp_pool = deepcopy(tw.can_be_sent_load_plan)
        for u in unbound_lp_list:
            weight += u.load
        print("匹配时刻: ", start_time, " 目前重量: ", weight)

print("res = ", weight)
z = time.time()
print((int(round((z - t) * 1000))))

# if __name__ == '__main__':
#     print(car_list)