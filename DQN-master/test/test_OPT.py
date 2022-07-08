#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：test_OPT.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/25 3:33 PM 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：test_ILPD.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/24 4:31 PM 
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
        # temp_car.city = literal_eval(temp_car.city)[0] + '市'
        # temp_car.commodity = literal_eval(temp_car.commodity)[0]
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
t = time.time()
tw = Time_Window()
tw.car_list = deepcopy(car_list)
tw.can_be_sent_load_plan = deepcopy(lp_list)
matcher = kuhn_munkras(tw)
res, match = matcher.km()
unbound_lp_list = tw.node_clear(match)
# id_list = []
# for u in unbound_lp_list:
#     if u.id not in id_list:
#         id_list.append(u.id)
# car_cache = deepcopy(tw.car_list)
# temp_lps = []
# for u in range(len(lp_pool)):
#     if lp_pool[u].id not in id_list:
#         temp_lps.append(deepcopy(lp_pool[u]))
# lp_pool = deepcopy(temp_lps)
# lp_pool = deepcopy(tw.can_be_sent_load_plan)
for u in unbound_lp_list:
    weight += u.load
print("Weight = ", weight)
# print("匹配时刻: ", start_time, " 目前重量: ", weight)

z = time.time()
print((int(round((z - t) * 1000))))

# while start_time <= 235959:
#     start_time += 100
#     print(start_time)
#     k = 0
#     while k < len(car_cache) and car_cache[k].arrive_time + 1000 < start_time:
#         max_weight = 0
#         max_index = -1
#         for u in range(len(lp_list)):
#             if lp_list[u].city == car_cache[k].city:
#                 if max_weight < lp_list[u].load:
#                     max_weight = lp_list[u].load
#                     max_index = u
#         if max_index != -1:
#             del lp_pool[max_index]
#             weight += max_weight
#         k += 1
#     del car_cache[0:k]
#
#     # k = 0
#     # while k < len(lp_pool) and lp_pool[k].arriving_time + 1000 < start_time:
#     #     k += 1
#     # del lp_pool[0:k]
#     while i < len(lp_list) and lp_list[i].arriving_time < start_time:
#         lp_pool.append(lp_list[i])
#         i += 1
#     while j < len(car_list) and car_list[j].arrive_time < start_time:
#         car_cache.append(car_list[j])
#         j += 1
#     time_range = None
#     if len(car_cache) == 0:
#         time_range = 0
#     else:
#         time_range = int(math.ceil(start_time - car_cache[0].arrive_time) / 100)
#     state = np.random.randint(10, size=4)
#     time_period = start_time // 10000
#     state[0] = len(car_cache)
#     state[1] = len(lp_pool)
#     state[2] = time_period // 2
#     state[3] = time_range
#     l = q_table[state[0]][state[1]][state[2]][state[3]]
#     if l <= time_range and state[0] or time_range >= 10:
#         tw = Time_Window()
#         tw.car_list = deepcopy(car_cache)
#         tw.can_be_sent_load_plan = deepcopy(lp_pool)
#         matcher = kuhn_munkras(tw)
#         res, match = matcher.km()
#         unbound_lp_list = tw.node_clear(match)
#         id_list = []
#         for u in unbound_lp_list:
#             if u.id not in id_list:
#                 id_list.append(u.id)
#         car_cache = deepcopy(tw.car_list)
#         temp_lps = []
#         for u in range(len(lp_pool)):
#             if lp_pool[u].id not in id_list:
#                 temp_lps.append(deepcopy(lp_pool[u]))
#         lp_pool = deepcopy(temp_lps)
#         # lp_pool = deepcopy(tw.can_be_sent_load_plan)
#         for u in unbound_lp_list:
#             weight += u.load
#         print("匹配时刻: ", start_time, " 目前重量: ", weight)
#
# print(weight)




