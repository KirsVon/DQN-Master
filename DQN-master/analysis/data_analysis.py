#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：data_analysis.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/23 3:45 PM 
'''
from datetime import datetime, timedelta
import pandas as pd
from config import curr_config_class
from ast import literal_eval


start_time = "20200924000000"
end_time = "20201014000000"
tmp_time = start_time

def time_add(first_time: str, td: timedelta):
    tmp = datetime.strptime(first_time, "%Y%m%d%H%M%S")
    tmp += td
    first_time = datetime.strftime(tmp, "%Y%m%d%H%M%S")
    return first_time

city_dict = {}
two_com = 0
total_com = 0
com_dict = {}
while tmp_time <= end_time:
    car_df = pd.read_csv(curr_config_class.CAR_DATA_ROOT_DIRECTORY + tmp_time + '.csv')
    city_list = list(car_df['city'])
    commodity_list = list(car_df['commodity'])
    for i in city_list:
        if i not in city_dict.keys():
            city_dict[i] = 1
        else:
            num = city_dict[i]
            num += 1
            city_dict[i] = num
    for i in commodity_list:
        commodity = literal_eval(i)
        if len(commodity) > 1:
            two_com += 1
        total_com += 1
        for j in commodity:
            if j == "型钢16以上":
                j = "型钢"
            if j not in com_dict:
                com_dict[j] = 1
            else:
                temp = com_dict[j]
                temp += 1
                com_dict[j] = temp
    tmp_time = time_add(tmp_time, timedelta(days=1))

commodity_df = pd.DataFrame(columns=["commodity","probability"])
city_df = pd.DataFrame(columns=["city", "probability"])
# prob_list = []
total_num = 0
for key in com_dict.keys():
    total_num += com_dict[key]
#
# num = 0
# for key in city_dict.keys():
#     num += city_dict[key]
#     prob_list.append(num / total_num)
#
# city_df["city"] = list(city_dict.keys())
# city_df["probability"] = list(prob_list)
#
# city_df.to_csv(curr_config_class.CITY_HIS)

# prob_list = []
# num = 0
# for key in com_dict.keys():
#     num += com_dict[key]
#     prob_list.append(num / total_num)
# commodity_df["commodity"] = list(com_dict.keys())
# commodity_df["probability"] = list(prob_list)
#
# commodity_df.to_csv(curr_config_class.COMMODITY_HIS)

print(two_com / total_com)