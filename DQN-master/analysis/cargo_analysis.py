#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：cargo_analysis.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/23 7:46 PM 
'''
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
    cargo_df = pd.read_csv(curr_config_class.STOCK_DATA_ROOT_DIRECTORY + tmp_time[0:8] +'/' + tmp_time + '.csv')
    cargo_df.dropna(subset=['city', 'weight', 'commodity'], inplace=True)
    city_list = list(cargo_df['city'])
    weight_list = list(cargo_df['weight'])
    commodity_list = list(cargo_df['commodity'])
    for i in range(len(city_list)):
        if city_list[i] not in city_dict.keys():
            city_dict[city_list[i]] = weight_list[i]
        else:
            num = city_dict[city_list[i]]
            num += weight_list[i]
            city_dict[city_list[i]] = num
    for i in range(len(commodity_list)):
        if commodity_list[i] == "型钢16以上":
            commodity_list[i] = "型钢"
        if commodity_list[i] not in com_dict.keys():
            com_dict[commodity_list[i]] = weight_list[i]
        else:
            num = com_dict[commodity_list[i]]
            num += weight_list[i]
            com_dict[commodity_list[i]] = num
    tmp_time = time_add(tmp_time, timedelta(minutes=20))



commodity_df = pd.DataFrame(columns=["commodity","probability"])
city_df = pd.DataFrame(columns=["city", "probability"])
prob_list = []
total_num = 0
for key in city_dict.keys():
    total_num += city_dict[key]

num = 0
for key in city_dict.keys():
    num += city_dict[key]
    prob_list.append(num / total_num)
#

city_df["city"] = list(city_dict.keys())
city_df["probability"] = list(prob_list)
prob_list = []
total_num = 0
for key in com_dict.keys():
    total_num += com_dict[key]

num = 0
for key in com_dict.keys():
    num += com_dict[key]
    prob_list.append(num / total_num)
commodity_df["commodity"] = list(com_dict.keys())
commodity_df["probability"] = list(prob_list)
#
city_df.to_csv(curr_config_class.CARGO_CITY_HIS)
commodity_df.to_csv(curr_config_class.CARGO_COMMODITY_HIS)

# prob_list = []
# num = 0
# for key in com_dict.keys():
#     num += com_dict[key]
#     prob_list.append(num / total_num)
# commodity_df["commodity"] = list(com_dict.keys())
# commodity_df["probability"] = list(prob_list)
#
# commodity_df.to_csv(curr_config_class.COMMODITY_HIS)
