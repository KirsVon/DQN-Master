#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：DataLoader.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/2/22 8:27 PM 
'''
import config
from datetime import datetime, timedelta
from copy import copy
import pandas as pd
from entity.car import Car
from entity.load_plan import LoadPlan
from typing import List
from config import curr_config_class
from tool.cargo_maintain import cargo_management
from tool.dp_cargoes import dp
from ast import literal_eval

CITY_MEANING = {
    "泰安": 1,
    "济宁": 2,
    "日照": 3,
    "淄博": 4,
    "济南": 5,
    "临沂": 6,
    "潍坊": 7,
    "青岛": 8,
    "烟台": 9,
    "威海": 10,
    "德州": 11,
    "滨州": 12,
    "莱芜": 13,
    "东营": 14,
    "聊城": 15,
    "菏泽": 16,
    "枣庄": 17
}


class DataLoader:
    """
        实时加载车辆数据和货物数据
        1. 实时未处理的车辆数据
        2. 实时未处理的货物数据
    """

    def __init__(self, car_date: str, end_date: str):
        """
        初始化维护:
            1. 所有车辆列表
            2. 所有货物列表
            3. 当前货物列表所打包成的装载计划列表
            4. 下一个车辆
        """
        self.__start_date = car_date
        self.__cur_date__ = car_date  # 当前训练日期
        self.__end_date__ = end_date  # 结束训练日期
        self.__car__ = None  # 当前车辆
        self.__cargo_time = self.__cur_date__ # 货物更新时间
        self.__cur_time = self.__cur_date__  # 当前训练时间
        self.car_list = []  # 当前训练日期所有车辆
        self.__cargo_list = []  # 当前库存货物
        self.__can_load_cargos = []  # 当前可发库存货物
        self.__load_plan_list = []  # 当前货物生成的装载计划
        self.__car_index = -1  # 当前车辆的index
        self.__next_car_index = 0
        self.__next_car = None
        self.time = car_date
        self.just_update = False
        self.load_car_list(curr_config_class.CAR_DATA_ROOT_DIRECTORY, self.__cur_date__)

    def load_car_list(self, car_file_path, time_str):
        self.car_list = []
        car_train_data = pd.read_csv(car_file_path + time_str + ".csv")
        records_dict_list = car_train_data.to_dict(orient="records")
        for i in records_dict_list:
            temp_car = Car()
            temp_car.set_attr(i)
            temp_car.set_commodity_list()
            temp_car.set_district_list()
            temp_car.set_city_list()
            temp_car.city = literal_eval(temp_car.city)[0] + '市'
            temp_car.commodity = literal_eval(temp_car.commodity)[0]
            self.car_list.append(temp_car)

    def DLInterface(self, time_str: str, end_str: str):
        data_loader = self.__init__(time_str, end_str)
        return data_loader

    def load_cargo(self, time_str):
        # 加载当前时间货物
        cargo_management.init_cargo_dic(time_str)
        self.__cargo_list = cargo_management.cargo_all()
        # self.__load_plan_list = knapsack_algorithm_rule()

    def update_cargo(self):
        self.load_cargo(self.__cargo_time)
        tmp = datetime.strptime(self.__cargo_time, "%Y%m%d%H%M%S")
        tmp += timedelta(minutes=20)
        self.__cargo_time = datetime.strftime(tmp, "%Y%m%d%H%M%S")

    def get_next_car(self):
        self.__car_index += 1
        self.__next_car_index += 1
        self.__car__ = self.car_list[self.__car_index]
        self.__next_car = self.car_list[self.__next_car_index]

    # def get_next_node(self):
    #     """
    #         1. 更新新到车辆以及对应货物
    #     :return: 1. 最新车辆 2. 新加入货物
    #     """
    #     self.get_next_car()
    #     return self.__car__, self.__cargo_list

    def screen_car_cargo(self):
        self.__can_load_cargos = []  # 清空之前货物
        for i in self.__cargo_list:
            if self.__car__.city == i.city:
                self.__can_load_cargos.append(i)

    def is_episode_over(self):
        if self.__cur_time[0:8] == self.__end_date__[0:8] and self.__car_index >= len(self.car_list):
            return True
        else:
            return False



    def __get_obs(self):
        '''
            状态表示：(成单数，半成单半尾货数，尾单数，需求重量)
            成单和半成单判别：动态规划
            尾单判断：小于29吨
        :return:
        '''
        self.screen_car_cargo()
        num_of_total = 0  # 成单数量
        num_of_half = 0  # 半成单数量
        num_of_tail = 0  # 尾单数量
        for i in self.__can_load_cargos:
            if i.c_weight < 29:
                num_of_tail += 1
            elif i.c_weight >= 29 and i.c_weight <= 36:
                num_of_total += 1
            else:
                load_plan_list, tail_cargo_list = dp(copy(i))
                if len(tail_cargo_list) == 0:
                    num_of_total += 1
                else:
                    num_of_half += 1

    def time_add(self, time_str: str, td: timedelta):
        tmp = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        tmp += td
        time_str = datetime.strftime(tmp, "%Y%m%d%H%M%S")
        return time_str

    """
        Description: 标记已经发运走的货物
    """
    def drop_sent_load_plan(self, unbound_lp_list: List[LoadPlan]):
        for i in unbound_lp_list:
            cargo_management.add_status(i)

    @property
    def get_cargo(self):
        return self.__cargo_list

    @property
    def get_time(self):
        return self.__cur_time

    @property
    def get_next(self):
        self.__cur_time = self.time_add(self.__cur_time, timedelta(minutes=1))
        self.just_update = False
        car = None
        try:
            index = self.__next_car_index
            if self.__cur_time >= self.car_list[index].arrive_time:
                car = self.get_car()
            if self.__cur_time >= self.__cargo_time:
                self.update_cargo()
                self.just_update = True
        except IndexError:
            print("当天训练结束, 执行下一天训练")
            cargo_management.outbound = {}
            cargo_management.cargo_dic = {}
            self.__cur_date__ = self.time_add(self.__cur_date__, timedelta(days=1))
            self.load_car_list(curr_config_class.CAR_DATA_ROOT_DIRECTORY, self.__cur_date__)
            self.__cur_time = self.__cur_date__
            self.__car_index = -1
            self.__next_car_index = 0
            if self.__cur_time >= self.car_list[self.__next_car_index].arrive_time:
                car = self.get_car()
        return car

    def get_car(self):
        self.get_next_car()
        return self.__car__

    def reset(self):
        cargo_management.outbound = {}
        cargo_management.cargo_dic = {}
        cargo_management.init_cargo_dic(self.__start_date)
        self.__init__(self.__start_date, self.__end_date__)

data_loader = DataLoader("20200924000000", "20201014000000")
