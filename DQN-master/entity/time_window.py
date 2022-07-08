# -*- coding: utf-8 -*-
# Description:车辆信息数据类
# Created: fengchong  2020/12/22
import operator

import config
from entity.car import Car
from entity.load_plan import LoadPlan
from typing import List
# from tool.DataLoader import data_loader
import pandas as pd
from operator import itemgetter, attrgetter
from tool.knapsack_algorithm_rule import knapsack_algorithm_rule
from tool.packaging import packaging
from config import curr_config_class


class Time_Window:
    """
        Description:初始化批内容：批的car_list、load_plan_list的内容为空
    """

    def __init__(self):
        self.car_list = []
        self.cargo_list = []
        self.load_plan_id = 0  # load_plan_id 唯一
        self.can_load_cargo_dict = {}
        self.shipping_destination = []
        self.shipping_destination_dict = {}
        self.load_plan_list = []
        self.can_be_sent_load_plan = []
        self.__city_load_plan_dict = dict()
        # self.matcher = kuhn_munkras(self)  # 循环调用
        self.matched_pool = set()
        # self.time = data_loader.get_time
        self.match_res = pd.DataFrame(columns=['car_mark', 'load', 'time'])
        self.match_car_list = []
        self.match_car_load = []
        self.match_car_time = []
        # self.end_time = data_loader.__end_date__


    '''
        Description: sort load plan by load and city
    '''

    def sort_load_plan(self):
        cmpfunc = operator.attrgetter('city', 'load')
        self.load_plan_list.sort(key=cmpfunc, reverse=True)
        # a = sorted(self.load_plan_list, key=attrgetter('city'))
        # a = sorted(a, key=attrgetter('load'), reverse=True)
        # return a

    def sort_load_plan_by_load(self):
        cmpfunc = operator.attrgetter('load')
        self.load_plan_list.sort(key=cmpfunc, reverse=True)

    '''
        Description: 为每个车提取前10个load_plan
    '''

    def cal_can_be_sent_load_plan(self):
        self.can_be_sent_load_plan = []
        num = 0
        city = ''
        for i in range(len(self.load_plan_list)):
            if self.load_plan_list[i].id not in self.matched_pool:
                if city != self.load_plan_list[i].city:
                    city = self.load_plan_list[i].city
                    if city in self.shipping_destination_dict.keys():
                        num = 1
                        self.can_be_sent_load_plan.append(self.load_plan_list[i])
                else:
                    if city in self.shipping_destination_dict.keys() and num < 10 * self.shipping_destination_dict[city]:
                        self.can_be_sent_load_plan.append(self.load_plan_list[i])
                        num += 1

    '''
        Description: 下一分钟
    '''
    # def get_next_min(self):
    #     car = data_loader.get_next
    #     self.time = data_loader.get_time
    #     print(self.time)
    #     if data_loader.just_update:
    #         self.cargo_list = data_loader.get_cargo
    #         self.load_plan_list = packaging(self.cargo_list)
    #         self.sort_load_plan_by_load()
    #         cnt = 0
    #         for i in range(len(self.load_plan_list)):
    #             if self.load_plan_list[i].load < 29:
    #                 cnt = i
    #                 break
    #         self.load_plan_list = self.load_plan_list[0:cnt]
    #         for i in self.load_plan_list:
    #             i.id = self.load_plan_id
    #             self.load_plan_id += 1
    #         self.sort_load_plan()
    #     if car is not None:
    #         self.car_list.append(car)
    #         self.shipping_destination = []
    #         self.shipping_destination_dict = {}
    #         for car in self.car_list:
    #             if car.city + '市' not in self.shipping_destination:  # 加入运输终点
    #                 self.shipping_destination.append(car.city)
    #                 self.shipping_destination_dict[car.city] = 1
    #             else:
    #                 num = self.shipping_destination_dict[car.city]
    #                 num += 1
    #                 self.shipping_destination_dict[car.city] = num
    #         self.cal_can_be_sent_load_plan()

    """
        Description: 匹配后节点清除
    """

    def node_clear(self, match_list):
        nl = len(self.car_list)
        nr = len(self.can_be_sent_load_plan)
        car_list = []
        load_plan_list = []
        unbound_lp_list = []
        if nl < nr:
            car_set = set()
            lp_set = set()
            for i in range(nr):
                if match_list[i] < nl and match_list[i] >= 0:
                    car_set.add(match_list[i])
                    lp_set.add(i)
            for i in range(nl):
                if i not in car_set:
                    car_list.append(self.car_list[i])
            for i in range(nr):
                if i not in lp_set:
                    load_plan_list.append(self.can_be_sent_load_plan[i])
                else:
                    self.can_be_sent_load_plan[i].car = self.car_list[match_list[i]]
                    unbound_lp_list.append(self.can_be_sent_load_plan[i])
                    # self.store_match_res(self.can_be_sent_load_plan[i].car.license_plate_number,
                    #                      self.can_be_sent_load_plan[i].load)
                    # print(self.can_be_sent_load_plan[i].car.license_plate_number, self.can_be_sent_load_plan[i].load,
                    #       self.time)
        else:
            car_set = set()
            lp_set = set()
            for i in range(nr):
                if match_list[i] < nl and match_list[i] >= 0:
                    car_set.add(match_list[i])
                    lp_set.add(i)
            for i in range(nl):
                if i not in car_set:
                    car_list.append(self.car_list[i])
            for i in range(nr):
                if i not in lp_set:
                    load_plan_list.append(self.can_be_sent_load_plan[i])
                else:
                    self.can_be_sent_load_plan[i].car = self.car_list[match_list[i]]
                    unbound_lp_list.append(self.can_be_sent_load_plan[i])
        self.car_list = car_list
        self.can_be_sent_load_plan = load_plan_list
        for i in unbound_lp_list:
            self.matched_pool.add(i.id)
        return unbound_lp_list

    '''
        Description: 删除已经发走的lp
    '''

    # def drop_load_plan(self, lps: List[LoadPlan]):
    #     data_loader.drop_sent_load_plan(lps)

    """
        Description: 时间窗口重制
    """

    # def reset(self):
    #     data_loader.reset()
    #     self.__init__()

    """
        Description: 储存匹配结果
    """

    # def store_match_res(self, car_mark: str, load: float):
    #     self.match_car_list.append(car_mark)
    #     self.match_car_load.append(load)
    #     self.match_car_time.append(self.time)

    """
        Description: 储存到本地
    """

    def store_match_res_to_local(self):
        self.match_res['car_mark'] = self.match_car_list
        self.match_res['load'] = self.match_car_load
        self.match_res['time'] = self.match_car_time
        # self.match_res.to_csv(curr_config_class.MATCH_RES_DIRECTORY)


time_window = Time_Window()
