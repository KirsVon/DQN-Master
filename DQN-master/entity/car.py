# -*- coding: utf-8 -*-
# Description:车辆信息数据类
# Created: liujiaye  2019/07/09
# Last-Modified: liujiaye 2019/08/29

import traceback
from utils.base_entity import BaseEntity


class Car(BaseEntity):
    '''车辆信息基础类'''

    def __init__(self):
        """
        初始化数据成员
        :param row: 字典类型或者dataframe一行数据，格式见__dict__函数
        """
        try:
            self.license_plate_number = ""  # 车牌号
            self.city = ""  # 城市
            self.district = ""  # 区
            self.commodity = ""  # 货物类
            self.load_weight = 35 # 载重
            self.arrive_time = ""  # 车辆到达的时刻
            self.district_list = []
            self.commodity_list = []
            self.commodity_set = set()
            self.city_list = []
            self.city_set = set()
        except Exception as e:
            print("car init error")
            traceback.print_exc()

    # def __int__(self,car_mark:str, city:str, district:str, commodity:str, arrive_time:str):
    #     self.license_plate_number = car_mark
    #     self.city = city
    #     self.district = district
    #     self.arrive_time = arrive_time
    #     self.commodity = self.strlist_to_list(commodity)
    #     self.district = self.strlist_to_list(district)

    def strlist_to_list(self, strlist: str):
        obj_list = []
        front = strlist.find('\'')
        rear = strlist.find('\'', front + 1)
        while front != -1:
            obj_list.append(strlist[front + 1:rear])
            front = strlist.find('\'', rear + 1)
            rear = strlist.find('\'', front + 1)
        return obj_list

    def set_attr(self, value: dict):
        for attr in self.__dict__.keys():
            if value.__contains__(attr):
                setattr(self, attr, value.get(attr))
        index = self.city.find('\'')
        # self.arrive_time = str(self.arrive_time)
        if index != -1:
            self.set_city_list()
            self.set_district_list()
            self.set_commodity_list()

    def set_commodity_list(self):
        self.commodity_list = self.strlist_to_list(self.commodity)
        for i in range(len(self.commodity_list)):
            self.commodity_set.add(self.commodity_list[i])

    def set_district_list(self):
        self.district_list = self.strlist_to_list(self.district)

    def set_city_list(self):
        self.city_list = self.strlist_to_list(self.city)
        for i in range(len(self.city_list)):
            self.city_list[i] += '市'
            self.city_set.add((self.city_list[i]))
