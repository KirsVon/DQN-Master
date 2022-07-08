# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/07/02
# Last-Modified: liujiaye 2019/08/29


import traceback

from tool.priority_ratio_calculation import priority_ratio_calculate
from controller.config_name_management import curr_config_class

class LoadPlan:
    '一个车次信息——装车清单'

    def __init__(self, car):
        '''
        构造函数
        '''
        try:
            self.car = car
            # self.id = loading_plan_id_increment()
            self.id = 0
            self.load = 0.0
            self.city = ""
            self.is_full = False  # 是否已满
            self.priority = -1.0  # 货物优先级 负数表示未计算过
            self.cargo_list = []  # 货物列表
            self.commodity = ""
            self.unloading_address_list = set()
            self.stock_list = set()
            self.commodity_list = set()
            self.arriving_time = ""
            # self.send_time = ""
        except Exception as e:
            print("load plan init error")
            traceback.print_exc()

    def set_commodity_and_address_and_stock_set(self):
        for i in self.cargo_list:
            if i.commodity not in self.commodity_list:
                self.commodity_list.add(i.commodity)
            if i.unloading_address not in self.unloading_address_list:
                self.unloading_address_list.add(i.unloading_address)
            if i.out_stock not in self.stock_list:
                self.stock_list.add(i.out_stock)

    def set_attr(self, value: dict):
        for attr in self.__dict__.keys():
            if value.__contains__(attr):
                setattr(self, attr, value.get(attr))

    def add(self, cargo):
        # self.load += cargo.c_weight
        # if self.load >= curr_config_class.MAX_LOAD_CAPACITY:
        #     return False
        # self.cargo_list.append(cargo)
        # self.unloading_address_list.add(cargo.unloading_address)
        # self.stock_list.add(cargo.out_stock)
        # self.commodity_list.add(cargo.commodity)
        # if self.load >= curr_config_class.MIN_LOAD_CAPACITY:
        #     self.is_full = True
        #     return True
        # return True
        """
        往cargo_list中增加一个Cargo对象, 增加的判断条件为:
        1.货物是否满
            >根据车辆载重与当前货物总重进行比较
            >车辆载重的浮动是否需要考虑
        2.
        需要改变的变量:
        self.load, 货物总重
        self.priority 当前优先级(取货物优先级最高还是求所有的和)
        self.cargo_list
        self.is_full
        :param cargo: 一件货物
        """
        if curr_config_class.MIN_LOAD_CAPACITY > self.load + cargo.c_weight:
            result_type = 0
        elif curr_config_class.MAX_LOAD_CAPACITY >= self.load + cargo.c_weight >= curr_config_class.MIN_LOAD_CAPACITY:
            result_type = 1
            self.is_full = True
        else:
            result_type = -1

        if curr_config_class.MAX_LOAD_CAPACITY >= self.load + cargo.c_weight:
            self.cargo_list.append(cargo)
            self.load += cargo.c_weight
            self.unloading_address_list.add(cargo.unloading_address)
            self.stock_list.add(cargo.out_stock)
            self.commodity_list.add(cargo.commodity)
            # self.update_priority()
            # self.priority = self.priority + cargo.priority
        return result_type

    def get_stock_list(self):
        return self.stock_list

    def get_unloading_address_list(self):
        return self.unloading_address_list

    def get_commodity_list(self):
        return self.commodity_list

    def copy(self, cargo_list):
        """
        只做拷贝, 不做判断
        可能出现load>car.load_capacity的情况
        :param cargo_list: [Cargo...]
        """
        self.cargo_list = cargo_list
        self.load = sum([cargo.c_weight for cargo in cargo_list])
        self.update_priority()
        if curr_config_class.MAX_LOAD_CAPACITY >= self.load >= curr_config_class.MIN_LOAD_CAPACITY:
            self.is_full = True

    def __str__(self):
        result = "{0},{1},{2},{3}，{4}，{5}\n[".format(self.id, self.car.city, self.car.district, self.car.commodity,
                                                     str(self.load), str(self.priority))
        for i in self.cargo_list:
            result += i.__str__()
        result += "]"
        return result

    def __cmp__(self, other):
        if self.priority < other.priority:
            return -1
        elif self.priority > other.priority:
            return 1
        else:
            return 0

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def update_priority(self):
        self.priority = priority_ratio_calculate(self.cargo_list)

    def __hash__(self):
        return hash(self.id)
    # def add_goods(self, goods):
    #     '''
    #     给车次增加货物
    #     :param goods: 货物信息，Goods类型
    #     :return: 返回-1为添加失败，0为车次已满，1为已添加未满
    #     '''
    #     tmp_residual_space = self.get_residual_space(goods.can_send_weight)
    #     if tmp_residual_space < 0:
    #         return -1
    #     if tmp_residual_space == 0 or (self.is_full is True):
    #         self.residual_space = 0.0
    #         self.load += goods.can_send_weight
    #         self.goods_list.append(goods)
    #         self.add_stock(goods.out_stock)
    #         self.add_commodity(goods.commodity)
    #         self.add_unloading_address(goods.unloading_address)
    #         self.task_priority += goods.mark
    #         self.is_full = True
    #         # 添加货物同时 添加收货人的信息
    #         self.receiving_users.append(goods.receiving_user)
    #         return 0
    #     if tmp_residual_space > 0:
    #         self.residual_space = tmp_residual_space
    #         self.load += goods.can_send_weight
    #         self.goods_list.append(goods)
    #         self.task_priority += goods.mark
    #         self.add_stock(goods.out_stock)
    #         self.add_commodity(goods.commodity)
    #         self.add_unloading_address(goods.unloading_address)
    #         # 添加货物同时 添加收货人的信息
    #         self.receiving_users.append(goods.receiving_user)
    #         return 1
    #
    # def get_residual_space(self, weight):
    #     '''
    #     若加入当前weight，返回剩余空间
    #     :param weight:  下一次需要加入的重量
    #     :return: 若weight加入后的剩余重量
    #     '''
    #     if self.residual_space - weight >= 0:
    #         return self.residual_space - weight
    #     if weight - self.residual_space < self.load_limit['high'] - self.car.load_capacity and not self.is_full:
    #         self.is_full = True
    #         return self.residual_space + self.load_limit['high'] - self.car.load_capacity - weight
    #     else:
    #         return self.residual_space - weight
    #
    # def change_mark(self):
    #     '''
    #     标记车次已经匹配上车辆
    #     '''
    #     self.mark = 1
    #
    # def add_stock(self, stock):
    #     if stock not in self.stock_list:
    #         self.stock_list.append(stock)
    #
    # def add_commodity(self, commodity):
    #     if commodity not in self.commodity_list:
    #         self.commodity_list.append(commodity)
    #
    # def add_unloading_address(self, unloading_address):
    #     if unloading_address not in self.unloading_address_list:
    #         self.unloading_address_list.append(unloading_address)
    #
    # def set_trans_group_name(self, trans_group_name):
    #     self.car.trans_group_name = trans_group_name
    #
    # def set_task_priority(self):
    #     '''
    #     设置车次的优先级：车上货物的最大优先级
    #     '''
    #     result = - sys.maxsize
    #     for i in self.goods_list:
    #         if i.mark > result:
    #             result = i.mark
    #     self.task_priority = result
    #     return result
    #
    # def set_can_send_date(self):
    #     '''
    #     车次的可发时间取所有货物的最晚时间
    #     '''
    #     max_can_send_date = self.goods_list[0].can_send_date
    #     for i in self.goods_list:
    #         can_send_time = i.can_send_date
    #         if can_send_time > max_can_send_date:
    #             max_can_send_date = can_send_time
    #     self.can_send_date = max_can_send_date
    #
