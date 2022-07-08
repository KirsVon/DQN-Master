# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/07/02
# Last-Modified: liujiaye 2019/08/29

import traceback
from _datetime import datetime
from controller.config_name_management import cargo_id_increment
from tool.z_score import sigmoid
from controller.config_name_management import current_time_str
from utils.base_entity import BaseEntity
from tool.commodity_transform import commodity_transform


class Cargo(BaseEntity):
    '一条货物信息'

    def __init__(self):
        try:
            # order information
            self.order_number = ""  # 订单号
            self.shipping = ""  # 发货通知单号
            self.city = ""  # 城市
            self.district = ""  # 区县
            self.unloading_address = ""  # 卸货地址
            self.receiving_user = ""  # 收货用户
            # cargo information
            self.id = -1
            self.prod_name = ""  # 小品名
            self.commodity = ""  # 大品名
            self.current_time = '' # 当前匹配时间
            # self.weight = 0.0  # 总重量
            # self.count = 0  # 总件数
            self.unit_weight = 0.0  # 单件重量
            # self.w_weight = 0.0  # 计划重量
            # self.w_count = 0  # 计划件数
            self.c_weight = 0.0  # 可发重量
            self.c_count = 0  # 可发件数
            self.can_send_date = None  # 可发日期
            self.time_period = None  # 当前数据所属时段
            self.out_stock = ""  # 出库仓库
            self.priority = -1  # 优先级
            self.status = ""  # 描述库存状态（）
            # self.fields = ["order_number", "shipping", "city", "district", "unloading_address", "receiving_user", "id"
            #     , "prod_name", "commodity", "weight", "count", "w_weight", "w_count", "c_weight", "c_count"
            #     , "can_send_date", "time_period", "out_stock", "priority", "status"]
        except Exception as e:
            '需要增加输入参数缺失的错误判断'
            print("cargo init error")
            traceback.print_exc()

    def set_current_time(self, timestr:str):
        self.current_time = timestr

    def get_pri(self,cargo_date:str):
        current_time = datetime.strptime(cargo_date, "%Y%m%d%H%M%S")
        contract_s_date = datetime.strptime("20" + self.shipping[1:], "%Y%m%d%H%M%S")
        gap = current_time - contract_s_date
        res = gap.days + 1
        self.priority = res

    def clu_weight_and_count(self):
        self.c_weight = float(self.c_weight)
        self.c_count = int(self.c_count)

        # self.w_weight = float(self.w_weight)
        # self.w_count = int(self.w_count)

        # self.weight = self.w_weight + self.c_weight
        # self.count = self.w_count + self.c_count
        if self.c_count == 0:
            self.c_count = 1
        elif self.c_weight == 0:
            self.unit_weight = 0
            return
        self.unit_weight = self.c_weight / self.c_count

    def set_attr(self, value: dict):
        for attr in self.__dict__.keys():
            if value.__contains__(attr):
                setattr(self, attr, value.get(attr))
        # self.id = cargo_id_increment()
        self.clu_weight_and_count()
        self.commodity = commodity_transform.change_to_big(self.prod_name)

    def set_weight(self, weight, count=0):
        self.c_weight = weight
        if count != 0:
            self.c_count = count
            self.unit_weight = self.c_weight / count
        else:
            if self.unit_weight != 0 and self.c_weight != 0:
                self.c_count = int(self.c_weight / self.unit_weight)
            else:
                self.c_count = 0
                self.unit_weight = 0.0
                self.c_weight = 0.0

        # self.clu_weight_and_count()
        # self.get_priority()

    # def __repr__(self):
    #     return repr(self.c_weight, self)

    def __str__(self):
        return "{0},{1},{2},{3},{4}\n".format(self.commodity, self.city, str(self.c_weight), str(self.c_count),
                                              str(self.priority))

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id


