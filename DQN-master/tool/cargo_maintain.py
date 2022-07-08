# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/08/07

from tool.stock_msg import get_stock_msg


class CargoManagement:
    def __init__(self, start_time):
        self.cargo_dic = {}
        self.outbound = {}
        self.outbound_queue = CountQueue(50)  # 设置最大历史记录处理数量
        self.init_cargo_dic(start_time)

    def init_cargo_dic(self, start_time, mark=1):
        '''
        输入的货物必须以键值唯一标识，没有重复项
        @param start_time:
        @param mark:
        @return:
        '''
        print("init_cargo_dic:" + start_time)
        cargos = get_stock_msg(start_time, mark)
        #self.outbound = {}
        self.outbound_queue = CountQueue(50)
        # 维护数据格式:{id:[status,cargo]}
        self.cargo_dic = {}
        for g in cargos:
            key_str = g.order_number + "," + g.shipping + "," + g.out_stock
            if key_str in self.outbound:  # 去除前面时段已经分出去的量
                weight = g.c_weight - self.outbound[key_str].c_weight
                g.set_weight(weight)
            if g.c_weight > 0:  # 货物存在重量时才在内存管理中增加
                self.cargo_dic[key_str] = g
                self.cargo_dic[key_str].get_pri(start_time)

    def add_status(self, load_plan):
        for g in load_plan.cargo_list:
            key_str = g.order_number + "," + g.shipping + "," + g.out_stock
            if key_str in self.outbound:
                weight = self.outbound[key_str].c_weight + g.c_weight
                self.outbound[key_str].set_weight(weight)
            else:
                self.outbound[key_str] = g
            del_key = self.outbound_queue.put(key_str)
            if del_key is not None and del_key in self.outbound:
                self.outbound.pop(del_key)

    def cargo_all(self):
        cargo_list = []
        for key in self.cargo_dic:
            g = self.cargo_dic[key]
            if key in self.outbound:
                if g.c_weight <= self.outbound[key].c_weight:
                    continue
                else:
                    weight = g.c_weight - self.outbound[key].c_weight
                    g.set_weight(weight)
            else:
                cargo_list.append(g)
        return cargo_list




    def cargo_list_filter(self, condition):
        """
        根据条件筛选内存中的货物
        :param condition: [city]
        :return:
        """
        cargo_list = []
        for c in condition:
            for key in self.cargo_dic:
                g = self.cargo_dic[key]
                if key in self.outbound:
                    if g.c_weight <= self.outbound[key].c_weight:
                        continue
                    else:
                        weight = g.c_weight - self.outbound[key].c_weight
                        g.set_weight(weight)
                if g.city == c:
                    cargo_list.append(g)
        return cargo_list


class CountQueue:
    """ 计数，增加键值对其他所有已有键值计数一次，超过队列最大值，pop计数次数最多的值 """

    def __init__(self, size):
        self.key_count = {}
        self.max_size = size

    def put(self, key):
        self.key_count.setdefault(key, 0)
        for k in self.key_count:
            if k == key:
                continue
            self.key_count[k] += 1

        if len(self.key_count) > self.max_size:
            key_sort = sorted(self.key_count.items(), key=lambda k: k[1], reverse=True)
            return key_sort[0]
        return None


cargo_management = CargoManagement('20200924000000')
