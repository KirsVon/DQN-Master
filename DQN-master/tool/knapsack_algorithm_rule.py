#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 15:57
# @Author  : ping yu
# @File    : knapsack_algorithm_rule.py
# @Software: PyCharm
import time
from typing import List
from entity.cargo import Cargo
from entity.load_plan import LoadPlan
from entity.car import Car
import math
import numpy as np
from copy import deepcopy

start = time.perf_counter()
RG_J_GROUP = ['老区-卷板', '新产品-卷板', '新产品-白卷']

def knapsack_algorithm_rule(stock_list: List):  # truck=Car()
    """
    1. 大件订单最小甩货拆分
    2. 不足标载订单按规则拼货
    """
    order_dict = dict()
    for c in stock_list:
        order_dict.setdefault(c.order_number + "," + c.out_stock, []).append(c)
    load_plan_list = list()
    tail_stock_list = list()
    MIN_LOAD_CAPACITY = 29        # math.ceil((truck.load_weight - ModelConfig.RG_SINGLE_LOWER_WEIGHT) / 1000)
    MAX_LOAD_CAPACITY = 36        # ((truck.load_weight + ModelConfig.RG_SINGLE_UP_WEIGHT) / 1000)
    # if truck.commodity in ModelConfig.RG_J_GROUP:
    #     if 29000 <= truck.load_weight and 35000 >= truck.load_weight:
    #         MAX_LOAD_CAPACITY = ((truck.load_weight + ModelConfig.RG_SINGLE_UP_WEIGHT) / 1000)
    #         MIN_LOAD_CAPACITY = 29
    # else:
    #     if 31000 <= truck.load_weight and 35000 >= truck.load_weight:
    #         MAX_LOAD_CAPACITY = ((truck.load_weight + ModelConfig.RG_SINGLE_UP_WEIGHT) / 1000)
    #         MIN_LOAD_CAPACITY = 31

    def dp(cargos):
        result_load_plan_list = list()  # :List[LoadTask]
        tail_list = []  # :List[Stock]
        shipping_dict = {}
        sum_weight = 0.0
        sum_count = 0
        # -----------------------疑问：每个订单号对应一个规格货物吗？---------------- ---------
        unit_weight = cargos[0].unit_weight
        if unit_weight == 0.0:
            return result_load_plan_list, tail_list

        for c in cargos:
            shipping_dict.setdefault(c.shipping, []).append(c)  # 验证是否同个订单下有多个发货通知单，是否有区别
            sum_weight += c.c_weight
            sum_count += c.c_count


        # 动态规划划分订单
        # 1:找到达到装载限制的list
        load_type = []

        if sum_weight < MIN_LOAD_CAPACITY or unit_weight > MAX_LOAD_CAPACITY:
            tail_list.extend(cargos)
            return result_load_plan_list, tail_list

        max_count = min_count = math.floor(MIN_LOAD_CAPACITY / unit_weight)
        tmp_weight = math.floor(MIN_LOAD_CAPACITY / unit_weight) * unit_weight
        while tmp_weight < MAX_LOAD_CAPACITY:
            tmp_weight += unit_weight
            max_count += 1
        # 判断最小数量的合理性：总重量大于最小值
        min_weight = min_count * unit_weight
        if min_weight < MIN_LOAD_CAPACITY:
            min_count += 1
            min_weight += unit_weight
        # 判断最大数量的合理性：总重量小于最大值
        max_weight = max_count * unit_weight
        if max_weight > MAX_LOAD_CAPACITY:
            max_count -= 1
            max_weight -= unit_weight
        if min_count > max_count:
            tail_list.extend(cargos)
            return result_load_plan_list, tail_list
        # 通过最小件数和最大件数获取打包类型
        for i in range(min_count, max_count + 1):
            load_type.append(i)

        # 2:dp
        # i=装一车的方案; j=总重量拆分
        #   1.不使用当前方案arr[i]-->dp[i-1][j]
        #   2.用n个当前方案arr[i]-->max{sum-dp[i-1][j-n*arr[i]]-n*arr[i],0}
        # dp[i][j]=min{dp[i-1][j],max{sum-dp[i-1][j-n*arr[i]]-n*arr[i],0}}
        arr_j = [x * unit_weight for x in load_type]

        arr_i = []
        for index in range(MIN_LOAD_CAPACITY, math.ceil(sum_weight) + 1):
            arr_i.append(index)

        dp_matrix = np.arange(len(arr_i) * len(arr_j)).reshape(len(arr_i), len(arr_j))
        result = np.arange(len(arr_i) * len(arr_j)).reshape(len(arr_i), len(arr_j))
        for index_j in range(len(arr_j)):
            # 初始化第一行
            if arr_j[index_j] < arr_i[0]:
                dp_matrix[0][index_j] = arr_i[0] - arr_j[index_j]
                result[0][index_j] = 1
            else:
                dp_matrix[0][index_j] = arr_i[0]
                result[0][index_j] = 0
        for index_i in range(len(arr_i)):
            if arr_i[index_i] >= arr_j[0]:
                dp_matrix[index_i][0] = arr_i[index_i] - math.floor(arr_i[index_i] / arr_j[0]) * arr_j[0]
                result[index_i][0] = math.floor(arr_i[index_i] / arr_j[0])
            else:
                dp_matrix[index_i][0] = arr_i[index_i]
                result[index_i][0] = 0
        i = 1
        j = 1
        while i < len(arr_i):
            j = 1
            while j < len(arr_j):
                dp_1 = dp_matrix[i][j - 1]
                min_n = 1
                n = 1
                # next_i 下一个重量的index，arr_i[i] - min_n * arr_j[j]为下一个重量值，- curr_config_class.MIN_LOAD_CAPACITY做下标对其
                next_i = max(math.floor(arr_i[i] - min_n * arr_j[j] - MIN_LOAD_CAPACITY), 0)
                while i - n * arr_j[j] > 0:
                    n += 1
                    tmp_i = math.floor(arr_i[i] - min_n * arr_j[j] - MIN_LOAD_CAPACITY)
                    if tmp_i < 0:
                        break
                    if dp_matrix[next_i][j - 1] > dp_matrix[tmp_i][j - 1]:
                        min_n = n
                        next_i = tmp_i
                n = min_n
                # next_i = max(math.floor(i - min_n * arr_j[j] - curr_config_class.MIN_LOAD_CAPACITY), 0)
                dp_2 = dp_matrix[next_i][j - 1]
                if dp_2 < 0:
                    dp_2 = arr_i[i]
                    n = 0
                if dp_1 < dp_2:
                    dp_matrix[i][j] = dp_1
                    result[i][j] = 0
                else:
                    dp_matrix[i][j] = dp_2
                    result[i][j] = n
                j += 1
            i += 1
        i -= 1
        j -= 1

        while i >= 0 and j >= 0:
            if result[i][j] == 0:
                i -= max(math.floor(result[i][j] * arr_j[j]), 1)
                continue
            tmp_cargo = Cargo()
            tmp_cargo.set_attr(cargos[0].as_dict())
            # tmp_cargo.set_weight(arr_j[j], load_type[j])
            tmp_cargo.c_weight = arr_j[j]
            if load_type[j] != 0:
                tmp_cargo.c_count = load_type[j]
                tmp_cargo.unit_weight = tmp_cargo.c_weight / tmp_cargo.c_count
            sum_weight -= arr_j[j]
            sum_count -= load_type[j]
            virtual_load_plan = get_load_plan_by_virtual_car([tmp_cargo])
            set_load_task_by_items(virtual_load_plan)
            result_load_plan_list.append(virtual_load_plan)
            for index in range(1, result[i][j]):
                # virtual_load_plan = get_load_plan_by_virtual_car([tmp_cargo])
                result_load_plan_list.append(deepcopy(virtual_load_plan))
                sum_weight -= arr_j[j]
                sum_count -= load_type[j]

            i -= max(math.floor(result[i][j] * arr_j[j]), 1)
            j -= 1
        if sum_weight > 0 and sum_count > 0:
            tail = Cargo()
            tail.set_attr(cargos[0].as_dict())
            tail.c_weight = sum_weight
            tail.c_count = sum_count
            tail_list.append(tail)

        return result_load_plan_list, tail_list

    for key, value in order_dict.items():
        tmp_load_plan_list, tmp_tail_list = dp(value)
        load_plan_list.extend(tmp_load_plan_list)
        tail_stock_list.extend(tmp_tail_list)

    def can_collocate(stock, load_task: LoadPlan):
        if len(load_task.cargo_list) == 0:
            return True
        # 仓库拼货限制
        # F10、F20可互相拼货，但不可与其他仓库拼货，其余仓库可自由拼货
        out_stock_list = []
        for i in load_task.cargo_list:
            if i.out_stock not in out_stock_list:
                out_stock_list.append(i.out_stock)
        if stock.out_stock == "F10" or stock.out_stock == "F20":
            for i in out_stock_list:
                if i != "F10" or i != "F20":
                    return False
        if "F10" in out_stock_list or "F20" in out_stock_list:
            if stock.out_stock not in ["F10", "F20"]:
                return False
        # 最多两个仓库拼货
        if len(out_stock_list) == 2 and stock.out_stock not in out_stock_list:
            return False

        # 品种限制
        # 单品种拼货
        single_delivery_carpool = ["老区-型钢", "老区-线材", "老区-螺纹", "老区-卷板", "老区-开平板", "新产品-冷板"]
        commodity_list = []
        for i in load_task.cargo_list:
            if i.commodity not in commodity_list:
                commodity_list.append(i.commodity)
        if stock.commodity in single_delivery_carpool:
            if len(commodity_list) > 1:
                return False
            elif len(commodity_list) == 1 and stock.commodity != commodity_list[0]:
                if commodity_list[0] == "新产品-白卷" and stock.commodity == "老区-卷板":
                    pass
                else:
                    return False
        # 老区-卷板规格限制
        if stock.commodity == "老区-型钢":
            specs = []
            for i in load_task.cargo_list:
                if i.specs not in specs:
                    specs.append(i.specs)
                if i.commodity != "老区-型钢":
                    return False
            if len(specs) > 2 or (len(specs) == 2 and stock.specs not in specs):
                return False
        return True

    def sort(cargo_list):
        """ 货物列表排序 """
        # 按优先级排序
        cargo_list = sorted(cargo_list, key=lambda cargo: cargo.c_weight, reverse=True)
        return cargo_list

    def sort_by_unit_weight(cargo_list):
        """ 货物列表排序 """
        # 按优先级排序
        cargo_list = sorted(cargo_list, key=lambda cargo: cargo.unit_weight, reverse=True)
        return cargo_list

    tmp_tail_stock_list = deepcopy(sort(tail_stock_list))
    remaining_cargo_list = []
    # 单间拆拼
    for i in tmp_tail_stock_list:
        for j in range(0, i.c_count):
            tmp = Cargo()
            tmp.set_attr(i.__dict__)
            tmp.c_weight = i.unit_weight
            tmp.c_count = 1
            remaining_cargo_list.append(tmp)
    list_index = 0
    for i in range(len(remaining_cargo_list)):
        if remaining_cargo_list[i].c_weight > 0.1:
            list_index = i
        else:
            break
    remaining_cargo_list = remaining_cargo_list[0:list_index + 1]
    remaining_cargo_list = sort_by_unit_weight((remaining_cargo_list))

    def find_combination(main_cargo: Cargo, remaining_cargo_list: List[Cargo]):
        # 寻找可组合货物
        load_plan = get_load_plan_by_virtual_car([main_cargo])
        set_load_task_by_items(load_plan)
        index = 0
        while index < len(remaining_cargo_list):
            # 判断、执行<添加-删除>
            if can_collocate(remaining_cargo_list[index], load_plan):
                # split order
                sum_weight = remaining_cargo_list[index].c_weight
                sum_count = remaining_cargo_list[index].c_count
                add_type = load_plan_add_stock(load_plan, remaining_cargo_list[index])
                if add_type == 1:
                    tmp = Cargo()
                    tmp.set_attr(remaining_cargo_list[index].as_dict())
                    tmp.c_weight = tmp.unit_weight = remaining_cargo_list[index].unit_weight
                    tmp.c_count = 1
                    load_plan.cargo_list.append(tmp)
                    load_plan.load += tmp.unit_weight
                    remaining_cargo_list[index].c_weight -= remaining_cargo_list[index].unit_weight
                    remaining_cargo_list[index].c_count -= 1
                elif add_type == 0:
                    load_task_weight = load_plan.load
                    unit_weight = remaining_cargo_list[index].unit_weight
                    lower_bound = int((MIN_LOAD_CAPACITY - load_task_weight) / unit_weight)
                    upper_bound = int((MAX_LOAD_CAPACITY - load_task_weight) / unit_weight)
                    tmp = Cargo()
                    tmp.set_attr(remaining_cargo_list[index].as_dict())
                    tmp.c_weight = tmp.unit_weight = remaining_cargo_list[index].unit_weight
                    tmp.c_count = 1
                    if lower_bound == upper_bound and lower_bound <= remaining_cargo_list[index].c_count:
                        for i in range(0, lower_bound):
                            load_plan.cargo_list.append(deepcopy(tmp))
                        set_load_task_by_items(load_plan)
                        remaining_cargo_list[index].c_count = 0
                        remaining_cargo_list[index].c_weight = 0
                    elif lower_bound == upper_bound and lower_bound > remaining_cargo_list[index].c_count:
                        for i in range(0, remaining_cargo_list[index].c_count):
                            load_plan.cargo_list.append(deepcopy(tmp))
                        set_load_task_by_items(load_plan)
                        remaining_cargo_list[index].c_count = 0
                        remaining_cargo_list[index].c_weight = 0
                    elif lower_bound < upper_bound and remaining_cargo_list[index].c_count >= upper_bound:
                        for i in range(0, upper_bound):
                            load_plan.cargo_list.append(deepcopy(tmp))
                        set_load_task_by_items(load_plan)
                        remaining_cargo_list[index].c_count -= upper_bound
                        remaining_cargo_list[index].c_weight -= upper_bound * remaining_cargo_list[index].unit_weight
                    elif lower_bound < upper_bound and remaining_cargo_list[index].c_count < upper_bound:
                        for i in range(0, remaining_cargo_list[index].c_count):
                            load_plan.cargo_list.append(deepcopy(tmp))
                        set_load_task_by_items(load_plan)
                        remaining_cargo_list[index].c_count = 0
                        remaining_cargo_list[index].c_weight = 0
                elif add_type == -1:
                    index += 1
                    continue
                sum_weight = remaining_cargo_list[index].c_weight
                sum_count = remaining_cargo_list[index].c_count
                if sum_weight <= 0:
                    del remaining_cargo_list[index]
                else:
                    index += 1
            else:
                index += 1
        return load_plan, remaining_cargo_list

    def load_plan_add_stock(load_task: LoadPlan, stock: Cargo):
        return_type = 0
        if MIN_LOAD_CAPACITY > (load_task.load + stock.unit_weight):
            return_type = 0
        elif MAX_LOAD_CAPACITY >= (load_task.load + stock.unit_weight) >= MIN_LOAD_CAPACITY:
            return_type = 1
        else:
            return_type = -1
        return return_type

    remaining_load_plan_list = []
    remaining_sum_weight = 0.0
    for i in remaining_cargo_list:
        remaining_sum_weight += i.c_weight

    one_load_uncompleted_list = []
    beginning_queue = deepcopy(remaining_cargo_list)
    while len(remaining_cargo_list) > 0:
        main_cargo = beginning_queue[0]
        beginning_queue.pop(0)
        remaining_cargo_list = deepcopy(beginning_queue)
        # 循环取件
        while main_cargo.c_count >= 1:
            # 取一件
            # print("当前拼货对象：",main_cargo.unit_weight, main_cargo.c_weight)
            tmp = Cargo()
            tmp.set_attr(main_cargo.as_dict())
            tmp.c_weight = tmp.unit_weight = main_cargo.unit_weight
            tmp.c_count = 1
            main_cargo.c_weight = main_cargo.c_weight - main_cargo.unit_weight
            main_cargo.c_count = main_cargo.c_count - 1
            # 拼车过程
            can_carpooling_stock = []
            cant_carpooling_stock = []
            for i in range(len(remaining_cargo_list)):
                if remaining_cargo_list[i].city == main_cargo.city:
                    can_carpooling_stock.append(deepcopy(remaining_cargo_list[i]))
                else:
                    cant_carpooling_stock.append(deepcopy(remaining_cargo_list[i]))
            remaining_cargo_list = cant_carpooling_stock
            tmp_mark = str(tmp.out_stock) + ";" + tmp.order_number + ";" + tmp.shipping
            # 一装货物
            if tmp_mark not in one_load_uncompleted_list:
                one_load_stock = []
                two_load_stock = []
                for i in can_carpooling_stock:
                    if i.out_stock == tmp.out_stock:
                        one_load_stock.append(deepcopy(i))
                    else:
                        two_load_stock.append(deepcopy(i))
                tmp_load_plan, one_load_stock = find_combination(tmp, one_load_stock)
                remaining_cargo_list.extend(two_load_stock)
                if tmp_load_plan.load >= MIN_LOAD_CAPACITY and tmp_load_plan.load <= MAX_LOAD_CAPACITY:
                    remaining_load_plan_list.append(tmp_load_plan)
                    after_queue = []
                    for i in range(1, len(tmp_load_plan.cargo_list)):
                        for j in range(len(beginning_queue)):
                            if beginning_queue[j].out_stock == tmp_load_plan.cargo_list[i].out_stock and \
                                    beginning_queue[j].order_number == tmp_load_plan.cargo_list[i].order_number and \
                                    tmp_load_plan.cargo_list[i].shipping == beginning_queue[j].shipping and \
                                    tmp_load_plan.cargo_list[i].c_weight == beginning_queue[j].c_weight:
                                after_queue.append(beginning_queue[j])
                                break
                    for i in after_queue:
                        for j in beginning_queue:
                            if j.out_stock == i.out_stock and j.order_number == i.order_number and i.shipping == j.shipping and i.c_weight == j.c_weight:
                                beginning_queue.remove(j)
                                break
                else:
                    beginning_queue.append(tmp)
                    one_load_uncompleted_list.append(tmp_mark)
            else:
                tmp_load_plan, can_carpooling_stock = find_combination(tmp, can_carpooling_stock)
                if (
                        tmp_load_plan.load >= MIN_LOAD_CAPACITY and tmp_load_plan.load <= MAX_LOAD_CAPACITY) or \
                        (len(tmp_load_plan.cargo_list) == 1 and tmp_load_plan.cargo_list[
                            0].commodity in RG_J_GROUP and tmp_load_plan.load >= 26):
                    remaining_load_plan_list.append(tmp_load_plan)
                    after_queue = []
                    for i in range(1, len(tmp_load_plan.cargo_list)):
                        for j in range(len(beginning_queue)):
                            if beginning_queue[j].out_stock == tmp_load_plan.cargo_list[i].out_stock and \
                                    beginning_queue[j].order_number == tmp_load_plan.cargo_list[i].order_number and \
                                    tmp_load_plan.cargo_list[i].shipping == beginning_queue[j].shipping and \
                                    tmp_load_plan.cargo_list[i].c_weight == beginning_queue[j].c_weight:
                                after_queue.append(beginning_queue[j])
                                break
                    for i in after_queue:
                        for j in beginning_queue:
                            if j.out_stock == i.out_stock and j.order_number == i.order_number and i.shipping == j.shipping and i.c_weight == j.c_weight:
                                beginning_queue.remove(j)
                                break
                else:
                    after_queue = []
                    for i in range(len(can_carpooling_stock)):
                        for j in range(len(beginning_queue)):
                            if beginning_queue[j].out_stock == can_carpooling_stock[i].out_stock and \
                                    beginning_queue[j].order_number == can_carpooling_stock[i].order_number and \
                                    can_carpooling_stock[i].shipping == beginning_queue[j].shipping and \
                                    beginning_queue[j].c_weight == can_carpooling_stock[i].c_weight:
                                after_queue.append(beginning_queue[j])
                                break
                    for i in after_queue:
                        for j in beginning_queue:
                            if j.out_stock == i.out_stock and j.order_number == i.order_number and i.shipping == j.shipping and i.c_weight == j.c_weight:
                                beginning_queue.remove(j)
                                break
    load_plan_list.extend(remaining_load_plan_list)
    for i in load_plan_list:
        i.set_commodity_set()

    # load_plan_list = load_task_screening_deduplication(load_plan_list)
    # 要不要返回stock_matrix?
    return load_plan_list


    # 多拼类型判别
    # final_load_plan_list = []
    # for i in load_plan_list:
    #     big_commodity_list = []
    #     for j in i.items:
    #         big_commodity_list.append(j.commodity)
    #     if truck.commodity in big_commodity_list:
    #         final_load_plan_list.append(i)
    # 单卷判别

    # stock_matrix = []
    # # final_str = ""
    # for i in final_load_plan_list:
    #     stock_item_list = []
    #     stock_item_dict = {}
    #     for j in i.items:
    #         stock_str = j.out_stock + ';' + j.order_number + ';' + j.shipping
    #         if stock_str not in stock_item_dict.keys():
    #             stock_item_dict[stock_str] = []
    #             stock_item_dict[stock_str].append(j)
    #         else:
    #             stock_item_dict[stock_str].append(j)
    #     for key in stock_item_dict.keys():
    #         tmp = Cargo()
    #         tmp.set_attr(stock_item_dict[key][0].__dict__)
    #         if len(stock_item_dict[key]) > 1:
    #             for j in range(1, len(stock_item_dict[key])):
    #                 tmp.c_weight += stock_item_dict[key][j].c_weight
    #                 tmp.c_count += stock_item_dict[key][j].c_count
    #         stock_item_list.append(tmp)
    #     stock_matrix.append(stock_item_list)
    #     s = json.dumps(i, default=lambda i:i.__dict__,ensure_ascii=False)
    #     final_str += s
    # fh = open('/Users/lalala/Desktop/test.txt', 'w',encoding='utf-8')
    # fh.write(final_str)
    # fh.close()
    # return stock_matrix


def set_load_task_by_items(load_task: LoadPlan):
    load_task.count = 0
    load_task.load = 0
    load_task.city = load_task.cargo_list[0].city
    load_task.end_point = []
    load_task.consumer = []
    for i in load_task.cargo_list:
        load_task.count += i.c_count
        load_task.load += i.c_weight


def get_load_plan_by_virtual_car(cargo_list: List[Cargo]) -> LoadPlan:
    """
    1.  使用货物列表补充装车清单
    """
    car = Car()
    load_task = LoadPlan(car)
    if len(cargo_list) >= 1:
        # 补充装车清单
        for cargo in cargo_list:
            load_task.cargo_list.append(cargo)
    load_task.city = load_task.cargo_list[0].city
    return load_task

end = time.perf_counter()
print("knapsack_algorithm_rule", end - start)
