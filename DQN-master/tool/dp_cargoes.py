#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DQN-master 
@File    ：dp_cargoes.py
@IDE     ：PyCharm 
@Author  ：fengchong
@Date    ：2022/3/2 3:34 PM 
'''
import math
import numpy as np
from entity.cargo import Cargo
from copy import copy
from entity.car import Car
from entity.load_plan import LoadPlan
from typing import List

def set_load_task_by_items(load_task: LoadPlan):
    load_task.count = 0
    load_task.load = 0
    load_task.city = load_task.cargo_list[0].city
    load_task.province = load_task.cargo_list[0].province
    load_task.end_point = []
    load_task.consumer = []
    for i in load_task.cargo_list:
        load_task.count += i.c_count
        load_task.load += i.c_weight
        if i.actual_end_point not in load_task.end_point:
            load_task.end_point.append(i.actual_end_point)
        if i.consumer not in load_task.consumer:
            load_task.consumer.append(i.consumer)



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
    return load_task

def dp(cargos):
    MIN_LOAD_CAPACITY = 29        # math.ceil((truck.load_weight - ModelConfig.RG_SINGLE_LOWER_WEIGHT) / 1000)
    MAX_LOAD_CAPACITY = 36        # ((truck.load_weight + ModelConfig.RG_SINGLE_UP_WEIGHT) / 1000)
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
    sum_weight /= 1000
    unit_weight /= 1000

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
        tmp_cargo.c_weight = arr_j[j] * 1000
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
            result_load_plan_list.append(copy(virtual_load_plan))
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