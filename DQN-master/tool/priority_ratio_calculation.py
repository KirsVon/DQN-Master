# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/12/20
from tool.z_score import sigmoid


def priority_ratio_calculate(cargo_list):
    priority_ratio = 0
    sum_weight = 0.0
    sum_priority_weight = 0.0
    for i in cargo_list:
        sum_weight += i.c_weight
        sum_priority_weight += i.priority * i.c_weight
    if sum_weight > 0.0:
        priority_ratio = sum_priority_weight / sum_weight
    return priority_ratio


def priority_ratio_calculate_limit(cargo_list, load_capacity):
    priority_ratio = 0
    sum_priority_weight = 0.0
    for i in cargo_list:
        sum_priority_weight = i.priority * i.c_weight
    if load_capacity > 0.0:
        priority_ratio = sum_priority_weight / load_capacity
    return priority_ratio


def priority_ratio_calculate_by_list(weight_list):
    '''
    weight_list->[[weight,date],[weight,date],...]
    '''
    priority_ratio = 0
    sum_weight = 0.0
    sum_priority_weight = 0.0
    for item in weight_list:
        sum_weight += item[0]
        sum_priority_weight += sigmoid(item[1]) * item[0]
    if sum_weight > 0.0:
        priority_ratio = sum_priority_weight / sum_weight
    return priority_ratio
