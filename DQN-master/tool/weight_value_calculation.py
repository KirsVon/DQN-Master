# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/12/20
from entity.car import Car
from entity.load_plan import LoadPlan
from tool.z_score import sigmoid


def cul_Jaccard_coefficient(car_set: set, lp_set: set):
    intersec = car_set.intersection(lp_set)
    uni = car_set.union(lp_set)
    return len(intersec) / len(uni)


def weight_value_calculate(car: Car, lp: LoadPlan):
    weight = 0
    lp_car_city_set = set()
    for i in range(len(lp.cargo_list)):
        lp_car_city_set.add(lp.cargo_list[i].city)
    lp_car_city_set.add(lp.city)
    car.city_set.add(car.city)
    if cul_Jaccard_coefficient(car.city_set, lp_car_city_set) == 0:
        weight = -1
        return weight
    car.commodity_set.add(car.commodity)
    lp.commodity_list.add(lp.commodity)
    jacc = cul_Jaccard_coefficient(car.commodity_set, lp.commodity_list)
    if jacc == 0:
        weight = 0.01 * lp.load
    else:
        weight = jacc * lp.load * 10
    weight = int(weight)
    return weight