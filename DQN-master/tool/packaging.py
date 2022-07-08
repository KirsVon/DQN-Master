# from app.tool.can_collocate import can_collocate
from entity.cargo import Cargo
from entity.car import Car
from entity.load_plan import LoadPlan
from typing import List
from config import curr_config_class
import math
import numpy as np
import pandas as pd
from copy import copy
import datetime
from decimal import Decimal

def unloading_address_can_collocate(cargo_unloading_address: str, unloading_address_list: list):
    if len(unloading_address_list) >= 2 and cargo_unloading_address not in unloading_address_list:
        return False
    return True

def stock_can_collocate(cargo_out_stock:str, stock_list:list):
    if len(stock_list) >= 2 and cargo_out_stock not in stock_list:
        return False
    return True

def can_collocate(goods, load_task):
    # 装车清单已满则False
    if load_task.is_full:
        return False
    # # 货物线路不同于车辆信息中的线路则False
    # if goods.line_name != load_task.car.line_name:
    #     return False
    # 货物城市不同于车辆信息中的城市则False
    if goods.city != load_task.car.city:
        return False
    # 装车清单的卸货地址列表超过2个则False 或 等于2个 但此次货物的卸货地址并不在装车清单的卸货地址列表中
    # if not unloading_address_can_collocate(goods.unloading_address, load_task.get_unloading_address_list()):
    #     return False
    # # # 货物为外库 且车辆不去外库则False
    # # if load_task.car.mark == '-' and is_outside_stock(goods.out_stock):
    # #     return False
    # # 货物的出库仓库的可搭配仓库  并没有在装车清单的仓库列表里 则False
    # 或者 装车清单的仓库列表大于2  又或者 装车清单等于2但货物的出库仓库不在列表中 = False
    # if not stock_can_collocate(goods.out_stock, load_task.get_stock_list()):
    #     return False
    # # 货物的大品名的可搭配品种  并没有在装车清单的品名列表里 则False
    # if not commodity_can_collocate(goods.commodity, load_task.get_commodity_list()):
    #     return False
    # # 如果load_task的receiving_users 收货人列表中的不重复的个数少于2个 则可以添加货物
    # if not (len(set(load_task.receiving_users)) < 2 or (goods.receiving_user in load_task.receiving_users)):
    #     return False
    return True


def packaging(cargo_list):
    """    对库存列表进行全局配载，转换为车次列表   cargo_list: List[Cargo] """

    # 货物按订单+出库仓库group
    order_dict = dict()
    for c in cargo_list:
        order_dict.setdefault(c.order_number + "," + c.out_stock, []).append(c)

    # 车次,List[尾单]=动态规划(订单)
    global_load_plan_list = list()  # :List[LoadPlan]
    global_tail_list = list()  # :List[Cargo]

    def dp(cargos):
        result_load_plan_list = list()  # :List[LoadPlan]
        tail_list = list()  # :List[Cargo]
        shipping_dict = {}
        sum_weight = 0.0
        sum_count = 0
        # -----------------------疑问：每个订单号对应一个规格货物吗？-------------------------
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
        # 去除单件大于最大重量的异常情况
        if sum_weight < curr_config_class.MIN_LOAD_CAPACITY or unit_weight > curr_config_class.MAX_LOAD_CAPACITY:
            tail_list.extend(cargos)
            return result_load_plan_list, tail_list
        # 计算最大最小件数
        max_count = min_count = math.floor(curr_config_class.MIN_LOAD_CAPACITY / unit_weight)
        tmp_weight = math.floor(curr_config_class.MIN_LOAD_CAPACITY / unit_weight) * unit_weight
        while tmp_weight < curr_config_class.MAX_LOAD_CAPACITY:
            tmp_weight += unit_weight
            max_count += 1
        # 判断最小数量的合理性：总重量大于最小值
        min_weight = min_count * unit_weight
        if min_weight < curr_config_class.MIN_LOAD_CAPACITY:
            min_count += 1
            min_weight += unit_weight
        # 判断最大数量的合理性：总重量小于最大值
        max_weight = max_count * unit_weight
        if max_weight > curr_config_class.MAX_LOAD_CAPACITY:
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

        # if len(arr_j) == 0:
        #     print(str(sum_weight) + " and " + str(unit_weight))
        arr_i = []
        for index in range(curr_config_class.MIN_LOAD_CAPACITY, math.ceil(sum_weight) + 1):
            arr_i.append(index)
        # if len(arr_i) == 0:
        #     print(str(curr_config_class.MIN_LOAD_CAPACITY) + " and " + str(sum_weight))
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
                # if arr_i[index_i] == 0:
                #     print(arr_i)
                # if arr_j[0] == 0:
                #     print(arr_j)
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
                next_i = max(math.floor(arr_i[i] - min_n * arr_j[j] - curr_config_class.MIN_LOAD_CAPACITY), 0)
                while i - n * arr_j[j] > 0:
                    n += 1
                    tmp_i = math.floor(arr_i[i] - min_n * arr_j[j] - curr_config_class.MIN_LOAD_CAPACITY)
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
            tmp_cargo.set_weight(arr_j[j], load_type[j])
            sum_weight -= arr_j[j]
            sum_count -= load_type[j]
            virtual_load_plan = get_load_plan_by_virtual_car([tmp_cargo])
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
            tail.set_weight(sum_weight, sum_count)
            tail_list.append(tail)

        return result_load_plan_list, tail_list

    for key, value in order_dict.items():
        tmp_load_plan_list, tmp_tail_list = dp(value)
        global_load_plan_list.extend(tmp_load_plan_list)
        global_tail_list.extend(tmp_tail_list)

    # 组合(List[尾单])
    # 排序

    def sort(cargo_list):
        """ 货物列表排序 """
        # 按优先级排序
        cargo_list = sorted(cargo_list, key=lambda cargo: cargo.unit_weight, reverse=True)
        return cargo_list

    remaining_cargo_list = sort(global_tail_list)

    # 寻找可组合货物

    def find_combination(main_cargo, remaining_cargo_list):
        # 寻找可组合货物
        load_plan = get_load_plan_by_virtual_car([main_cargo])
        index = 0
        while index < len(remaining_cargo_list):
            # 判断、执行<添加-删除>
            if can_collocate(remaining_cargo_list[index], load_plan):
                # split order
                sum_weight = remaining_cargo_list[index].c_weight
                sum_count = remaining_cargo_list[index].c_count
                add_type = load_plan.add(remaining_cargo_list[index])
                if add_type != -1:
                    del remaining_cargo_list[index]
                    continue
                else:
                    add_type = 0
                while sum_weight > 0 and add_type == 0:
                    tmp = Cargo()
                    tmp.set_attr(remaining_cargo_list[index].as_dict())
                    tmp.set_weight(remaining_cargo_list[index].unit_weight, 1)
                    add_type = load_plan.add(tmp)
                    if add_type == -1:
                        break
                    sum_weight -= remaining_cargo_list[index].unit_weight
                    sum_count -= 1
                    remaining_cargo_list[index].set_weight(sum_weight, sum_count)
                if sum_weight <= 0:
                    del remaining_cargo_list[index]
                else:
                    index += 1
            else:
                index += 1
        return load_plan, remaining_cargo_list

    # def check_weight(load_plan_list, remaining_cargo_list):
    #     check_weight = 0.0
    #     for lp in load_plan_list:
    #         for c in lp.cargo_list:
    #             check_weight += c.c_weight
    #     for c in remaining_cargo_list:
    #         check_weight += c.c_weight

    remaining_load_plan_list = []
    remaining_sum_weight = 0.0
    for i in remaining_cargo_list:
        remaining_sum_weight += i.c_weight

    while len(remaining_cargo_list) > 0:
        main_cargo = remaining_cargo_list[0]
        remaining_cargo_list.pop(0)
        # 循环取件
        while main_cargo.c_count >= 1:
            # 取一件
            tmp = Cargo()
            tmp.set_attr(main_cargo.as_dict())
            tmp.set_weight(main_cargo.unit_weight, 1)
            main_cargo.set_weight(main_cargo.c_weight - main_cargo.unit_weight, main_cargo.c_count - 1)
            # 拼车过程
            tmp_load_plan, remaining_cargo_list = find_combination(tmp, remaining_cargo_list)
            remaining_load_plan_list.append(tmp_load_plan)
            # check_weight(remaining_load_plan_list, remaining_cargo_list)
        remaining_cargo_list = sort(remaining_cargo_list)
        # remaining_cargo_list.pop(0)

    global_load_plan_list.extend(remaining_load_plan_list)
    for i in global_load_plan_list:
        if i.load > curr_config_class.MAX_LOAD_CAPACITY or i.load < curr_config_class.MIN_LOAD_CAPACITY:
            #     i.priority = 0
            # else:
            i.update_priority()
        i.set_commodity_and_address_and_stock_set()
    return global_load_plan_list


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
            load_task.load += cargo.c_weight
    load_task.city = load_task.cargo_list[0].city
    return load_task