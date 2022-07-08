# -*- coding: utf-8 -*-
# Description: 获取实时库存信息（提供给单车现场分货）
# Created: 邵鲁玉 2019/08/07
# Modified: 邵鲁玉 2019/08/07;
import os

# from sqlalchemy.exc import SQLAlchemyError
from entity.cargo import Cargo
from _datetime import datetime, timedelta

import copy
# from app.main.dao.link_data_base import link_data_base
import traceback
import pandas as pd
import config
from config import curr_config_class


def get_stock_msg(start_time, mark=1):
    if mark == 1:
        return get_stock(start_time)
    else:
        return get_stock_all(start_time)


def get_stock_all(day_time):
    """
    测试单车分货   获取历史库存
    """
    try:
        cargo_dic = {}
        d = datetime.strptime(day_time, "%Y%m%d%H%M%S")
        next_day = d + timedelta(days=1)
        while d < next_day:
            filename = curr_config_class.STOCK_DATA_ROOT_DIRECTORY_BY_DAY + day_time[0:8] + '/' \
                       + d.strftime("%Y%m%d%H%M%S") + ".csv"
            data = pd.read_csv(filename)
            records_dict_list = data.to_dict(orient="records")

            for i in records_dict_list:
                tmp = Cargo()
                tmp.set_attr(i)
                key_str = tmp.order_number + "," + tmp.shipping + "," + tmp.out_stock
                if tmp.unit_weight > curr_config_class.MAX_LOAD_CAPACITY:
                    continue
                if key_str in cargo_dic:
                    weight = cargo_dic[key_str].c_weight
                    cargo_dic[key_str].set_weight(tmp.c_weight + weight)
                else:
                    cargo_dic[key_str] = copy.deepcopy(tmp)
            d = d + timedelta(minutes=20)
        return cargo_dic.values()
    except Exception as e:
        traceback.print_exc()
        return []


def get_stock(start_time):
    """
    测试单车分货   获取历史库存
    """
    try:
        filename = curr_config_class.STOCK_DATA_ROOT_DIRECTORY_BY_DAY + start_time[0:8] + '/' + start_time + ".csv"
        data = pd.read_csv(filename)
        records_dict_list = data.to_dict(orient="records")
        stock_list = []
        for i in records_dict_list:
            tmp = Cargo()
            tmp.set_attr(i)
            if tmp.unit_weight <= curr_config_class.MAX_LOAD_CAPACITY:
                stock_list.append(tmp)
        return stock_list
    except Exception as e:
        traceback.print_exc()
        return []
