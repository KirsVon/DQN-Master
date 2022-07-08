# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/07/02

import config
from datetime import datetime

from config import curr_config_class

overall_cargo_id = 0  # cargo id
overall_loading_plan_id = 0  # 车次号

# def change_tct(end_time):
#     global tmp_curr_time
#     tmp_curr_time = datetime.strptime(end_time, "%Y%m%d%H%M%S")
model_type = 1

current_time_str = "20201010000000"


def cargo_id_increment():
    '''
    为内存中的货物产生唯一标识的id
    # 需求1216：方法按需修改
    '''
    global overall_cargo_id
    overall_cargo_id += 1
    return overall_cargo_id


def loading_plan_id_increment():
    global overall_loading_plan_id
    overall_loading_plan_id += 1
    return overall_loading_plan_id
