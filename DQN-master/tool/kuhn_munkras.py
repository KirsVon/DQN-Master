from datetime import datetime, timedelta

from entity.car import Car
from entity.load_plan import LoadPlan
from entity.time_window import Time_Window
import sys
import math
from tool.weight_value_calculation import weight_value_calculate
import numpy as np

import sys
sys.setrecursionlimit(100000)
class kuhn_munkras:
    def __init__(self, batch: Time_Window):    # 批、批最大长度、批最小长度
        self.batch = batch
        self.gama = 0.9
        self.lp_len = len(self.batch.can_be_sent_load_plan)
        self.ex_car = [0 for i in range(self.lp_len)]  # KM算法实现需要
        self.ex_huo = [0 for i in range(self.lp_len)]
        self.vis_car = [False for i in range(self.lp_len)]
        self.vis_huo = [False for i in range(self.lp_len)]
        self.match = [-1 for i in range(self.lp_len)]
        self.slack = [sys.maxsize for i in range(self.lp_len)]

    def dfs(self, car: int, matrix_len: int, matrix: [[]]):
        self.vis_car[car] = True
        for i in range(matrix_len):
            if self.vis_huo[i]:  # 若该右边节点访问过， 则跳过循环
                continue
            gap = self.ex_car[car] + self.ex_huo[i] - matrix[car][i]    # 否则，计算顶标和与边权之差
            if gap == 0:  # 如果符合要求、 即 Lx + Ly = weight[x][y]
                self.vis_huo[i] = True
                if self.match[i] == -1 or self.dfs(self.match[i], matrix_len, matrix):  # 找到没有一个匹配的右边节点 或者与右边节点的匹配的左边节点可以找到其他节点来匹配
                    self.match[i] = car
                    return True
            else:
                self.slack[i] = min(self.slack[i], gap)  # slack 意为该左边节点想要找到右边节点，还需要多少期望值。
        return False

    def change_batch(self, batch: Time_Window):
        self.batch = batch
        # self.batch.cul_can_be_sent_load_plan_by_car()
        self.lp_len = max(len(self.batch.can_be_sent_load_plan), len(self.batch.car_list))
        self.ex_car = [0 for i in range(self.lp_len)]  # KM算法实现需要
        self.ex_huo = [0 for i in range(self.lp_len)]
        self.vis_car = [False for i in range(self.lp_len)]
        self.vis_huo = [False for i in range(self.lp_len)]
        self.match = [-1 for i in range(self.lp_len)]
        self.slack = [sys.maxsize for i in range(self.lp_len)]


    def km(self):
        car_num = len(self.batch.car_list)
        if car_num == 0:
            print('what...')
        LP_num = len(self.batch.can_be_sent_load_plan)
        weight_matrix = [[0 for i in range(max(car_num, LP_num))] for i in range(max(car_num, LP_num))]  # 扩充相等子图
        for i in range(car_num):        # 计算二部图之间的权值
            for j in range(LP_num):
                weight_matrix[i][j] = weight_value_calculate(self.batch.car_list[i], self.batch.can_be_sent_load_plan[j])
        matrix_len = max(car_num, LP_num)
        self.lp_len = matrix_len
        self.match = [-1 for i in range(self.lp_len)]  # 初始每个乘客都没有匹配到司机
        self.ex_huo = [0 for i in range(self.lp_len)]  # 顶标LY    初始每个司机的期望值为0
        self.ex_car = [0 for i in range(self.lp_len)]  # 顶标LX    初始化左边顶标为该左节点所连接边权最大值
        # 每个司机的初始期望值是与他相连的乘客的最大订单价值期望。
        for i in range(matrix_len):
            self.ex_car[i] = weight_matrix[i][np.argmax(weight_matrix[i])]

        # 尝试解决每个司机拿到订单
        for i in range(matrix_len):
            self.slack = [sys.maxsize for i in range(matrix_len)]        # 松弛量slack、 每次开始找增广路径初始化为无穷大
            count = 0     # 计算期望下降次数
            while True:
                # 每次匹配首先将两边初始化为未访问过
                self.vis_car = [False for i in range(matrix_len)]
                self.vis_huo = [False for i in range(matrix_len)]
                if self.dfs(i,matrix_len,weight_matrix):          #  表示该左边节点已经有匹配点
                    break
                else:                                             #  表示左边节点匹配失败， 开始降低期望值
                    d = sys.maxsize
                    for j in range(matrix_len):
                        if self.vis_huo[j] == False:
                            d = min(d, self.slack[j])             #  找到最小松弛量
                                                                  # 松弛操作
                    for j in range(matrix_len):
                        # 所有访问过的左边节点降低期望值、 使访问过的左边节点选择更多
                        if self.vis_car[j]:
                            self.ex_car[j] -= d
                        # 所有访问过的右边节点增加期望值、 使矛盾在下一子图中
                        if self.vis_huo[j]:
                            self.ex_huo[j] += d
                        # else:
                        #     self.slack[j] -= d

                # 松弛操作的四种情况
                #     1. 已经配对的两端都在交错树中、  Lx[i] + Ly[j]不变， 它仍然属于相等子图，现在还属于相等子图
                #     2. 两端都不属于交错树， Lx[i] + Ly[j]不变， 它不属于相等子图，现在还不属于相等子图
                #     3. 左边节点不在交错树中， 右边节点在， 即Lx不变，Ly变大， Lx[i] + Ly[j]增大。 他原来不属于相等子图，现在仍不属于相等子图
                #     4. 左边节点在交错树中， 右边节点不在， 即Lx变小，Ly不变， Lx[i] + Ly[j]增小。 他原来不属于相等子图，现在可能属于相等子图、 使得子图得到了扩大

        res = 0
        match_lp_list = [0 for i in range(car_num)]
        for i in range(matrix_len):
            if self.match[i] < car_num:
                match_lp_list[self.match[i]] = i
        for i in range(len(match_lp_list)):
            reward = self.cul_cumulative_reward(i, len(match_lp_list) - 1, match_lp_list, weight_matrix)
            res += reward
        res = res / len(self.batch.car_list)
        return res, self.match


    def cul_cumulative_reward(self, start_index, end_index, match_list:[], weight_matrix:[[]]):
        reward = 0
        for i in range(end_index - start_index + 1):
            # times = datetime.strptime(self.batch.time, "%Y%m%d%H%M%S") - datetime.strptime(self.batch.car_list[start_index].arrive_time, "%Y%m%d%H%M%S")
            #
            times = timedelta(seconds=60)
            reward += (pow(self.gama, math.ceil(times.seconds/60)) * weight_matrix[start_index][match_list[start_index]]) / 360
        return reward



