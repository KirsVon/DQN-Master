# -*- coding: utf-8 -*-
# Description:
# Created: liujiaye  2019/12/09
import math
import numpy as np


def sigmoid(x, use_status=True):
    if use_status:
        return 1.0 / (1 + np.exp(-float(x)))
    else:
        return float(x)


# 归一化工具（my）
class ZScore:
    def __init__(self, sample):
        '''
        :param entity:样本数据  type:list
        '''
        if len(sample) == 0:
            print("sample is null")
            self.u = 0
            self.o = 1
        else:
            sample_sum = sum(data for data in sample)
            self.u = float(sample_sum / len(sample))  # 样本数据均值
            self.o = math.sqrt(sum(data - self.u for data in sample) / len(sample))  # 样本数据的标准差

    def z_score(self, data):
        return (float(data) - self.u) / self.o


# if __name__ == '__main__':
#     sample = [1, 213, 4, 12, 32, -213, 0, -3, -1, -42]
#     result = sigmoid(sample[6])
#     print(type(result))
#     print(result)
