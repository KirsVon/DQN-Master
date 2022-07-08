import operator  # 导入operator 包,pip install operator

Departs = []  # 待排序列表


class Department:  # 自定义的元素
    def __init__(self, id, name, id2):
        self.id = id
        self.name = name
        self.id2 = id2


# 创建元素和加入列表
Departs.append(Department(1, 'cbc', '1'))
Departs.append(Department(2, 'acd', '4'))
Departs.append(Department(3, 'bcd', '1'))
Departs.append(Department(1, 'bcd', '1'))
Departs.append(Department(2, 'acd', '3'))

# 划重点#划重点#划重点----排序操作
cmpfun = operator.attrgetter('id2')  # 参数为排序依据的属性，可以有多个，这里优先id，使用时按需求改换参数即可
Departs.sort(key=cmpfun)  # 使用时改变列表名即可
# 划重点#划重点#划重点----排序操作

# 此时Departs已经变成排好序的状态了，排序按照id优先，其次是name，遍历输出查看结果
for depart in Departs:
    print(str(depart.id) + depart.name + depart.id2)