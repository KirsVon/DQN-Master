# -*- coding: utf-8 -*-
# Description:工具
# Created: liujiaye  2019/07/09

# from app.main.dao.management_dao import management_dao
import config
import pandas as pd
curr_config_class = config.get_active_config()


class CommodityTransform:
    '''
    大小品名转换工具
    '''

    def __init__(self):
        data = pd.read_csv(curr_config_class.PROD_CHANGE)
        commodities = {}
        for index, row in data.iterrows():
            commodities[row['prod_kind']] = row['prod_kind_price_out']
        self.commodity_dic = commodities

    def change_to_big(self, commodity):
        if commodity in self.commodity_dic.keys():
            return self.commodity_dic[commodity]
        return None


commodity_transform = CommodityTransform()

# if __name__ == '__main__':
#     res = pd.read_csv('C:/Users/93742/Desktop/bancheng.csv')
#     prod_name = list(res['prod_name'])
#     commodity_transform = CommodityTransform()
#     commodity_list = []
#     for i in range(len(prod_name)):
#         name = commodity_transform.change_to_big(prod_name[i])
#         commodity_list.append(name)
#
#     res['commodity'] = commodity_list
#     res.dropna(subset=['commodity'], inplace=True)
#     res.to_csv('C:/Users/93742/Desktop/chengpin.csv')