import pandas as pd
import numpy as np
import copy

# 定义LLR类
class llr_class:
    def __init__(self, data, lag):
        self.data = copy.deepcopy(data)
        self.lag = copy.deepcopy(lag)
    def get_stock_list(self):
        Data = copy.deepcopy(self.data)
        stock_list = list(Data['name'].drop_duplicates())
        return stock_list
    def llr_pair(self, stock_X, stock_Y):
        pct_X = stock_X[['DateTime', 'pctchange']]
        pct_Y = stock_Y[['DateTime', 'pctchange']]
        pct_X.set_index('DateTime', inplace=True)    # 设置DateTime列为index，这个时候你不用去管原先的index，pct_X原先的index是0-959，pct_Y原先的index是960-1919，但他们的DateTime都是0-959
        pct_Y.set_index('DateTime', inplace=True)
        sum_pos = 0
        sum_neg = 0
        for k in np.arange(1, self.lag+1):
            pct_pos_lag = pct_X.shift(k)    #pct_X的所有数据向下移动k行，缺失的数据用nan代替
            pct_neg_lag = pct_X.shift(-k)
            corr_pos = pct_Y['pctchange'].corr(pct_pos_lag['pctchange'])   # 问题出在这两列计算出来的相关系数是nan
            corr_neg = pct_Y['pctchange'].corr(pct_neg_lag['pctchange'])
            sum_pos = sum_pos + corr_pos ** 2
            sum_neg = sum_neg + corr_neg ** 2
        llr = sum_pos / sum_neg
        return llr
    def llr_group(self):
        Data_group = self.data.groupby('name')
        stock_list = self.get_stock_list()
        llr_result = pd.DataFrame(np.ones((len(stock_list), len(stock_list))), index=stock_list, columns=stock_list)
        # 初始化llr_result表  >>print(llr_result)
        #          600519   568    600257
        # 600519    1.0     1.0     1.0
        # 568       1.0     1.0     1.0
        # 600257    1.0     1.0     1.0
        for i in np.arange(len(stock_list)-1):   #i=0,j=1,2; i=1,j=2;
            for j in np.arange(i+1, len(stock_list)):
                stock_X = Data_group.get_group(stock_list[i])
                stock_Y = Data_group.get_group(stock_list[j])
                llr = self.llr_pair(stock_X, stock_Y)
                llr_result.iloc[i:i+1, j:j+1] = llr    # 取第i行j列，赋值为llr，取第j行i列，赋值为1/llr
                llr_result.iloc[j:j+1, i:i+1] = 1/llr
        return llr_result
    def llr_winnum(self, llr_result):
        llr_result = copy.deepcopy(llr_result)
        win_num = np.sum(llr_result>1, axis=1)
        win_num.sort_values(ascending=False, inplace=True)
        return win_num
    def __call__(self):
        self.get_stock_list
        self.llr_pair
        self.llr_group
        self.llr_winnum
# 标记龙头股
csv_list = ['baijiu.csv']
lag = 1    # 相关系数滞后后期仅考虑1期
percent = 0.4    # 标记龙头股的为板块前40%
index_month1st = [0, 960, 2016, 3119]
for k in range(len(csv_list)):
    csv_name = csv_list[k]
    Data = pd.read_csv(csv_name, index_col=0, header=0, sep=None, engine='python')
    # index_col=0，在读csv的时候就指定了第一列作为索引
    sample_time = Data['DateTime'].drop_duplicates()    # 时间戳，会保留第一列和DataTime列
    for i in np.arange(1, len(index_month1st)-1):    # 1是因为需要一个月数据才可以估计
        index_end = index_month1st[i]
        index_start = index_month1st[i-1]
        #print(sample_time)
        cal_time = sample_time[index_start:index_end]
        # DateTime这一列里为期1个月的数据
        cal_Data = Data[Data.DateTime.isin(cal_time)]
        cal_Data.reset_index(drop=True, inplace=True)    
        #修改排序，drop=True，把原来的索引index列去掉，inplace=True，不创建新对象，直接对原始对象进行修改
        #就是说，你一开始第一列index是960 961 ... 5131, reset_index后变为0 1 ... 2111
        llr_example = llr_class(cal_Data, lag)
        llr_map = llr_example.llr_group()
        print('llr_map=\n', llr_map)
        winnum = llr_example.llr_winnum(llr_map)
        print('winnum=\n',winnum)
        leading_stock = list(winnum.index[0:int(len(winnum)*percent)])
        print('the leading_stock of this month is', leading_stock)
        mark_start = index_month1st[i]
        mark_end = index_month1st[i+1]
        mark_time = sample_time[mark_start:mark_end]
        condition = (Data.DateTime.isin(mark_time))&(Data.name.isin(leading_stock))
        Data.loc[condition, 'mark'] = 1
        print('第%s个月已完成'%i)
        Data['mark'].fillna(0, inplace=True)
        Data.to_csv('%s标记.csv'%k)
        print('第%s个板块完成'%k)

# 龙头股预测能力检验（逻辑回归）
#         


        
        
    
