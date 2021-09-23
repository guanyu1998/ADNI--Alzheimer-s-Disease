import numpy as np
import os
import  random
'''
data_path: 一个包含数据集名字，标签的txt文件，其格式为 name+‘ ’+label+‘ ’+其它  
'''
#1:372,0:774
def s_k_fold(data_path,k):
    positive = []
    negative = []
    k_fold = []
    file1 = open(data_path)
    lines = file1.readlines()
    for line in lines:
        row = line.split(' ')
        if row[1] == '1':
            #positive.append(row[0]+' '+row[1])
            positive.append(row)
        if row[1] == '0':
            #negative.append(row[0]+' '+row[1])
            negative.append(row)
    '''每个k的样本数量
        pos样本数量
        negativate的样本数量
    '''
    k_num = (len(positive) + len(negative)) / k
    positive_num = int((len(positive) / (len(positive) + len(negative))) * k_num)
    negative_num = int((len(negative) / (len(positive) + len(negative))) * k_num)
    # print((positive_num,negative_num))
    # print((len(positive),len(negative)))
    '''打乱顺序
    '''
    random.seed(10)
    random.shuffle(positive)
    random.shuffle(negative)
    '''分选
    '''
    for i in range(k-1):
        # print((i*positive_num,(i+1)*positive_num-1))
        # print((i * negative_num, (i + 1) * negative_num - 1))
        k_fold.append(positive[i * positive_num:(i + 1) * positive_num - 1] + negative[i * negative_num:(i + 1) * negative_num - 1])
    #     print('------')
    # print('****')
    # print((k-1)*positive_num)
    # print((k-1)*negative_num)
    k_fold.append(positive[(k-1)*positive_num:-1]+negative[(k-1)*negative_num:-1])

    return k_fold









if __name__=='__main__':
    path = 'E:/new_data/data.txt'
    s_k_fold(path,k = 5)


