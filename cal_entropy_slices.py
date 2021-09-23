import numpy as np
import scipy.stats


#计算图像切片熵
def cal_entropy(medical_data):#输入为reshape处理后的silce的数据,是一个一维数据
    entroy = 0.0
    if medical_data.shape[0] == 0:
        return  entroy
    else:
        #piex = Counter(medical_data)
        medical_data = np.around(medical_data,decimals=4)#medical_data.astype(int)  只会变成0和1 ;numpy.around(x)四舍五入到给定的小数位;numpy.rint(x)四舍五入到最近整数;截取整数部分 np.trunc,向上取整 np.ceil,向下取整np.floor
        unique, count = np.unique(medical_data,return_counts=True)
        a = count/len(medical_data)
        b = np.log2(count/len(medical_data))
        entroy = -(a*b).sum()
        return entroy



