import numpy as np
import math
import torch
import collections
from collections import Counter
from nibabel.viewers import OrthoSlicer3D
import imageio
import matplotlib.pyplot as plt
import scipy.stats#计算统计中的熵，不能直接用于图像熵
import  cal_entropy_slices
import  os
import nibabel as nib
#利用四，八，十分位数去除每一个患者MRI的离群值
#输入位npy文件
def deciles(img_npy,num):#img_npy是一个三维数组
    # 从上向下看
    x_list = []
    z_list = []
    y_list = []
    for i in range(img_npy.shape[0]):  #新数据从左到右
        silce = img_npy[i,:,:].reshape(-1)

        ten_per = np.percentile(silce,10)
        ninth_per =np.percentile(silce,90)
        silce = silce[(silce>ten_per)&(silce<ninth_per)]
        #计算香农熵之前记得审查[]是否为0
        entroy = cal_entropy_slices.cal_entropy(silce)
        x_list.append(entroy)
    a = np.array(x_list)
    x_silce_num = np.argsort(-a)[0:num]


    #从前向后看 #新数据从前向后
    for i in range(img_npy.shape[1]):
        silce = img_npy[:,i,:].reshape(-1)

        ten_per = np.percentile(silce,10)
        ninth_per =np.percentile(silce,90)
        silce = silce[(silce>ten_per)&(silce<ninth_per)]
        entroy = cal_entropy_slices.cal_entropy(silce)
        y_list.append(entroy)
    b = np.array(y_list)
    y_silce_num = np.argsort(-b)[0:num]
    #从左向右看 新数据从上向下
    for i in range(img_npy.shape[2]):
        silce = img_npy[:,:,i].reshape(-1)
        #imageio.imwrite('E:/image/' + str(i) + '.png', img_npy[:, :, i])
        ten_per = np.percentile(silce,10)
        ninth_per =np.percentile(silce,90)
        silce = silce[(silce>ten_per)&(silce<ninth_per)]
        entroy = cal_entropy_slices.cal_entropy(silce)
        z_list.append(entroy)
    c = np.array(z_list)
    z_silce_num = np.argsort(-c)[0:num]
    return x_silce_num,y_silce_num,z_silce_num#x从上往下看，y 从前向后看，z从左向右看

def csv_process(path,txt_path,new_text):
    dir = os.listdir(path)
    f1 = open(txt_path,'r')
    f2 = open(new_text,'w')
    lines = f1.readlines()
    for line in lines:
        list = []
        file_name = line.split(' ')[0]
        print(file_name)
        label = line.split(' ')[1]
        label = label.rstrip('\n')
        file = nib.load(path+'/'+'wm'+file_name+'.nii')
        list.append(file_name)
        list.append(label)
        img_npy =file.get_fdata()#得到numpy文件
        ''' num = 8'''
        x, y, z = deciles(img_npy, num=8)
        k1 = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1]) + '-' + str(x[-1])
        l1 = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1]) + '-' + str(y[-1])
        m1 = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1]) + '-' + str(z[-1])
        list.append(k1)
        list.append(l1)
        list.append(m1)
        ''' num = 16'''
        x, y, z = deciles(img_npy, num=16)
        k2 = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1]) + '-' + str(x[-1])
        l2 = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1]) + '-' + str(y[-1])
        m2 = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1]) + '-' + str(z[-1])
        list.append(k2)
        list.append(l2)
        list.append(m2)
        ''' num = 32'''
        x, y, z = deciles(img_npy, num=32)
        k3 = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1])+'-'+str(x[-1])
        l3 = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1]) + '-' + str(y[-1])
        m3 = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1]) + '-' + str(z[-1])
        list.append(k3)
        list.append(l3)
        list.append(m3)
        ''' num = 64'''
        x, y, z = deciles(img_npy, num=64)
        k4 = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1])+'-'+str(x[-1])
        l4 = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1]) + '-' + str(y[-1])
        m4 = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1]) + '-' + str(z[-1])
        list.append(k4)
        list.append(l4)
        list.append(m4)
        f2.write(file_name+' '+' '.join(u for u in list[1:len(list)])+'\n')
    f1.close()
    f2.close()






    # for name in dir:
    #     print(name)
    #     img_npy = np.load(path+'/'+name)
    #     x, y, z = deciles(img_npy, num=10)
    #
    #     k = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1])+'-'+str(x[-1])
    #     #df['10-top2down'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     #df['top2down'] = df.ID.apply(lambda o:str(x[0])+' '.join([str(z) for z in x[1:-1]]) if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0],'10-top2down'] =k
    #     '''
    #     print("name,silce:",[name,k])
    #     print(df.loc[df['10-top2down']==k,'ID'])
    #     print(df.loc[df['ID']== name.split('.')[0],'10-top2down'])
    #     '''
    #
    #     l = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1])+'-'+str(y[-1])
    #     #df['10-front2back'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '10-front2back'] = l
    #     '''
    #     print("name,silce:", [name, l])
    #     print(df.loc[df['10-front2back'] == l, 'ID'])
    #     '''
    #
    #     m = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1])+'-'+str(z[-1])
    #     #df['10-left2right'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '10-left2right'] = m
    #     '''
    #     print([name, m])
    #     print(df.loc[df['10-left2right'] == m, 'ID'])
    #     '''
    #     #16
    #     x, y, z = deciles(img_npy, num=16)
    #     k = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1])+'-'+str(x[-1])
    #     #df['16-top2down'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '16-top2down'] = k
    #
    #     l = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1])+'-'+str(y[-1])
    #     #df['16-front2back'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '16-front2back'] = l
    #
    #     m = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1])+'-'+str(z[-1])
    #     #df['16-front2back'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '16-left2right'] = m
    #
    #     #32
    #     x, y, z = deciles(img_npy, num=32)
    #     k = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1])+'-'+str(x[-1])
    #     # df['32-top2down'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '32-top2down'] = k
    #
    #     l = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1])+'-'+str(y[-1])
    #     # df['32-front2back'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '32-front2back'] = l
    #
    #     m = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1])+'-'+str(z[-1])
    #     # df['32-left2right'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '32-left2right'] = m
    #
    #     #64
    #     x, y, z = deciles(img_npy, num=64)
    #     k = str(x[0]) + '-' + '-'.join(str(u) for u in x[1:-1])+'-'+str(x[-1])
    #     # df['64-top2down'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '64-top2down'] = k
    #
    #     l = str(y[0]) + '-' + '-'.join(str(u) for u in y[1:-1])+'-'+str(y[-1])
    #     # df['64-front2back'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '64-front2back'] = l
    #
    #     m = str(z[0]) + '-' + '-'.join(str(u) for u in z[1:-1])+'-'+str(z[-1])
    #     # df['64-left2right'] = df.ID.apply(lambda o: 0 if o == name else 0)
    #     df.loc[df['ID'] == name.split('.')[0], '64-left2right'] = m
    # df.to_csv('E:/data/train/sample_valid_label.csv',index = False)


if __name__ == '__main__':
    path = 'D:/new_data/valid/valid'
    txt_path = 'D:/new_data/valid/valid.txt'
    new_text = 'D:/new_data/valid/valid_1.txt'
    x_ = []
    y_ = []
    z_ = []
    csv_process(path,txt_path,new_text)
    #img_npy = np.load('E:/data/train/002_S_0295=2006-04-18.npy')#256 *256* 166
    #x,y,z = deciles(img_npy,num =10)


#写个程序保存到csv文件
#绘制几个matplot直方图
