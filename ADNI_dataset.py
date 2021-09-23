from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib
import imageio
import  matplotlib.pyplot as plt
from collections import Counter
import  torch
from Stratifield_K_fold import s_k_fold
import copy
'''
构建三种dataset：
1.单通道，模型预测三个方向的数值取平均
2.三通道，三个方向合并成一张图
3.第一次使用的一张图作为一个样本
'''
'''

'''
def getPackedSlices(image_array, mean_direc, fraction, section):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[0])
    last_p = list(mean_array).index(filter(lambda x: x>0, mean_array)[-1])
    slice_i = int(round(first_p + (last_p - first_p)*fraction))
    slice_p = slice_i
    # Middle slice - R Channel
    slice_select_R = eval("image_array"+section)/1500.0*255
    zero_slice = np.zeros(slice_select_R.shape)
    slice_select_R = np.stack((slice_select_R, zero_slice ,zero_slice), axis = 2)
    slices_G = np.zeros(slice_select_R.shape)
    slices_B = np.zeros(slice_select_R.shape)
    # Above middle slice - G Channel
    for slice_i in range(slice_p - 20, slice_p, 2):
        slice_select_G = eval("image_array"+section)/1500.0*255
        slice_select_G = np.stack((zero_slice, slice_select_G, zero_slice), axis = 2)
        slices_G += slice_select_G*0.1
    # Below middle slice - B Channel
    for slice_i in range(slice_p + 2, slice_p + 22, 2):
        slice_select_B = eval("image_array"+section)/1500.0*255
        slice_select_B = np.stack((zero_slice, zero_slice, slice_select_B), axis = 2)
        slices_B += slice_select_B*0.1
    slice_2Dimg = slice_select_R + slices_G + slices_B
    return slice_2Dimg

def zero_padding(img,size):#size nxn
    if img.shape[0]!=size[0]:
        w = (size[0]-img.shape[0])//2
        img = np.pad(img,((w,w),(0,0)),'constant',constant_values=0)
    if img.shape[1]!=size[1]:
        h = (size[1] - img.shape[1]) // 2
        img = np.pad(img, ((0,0),(h, h)),'constant', constant_values=(0,0))
    else:
        pass
    return img
#121 145 121
def silce3(image, left2right_idx, front2back_idx, up2down_idx, silce):
    image = image.get_fdata()
    idx_1 = up2down_idx.split('-')
    idx_2 = front2back_idx.split('-')
    idx_3 = left2right_idx.split('-')
    #返回3通道拼接
    img_3c = np.zeros([silce*3,145,145])
    for i in range(silce):
        z1 = zero_padding(image[:,:,int(idx_1[i])],(145,145))
        z2 = zero_padding(image[:,int(idx_1[i]),:],(145,145))
        z3 = zero_padding(image[int(idx_1[i]),:,:],(145,145))
        img_3c[i,:,:] = z1
        img_3c[i+silce*1,:,:]= z2
        img_3c[i+silce*2,:,:]= z3
    return img_3c

def silce1(image, left2right_idx, front2back_idx, up2down_idx, silce):
    image = image.get_fdata()
    idx_1 = up2down_idx.split('-')
    idx_2 = front2back_idx.split('-')
    idx_3 = left2right_idx.split('-')
    img_up2down = np.zeros([silce, 145, 145])
    img_front2back = np.zeros([silce, 145, 145])
    img_left2right = np.zeros([silce, 145, 145])
    for i in range(silce):
        z1 = zero_padding(image[:, :, int(idx_1[i])], (145, 145))
        z2 = zero_padding(image[:, int(idx_2[i]), :], (145, 145))
        z3 = zero_padding(image[int(idx_3[i]), :, :], (145, 145))
        img_up2down[i, :, :] = z1
        img_front2back[i , :, :] = z2
        img_left2right[i , :, :] = z3
    return  img_front2back,img_front2back,img_left2right




class ADNI_Data(Dataset):
    def __init__(self,root_dir,train_list,silce,packed =True,train =True):
        self.root_dir = root_dir
        self.train_list = train_list
        self.silce = silce
        self.packed =packed

        if train:
            self.list = self.train_list[0]
            for i in range(1,len(self.train_list)):
                self.list = self.list + self.train_list[i]
        else:
            self.list = self.train_list


        '''
        Args:
        root_dir (string): Directory of all the images.
        data_file (string): File name of the train/test split file.
        '''

    def __len__(self):

        # num = sum(1 for line in open(self.data_file))
        # return  sum(1 for line in open(self.data_file))
        num = len(self.list)
        return num

    def __getitem__(self, item):
        # df = open(self.data_file)
        # lines = df.readlines()
        lst = self.list[item]
        img_name = lst[0]
        img_label = int(lst[1])
        if self.silce == 8:
            left2right_idx = lst[2]
            front2back_idx = lst[3]
            up2down_idx = lst[4]
        elif self.silce == 16:
            left2right_idx = lst[5]
            front2back_idx = lst[6]
            up2down_idx = lst[7]
        elif self.silce == 32:
            left2right_idx = lst[8]
            front2back_idx = lst[9]
            up2down_idx = lst[10]
        elif self.silce == 64:
            left2right_idx = lst[11]
            front2back_idx = lst[12]
            up2down_idx = lst[13]
        else:
            print('silce_num {} is wrong'.format(self.silce))
         #拼接路径，读取文件
        image_path = self.root_dir+'/'+'wm'+img_name+'.nii'
        image = nib.load(image_path)
        #拼接文件
        if self.packed == False:
            img = silce1(image,left2right_idx,front2back_idx,up2down_idx,self.silce)
            img = torch.from_numpy(img).to(torch.float32)
            label = np.array((img_label,img_label,img_label))
            label = torch.from_numpy(label).to(torch.float32)
        else:
            img = silce3(image,left2right_idx,front2back_idx,up2down_idx,self.silce)
            img = torch.from_numpy(img).to(torch.float32)
            label = torch.from_numpy(np.array(img_label)).to(torch.float32)

        return img,label,img_name


if __name__=='__main__':
    path = 'E:/new_data/data.txt'
    root_dir = 'E:/new_data/data_file'
    silce = 32
    k = 5
    k_fold_list = s_k_fold(path, k=k)
    for i in range(k):
        k_fold_list_copy = copy.deepcopy(k_fold_list)
        test_list = k_fold_list_copy[i]
        del k_fold_list_copy[i]
    # root_dir = 'E:/new_data/data_file'
    # data_file = 'E:/new_data/data.txt'
        dataset = ADNI_Data(root_dir, train_list=test_list,silce=silce ,packed=True,train=False)
        img =  dataset.__getitem__(10)
        break




