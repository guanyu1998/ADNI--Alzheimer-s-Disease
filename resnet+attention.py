import os
CUDA_VISIBLE_DEVICES=1
import torch
from  torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset
import  numpy as np
from time import *
import torchvision.models.resnet
import torch.optim as optim
from collections import Counter
import torch.optim.lr_scheduler as lr_scheduler
import math
from ADNI_dataset import ADNI_Data
from Stratifield_K_fold import s_k_fold
import copy

#root_dir = 'D:/new_data/train/train'
# data_file = 'D:/new_data/train/train_1.txt'
# root_dir_1 = 'D:/new_data/valid/valid'
# data_file_1 = 'D:/new_data/valid/valid_1.txt'

# root_dir = r'/home/gy/E/new_data/train/train'
# data_file = r'/home/gy/E/new_data/train/train_1.txt'
# root_dir_1 = r'/home/gy/E/new_data/valid/valid'
# data_file_1 = r'/home/gy/E/new_data/valid/valid_1.txt'


silce = 8 #8 ,16 32 ,64
#resnet 50
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 8*2048*1*1
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 8*2048*1*1
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)


        return self.sigmoid(x)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channel,out_channel,stride = 1,downsample = None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size= 1,stride = 1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x) :
        identity = x
        if self.downsample is not None: #not None表示对应着虚线残差结构，也就是w和h要变小的哪一个残差块，就是每个CONV_的第一个残差块；None为实线残差结构，wh已经改变了之后
            identity = self.downsample(x)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self,block,block_num,num_class=2,this_channel=16,include_top = True,):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.this_channel =this_channel

        self.conv0 = nn.Conv2d(self.this_channel,self.in_channel,kernel_size=(7,7),stride=2,padding=3,bias=False)
        self.bn0 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=False)

        self.layer1 = self._make_layer(block,64,block_num[0])
        self.layer2 = self._make_layer(block,128,block_num[1],stride =2)
        self.layer3 = self._make_layer(block,256,block_num[2],stride = 2)
        self.layer4 = self._make_layer(block,512,block_num[3],stride = 2)

        self.ca1 = ChannelAttention(512)#2048为最后的通道数
        self.sa1 = SpatialAttention()



        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #OUT_put= (1,1)1
            self.fc = nn.Linear(512*block.expansion,num_class)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def _make_layer(self,block,channel,block_num,stride = 1):#channel是每个残差结构第一个卷积层对应的卷积核个数
        downsample = None
        if stride!= 1 or self.in_channel!= channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel*block.expansion,stride = stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel,channel,downsample= downsample,stride=stride))

        self.in_channel = channel*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)#输入也可以是list,然后输入的时候用*来引用

    def forward(self,x):
        x =self.conv0(x)
        x =self.bn0(x)
        x =self.relu(x)
        x = self.maxpool(x)

        x= self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x= self.layer4(x)

        x = self.ca1(x) * x
        x = self.sa1(x) * x

        if self.include_top:
            x= self.avgpool(x)
            x = torch.flatten(x,1)
            x= self.fc(x)

        return x


def resnet50(num_class = 2,include_top=True,this_channel=16):
    return ResNet(Bottleneck,[3,4,6,3],num_class=num_class,include_top=include_top,this_channel=this_channel)




#resnet34
def resnet34(num_class=2, include_top=True,this_channel=16):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class,include_top=include_top,this_channel=this_channel)
#net = resnet34(num_class=2,this_channel=8 * 3)


#resnet18
def resnet18(num_class=2, include_top=True,this_channel=16):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class=num_class,include_top=include_top,this_channel=this_channel)
#net = resnet18(num_class=2,this_channel=8 * 3)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net = resnet50(num_class = 2,this_channel=32*3)

#model_weight_path = r'D:/Model/resnet34-pre.pth'
model_weight_path = r'/home/gy/D/Model/resnet18-pre.pth'

'''
测试


net = resnet50(num_class=2, this_channel=32 * 3)
print(net)

pre_weight = torch.load(model_weight_path,map_location=device)
del_key = []
for key,v in pre_weight.items():
    if "fc" in key:
         del_key.append(key)
for key in del_key:
    del pre_weight[key]

missing_keys, unexpected_keys = net.load_state_dict(pre_weight, strict=False)#missing_keys 网络中的结构但是加载参数中却没有；unexpected_keys 表示

print(missing_keys)
# net.to(device)
'''


ep_all = 300
batch_size = 16
lr = 0.0001

# txt_path = r'D:/Model/2021-8/la.txt'
# name_path = r'E:/new_data/name.txt'
txt_path = r'/home/gy/D/Model/res50/self-attention/res18/res18+att.txt'
name_path = r'/home/gy/D/Model/res50/self-attention/res34/res18+att-group.txt'
#txt_path = r'/home/gy/D/Model/res50/k-fold/silce-32/res-32.txt'
f = open(txt_path,'w')
file = open(name_path,'w')

# ep_train= []
#
# loss_list = []
# acc_list = []
# val_loss = []
# train_acc= []


#trian

# path = 'E:/new_data/data.txt'
# root_dir = 'E:/new_data/data_file'

root_dir = '/home/gy/E/news_data/data_file'
path = '/home/gy/E/news_data/data.txt'
k = 20
k_fold_list = s_k_fold(path,k=k)


for k_fold in range(k):
    k_fold_list_copy= copy.deepcopy(k_fold_list)
    test_list =k_fold_list_copy[k_fold]
    del k_fold_list_copy[k_fold]
    train_train = ADNI_Data(root_dir,train_list = k_fold_list_copy,silce=silce,packed = True)
    valid_valid = ADNI_Data(root_dir, train_list = test_list, silce = silce, packed = True,train= False)
    file.write('valid:'+str(k_fold)+' '+str(test_list)+'\n')
    train_dataset = torch.utils.data.DataLoader(train_train, shuffle=True, batch_size=batch_size)
    valid_dataset = torch.utils.data.DataLoader(valid_valid, shuffle=True, batch_size=batch_size)
    net = resnet18(num_class=2, this_channel=8 * 3)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.000)
    criterion = nn.CrossEntropyLoss()
    lf = lambda x: ((1 + math.cos(x * math.pi / ep_all)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    pre_weight = torch.load(model_weight_path, map_location=device)
    del_key = []
    for key, v in pre_weight.items():
        if "fc" in key:
            del_key.append(key)
    for key in del_key:
        del pre_weight[key]

    missing_keys, unexpected_keys = net.load_state_dict(pre_weight, strict=False)
    net.to(device)
    for epoch in range(ep_all):
        net.train()
        running_loss = 0.0
        validing_loss = 0.0
        num_iter = 0.0
        correct_ = 0.0
        total_ = 0.0
        sofmax = nn.Softmax(dim=1)
        for i, data in enumerate(train_dataset, 0):
            num_iter += 1
            input, label,name = data
            input = input.to(device)
            label = label.to(device)
            label = label.long()
            optimizer.zero_grad()
            output = net(input)
            output = sofmax(output)

            _, pre = torch.max(output.data, dim=1)
            correct_ += (pre == label).sum().item()
            total_ += label.size(0)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            #打印训练进程
            rate_ = (i+1)/len(train_dataset)
            a = "*"*int(rate_*50)
            b = "."*int((1-rate_)*50)
            print("\r train  : {:^3.0f}%[{}->{}]".format(int(rate_*100),a,b),end ="")
        print("this k_fold is %d,this epoch is :%d,train loss:" % (k_fold,epoch), running_loss / num_iter)
        #loss_list.append(running_loss / num_iter)
        #train_acc.append(correct_/total_)

        # valid
        #if epoch % 5 == 0:
        net.eval()
        TP = 0.0
        FN = 0.0
        FP = 0.0
        TN = 0.0
        correct = 0.0
        total = 0.0
        num_num = 0.0
        max_acc = 0.0
        with torch.no_grad():
            for i, data in enumerate(valid_dataset, 0):
                num_num += 1
                input, label,name  = data
                input = input.to(device)
                label = label.to(device)
                label = label.long()
                output = net(input)
                output = sofmax(output)
                _, pre = torch.max(output.data, dim=1)

                loss = criterion(output, label)
                validing_loss += loss.item()

                correct += (pre == label).sum().item()
                total += label.size(0)

                for id in range(len(label)):
                    if int(label[id].item()) == 1 and int(pre[id].item()) == 1:
                        TP += 1
                    if int(label[id].item()) == 1 and int(pre[id].item()) == 0:
                        FN += 1
                    if int(label[id].item()) == 0 and int(pre[id].item()) == 1:
                        FP += 1
                    if int(label[id].item()) == 0 and int(pre[id].item()) == 0:
                        TN += 1
        Accuracy = (correct / total)
        #val_loss.append(validing_loss/num_num)
        print("valid loss:" , validing_loss/num_num)
        if float(correct / total) > max_acc:  # correct/total为此次准确率，当此次准确率大于最大准确率的时候保存模型，并且将当前acc赋值给max_loss
            torch.save(net.state_dict(), f=r'/home/gy/D/Model/res50/self-attention/res18/res18-att-'+str(k_fold)+'.pth', _use_new_zipfile_serialization=False)
            max_acc = float(correct/total)

        f.write('k-fold:'+str(k_fold)+' '+'epoch:' + str(epoch) + ' ' + 'accuracy:' + str(Accuracy) + ' ' +
                'TP:' + str(TP) + ' ' + 'FN:' + str(FN) + ' ' + 'FP:' + str(FP) + ' ' + 'TN:' + str(
            TN) + ' ' + 'correct:' + str(correct) + ' ' + 'total:' + str(total) +' '+'train loss:'+str(running_loss / num_iter)+' '+'valid loss:'+str(validing_loss/num_num)+ '\n')
        #acc_list.append(Accuracy)
f.close()
#
# # fig1 = plt.figure(num=1, figsize=(8, 5))
# # fig1.suptitle('model')
# # plt.xlabel('epoch')
# # plt.ylabel('loss')
# # plt.plot(ep_train, loss_list, color='r', linewidth='1.0', label='train_loss')
# # plt.plot(ep_train, val_loss, color='b', linewidth='1.0', label='val_loss')
# # plt.savefig(r'/home/gy/D/Model/res50/res50-32-a.png')
# # plt.legend()
# # fig2 = plt.figure(num=2, figsize=(8, 5))
# # fig2.suptitle('model')
# # plt.xlabel('epoch')
# # plt.ylabel('accuracy')
# # plt.plot(ep_train, acc_list, color='r', linewidth='1.0', label='val_accuracy')
# # plt.plot(ep_train, train_acc, color='b', linewidth='1.0', label='train_accuracy')
# # plt.savefig(r'/home/gy/D/Model/res50/res50-32-b.png')
# # plt.legend()
# # plt.show()



