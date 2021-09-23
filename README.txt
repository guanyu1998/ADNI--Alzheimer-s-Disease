1.environment
	python 3.7
	pytorch 1.7.0
	numpy 1.19.2
	nibabel 3.2.1
	matplotlib 3.3.4
	imageio 2.9.0
2.introduction
	data.txt中包含我们所有样本的名称，以等号为分割，前一部分为ADNI原数据的默认编号，后边为日期，其对应的nii样例文件为 wm002_S_0295=2006-04-18.nii。
	resnet50+attention.py为resnet50+attention的训练文件；resnet+attention.py为resnet+attention的训练文件，文件运行前，请将path = 'E:/new_data/data.txt'的路径替换为您本机电脑上           data.txt文件的路径，root_dir = 'E:/new_data/data_file'请替换为您训练样本所在的父类文件夹的路径；name_path没什么用，可以删除；txt_path = r'D:/Model/2021-8/la.txt'用来记录每次训练	的数据。
 
