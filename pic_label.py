import re
import os
#Step1
#划分训练用的数据集
#根据CelebA这个库中的标注文件，生成标签索引文件
att_path = '.\CelebA\Anno\list_attr_celeba.txt'
to_path_Train = '.\CelebA\Img\Train'
to_path_Val = '.\CelebA\Img\Val'
to_path_Test = '.\CelebA\Img\Test'

f = open(att_path, 'r')
line = f.readline()
j = 0
train_txt = open(r'.\CelebA\Img\train_txt.txt', 'w')
val_txt = open(r'.\CelebA\Img\val_txt.txt', 'w')
test_txt = open(r'.\CelebA\Img\test_txt.txt', 'w')
while line:
    a = re.split("' '|'  '", line)
    a =a[0].split()
    if j>=2:
        label = int(a[21])   # list_attr_celeba.txt文件的第22位是性别标签
    if j>=2 and j<=18000:  # 制作训练集标签
        train_txt.write(to_path_Train+os.sep+a[0])
        train_txt.write(' ')
        if label == -1:
            train_txt.write('0')
        elif label == 1:
            train_txt.write('1')
        train_txt.write('\n')

    elif j>162770 and j<=163770:  # 制作验证集标签
        val_txt.write(to_path_Val+os.sep+a[0])
        val_txt.write(' ')
        if label == -1:
            val_txt.write('0')
        elif label == 1:
            val_txt.write('1')
        val_txt.write('\n')

    elif j>182637 and j<=183637:  # 制作测试集标签
        test_txt.write(to_path_Test+os.sep+a[0])
        test_txt.write(' ')
        if label == -1:
            test_txt.write('0')
        elif label == 1:
            test_txt.write('1')
        test_txt.write('\n')
    line = f.readline()
    j+=1
