# 导入解压及数据读取库（系统自带）
import gzip
import pickle
# 导入图形处理库
from PIL import Image
# 导入矩阵处理库
import numpy as np
# pyplot用于显示图形
# （如果使用Image内置的`.show()`方法会使用系统默认的图形查看器打开，速度较慢）
import matplotlib.pyplot as plt

# 首先进行解压，以字节形式读取
f = gzip.open('mnist.pkl.gz', 'rb')
# 读取pkl文件（Python特有的的数据形式）
# 需要设定编码，否则会报错
# 数据集中有三个部分，分别是训练集、验证集和测试集，本示例程序以a,b,c代替
a, b, c = pickle.load(f, encoding='latin1')
f.close()

# print(len(a[0]),len(b[0]),len(c[0]));exit()
# print(a[0][0]);exit()

# 选择打开第几(n)个图片
n = 20
# 根据索引选择图片，并改变形状（图片的原始形状为784*1）
arr = a[0][n].reshape(28, 28)
# 使用Image的`.fromarray()`方法读入图片矩阵
# 注意这里图片的灰度值是以0-1之间的float32类型存储的，需要先转换成RGB数值（0-255）
# 再对已读入的图片进行转换（否则会显示为彩色）
img = Image.fromarray(np.uint8(arr * 255)).convert("1")
# 使用`pyplot`的函数显示图像
plt.imshow(img)
plt.show()
# print(img.mode)
print("图片上的数字是：%s" % (a[1][n]))
