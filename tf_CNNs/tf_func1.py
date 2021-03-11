# coding:utf-8
import tensorflow as tf
import numpy as np


# x = tf.constant([[1,2,3], [4,5,6]], dtype=tf.float32)
# print(x)

# # tf.cast强制类型转换
# y = tf.cast(x, tf.int64)
# print(y)
#
# # 计算张量维度上的最值
# print(tf.reduce_min(y))
# print(tf.reduce_max(y))

'''
output:
tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32)
tf.Tensor([1 2 3], shape=(3,), dtype=int64)
tf.Tensor(1, shape=(), dtype=int64)
tf.Tensor(3, shape=(), dtype=int64)
'''

# # 指定轴计算平均值,0轴为经度(按列计算),1轴为纬度(按行计算)
# # 不指定的话则对张量的所有值进行操作
# print(tf.reduce_mean(x, axis=0))

'''
tf.Tensor(
[[1. 2. 3.]
 [4. 5. 6.]], shape=(2, 3), dtype=float32)
tf.Tensor([2.5 3.5 4.5], shape=(3,), dtype=float32)
'''


# # 将变量标记为"可训练的",被标记的变量在反向传播中会记录梯度信息
# # 权重向量的初始化
# w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
# print(w)

'''<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[ 0.9372736 ,  0.49235305],
       [-0.00218096,  0.5778176 ]], dtype=float32)>'''

# # 张量对应元素的四则运算
# # 维度相同才可以进行四则运算
# tf.add()
# tf.subtract()
# tf.multiply()
# tf.divide()

# # 平方、开方与次方(对张量的每个元素进行操作),int类型执行开方时候会报错
# x = tf.constant([1,2,3.])
# print(tf.square(x), tf.sqrt(x), tf.pow(x, 3))

# # 矩阵乘法(二维张量)
# m = tf.fill([2, 3], 3)
# n = tf.fill([3, 1], 4)
# print(tf.matmul(m, n))

# # 张量的配对
# feature = tf.constant([1,2,3,4])
# label = tf.constant([3,4,4,2])
#
# dataset = tf.data.Dataset.from_tensor_slices((feature, label))
# print(dataset)
# print([elem for elem in dataset])


# # 求导计算(张量的梯度)
# # with结构记录计算过程
# with tf.GradientTape() as tape:
# 	w = tf.Variable(tf.constant(13.))
# 	loss = tf.pow(w, 3)
#
# grad = tape.gradient(loss, w)
# print(grad)

# # one-hot编码
# print(tf.one_hot([1,2,8], depth=9))

# # 柔性最大值函数softmax
# print(tf.nn.softmax([2.,1,3]))

# # 自减操作
# print(tf.Variable(4.3).assign_add(20))

# 索引最大值
# print(tf.argmax(np.array([[1,2,4], [3,5,8]]), axis=0))

