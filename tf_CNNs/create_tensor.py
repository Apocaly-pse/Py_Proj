import tensorflow as tf
import numpy as np


# a = tf.constant([1, 5], dtype=tf.int64)
# print(a)
# print(a.dtype)
# print(a.shape)

# a = np.ones((1, 2))
# b = tf.convert_to_tensor(a, dtype=tf.int64)
# print(a)
# print(b)

# a = tf.zeros((2, 3))
# b = tf.ones([2, 2])
# 填充任意指定值的张量
# c = tf.fill([2, 3], 900)
# print(c)

# # 生成正态分布的随机数(参数为维度)
# d = tf.random.normal([2, 2], mean=.5, stddev=1)
# # 生成截断式正态分布的随机数(更加集中,2σ)
# e = tf.random.truncated_normal([2, 2], mean=.5, stddev=1)
# print(d)
# print(e)

# # 生成符合均匀分布的随机数(指定维度和区间)
# f = tf.random.uniform([2, 3], minval=1, maxval=3)
# print(f)
