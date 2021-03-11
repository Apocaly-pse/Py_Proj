import tensorflow as tf
import numpy as np

# # 返回[0, 1)之间的随机数,维度为(2, 3)
# print(np.random.RandomState(seed=1).rand(2, 3))

# 生成网格坐标点
x, y = np.mgrid[1:3:1, 2:4:.5]
# .ravel()用于拉直array
# np.c_[]用于配对array元素
grid = np.c_[x.ravel(), y.ravel()]
print(x, '\n', y)
print(grid)