import tensorflow as tf


# print(tf.__version__)

# tf.constant(张量内容, dtype=数据类型(可选))用于创建一个张量
a = tf.constant([1.0, 2.0], name='a', dtype=tf.float32)
b = tf.constant([2.0, 2.9], name='b', dtype=tf.float32)

ret = tf.add(a, b, name='add')

print(ret)
