import gzip
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten

np.set_printoptions(threshold=np.inf)
# 导入数据（image及label）
with gzip.open('mnist.pkl.gz', 'rb') as f:
	train, valid, test = pickle.load(f, encoding='bytes')

x_train, y_train, x_test, y_test = train[0], train[1], test[0], test[1]


class Mnist_Model(Model):
	def __init__(self):
		super(Mnist_Model, self).__init__()
		self.flatten = Flatten()
		self.l1 = Dense(128, activation='relu')
		self.l2 = Dense(10, activation='softmax')

	def call(self, x):
		x = self.flatten(x)
		x = self.l1(x)
		y = self.l2(x)
		return y


model = Mnist_Model()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

ckpt_path = "./ckpt/mnist.ckpt"

if os.path.exists(ckpt_path + ".index"):
	print('-----load model-----')
	model.load_weights(ckpt_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                 save_weights_only=1,
                                                 save_best_only=1)

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=3,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback],verbose=1)

# # 打印模型所有可训练参数
# # print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
# 	file.write(str(v.name) + '\n')
# 	file.write(str(v.shape) + '\n')
# 	file.write(str(v.numpy()) + '\n')
# file.close()

# 显示训练集和测试集的acc,loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(acc, label='训练精度')
plt.show()