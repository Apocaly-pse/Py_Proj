import gzip
import os
import pickle

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten


np.set_printoptions(threshold=np.inf)
# 导入数据（image及label）
with gzip.open('mnist.pkl.gz', 'rb') as f:
	train, valid, test = pickle.load(f, encoding='bytes')

x_train, y_train, x_test, y_test = train[0].reshape(-1, 28, 28, 1), train[1], test[0].reshape(-1, 28, 28, 1), test[1]


class Conv_Mnist_Model(Model):
	def __init__(self):
		super(Conv_Mnist_Model, self).__init__()
		self.c1 = Conv2D(filters=8, kernel_size=(3, 3))
		self.b1 = BatchNormalization()
		self.a1 = Activation('relu')
		self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)
		self.d1 = Dropout(.2)
		self.f0 = Flatten()
		self.f1 = Dense(128, activation='relu')
		self.d2 = Dropout(.2)
		self.f2 = Dense(10, activation='softmax')

	def call(self, x):
		x = self.c1(x)
		x = self.b1(x)
		x = self.a1(x)
		x = self.p1(x)
		x = self.d1(x)
		x = self.f0(x)
		x = self.f1(x)
		x = self.d2(x)
		y = self.f2(x)
		return y


model = Conv_Mnist_Model()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

ckpt_path = "./ckpt/mnist_conv.ckpt"

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
                    callbacks=[cp_callback],
                    verbose=1)

# # 打印模型所有可训练参数
# # print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
# 	file.write(str(v.name) + '\n')
# 	file.write(str(v.shape) + '\n')
# 	file.write(str(v.numpy()) + '\n')
# file.close()

# # 显示训练集和测试集的acc,loss曲线
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.plot(acc, label='训练精度')
# plt.show()