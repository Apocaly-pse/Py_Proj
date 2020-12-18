import numpy as np
import gzip
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD

# 导入数据（image及label）
with gzip.open('mnist.pkl.gz', 'rb') as f:
    training, validation, test = pickle.load(f, encoding='latin1')

# 训练数据(total:50000)
train_images = np.array([img.reshape(28, 28) for img in training[0][:6000]])
train_labels = training[1][:6000]
# 测试数据(total:10000)
test_images = np.array([img.reshape(28, 28) for img in test[0][:1000]])
test_labels = test[1][:1000]

train_images = train_images - 0.5
test_images = test_images - 0.5

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

model = Sequential([
    Conv2D(8, 3, input_shape=(28, 28, 1), use_bias=False),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(10, activation='softmax'),
])

print("开始训练模型")

# 编译模型，使用随机梯度下降算法（SGD,学习率为.005）
model.compile(
    SGD(lr=.005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(
    train_images, to_categorical(train_labels),
    shuffle=True,
    batch_size=10,
    epochs=1,
    verbose=1,
    validation_data=(test_images, to_categorical(test_labels)),
)
