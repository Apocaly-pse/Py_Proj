import gzip
import pickle
import tensorflow as tf

# 导入数据（image及label）
with gzip.open('mnist.pkl.gz', 'rb') as f:
	train, valid, test = pickle.load(f, encoding='bytes')

x_train, y_train, x_test, y_test = train[0], train[1], test[0], test[1]

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=0),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test), validation_freq=1)
# model.summary()
