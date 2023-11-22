import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# ---------資料下載---------
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# print(train_labels[:5])
# plt.imshow(train_images[1])
# plt.show()


# ---------建構模型---------
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.summary()
# 100480 = (784+1)*128    +1代表輸入層跟中間層都+bias
# 1290 = (128+1)*10    輸出層10個類別(0~9)

# ---------訓練&評估模型---------
train_images = train_images/255
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(train_images,train_labels, epochs=5)

test_images_scaled = test_images/255
model.evaluate(test_images_scaled, test_labels)

# print(model.predict([[test_images/255]])[0])
# print(np.argmax(model.predict([[test_images/255]])[0]))
# print(test_images[0])
# plt.imshow(test_images[0])
# plt.show()

