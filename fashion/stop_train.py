import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

# ---------資料下載---------
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()


# ---------建構模型---------
train_images_scaled = train_images/255
test_images_scaled = test_images/255
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# ---------訓練&評估模型---------

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(train_images_scaled, train_labels, epochs=5, callbacks=[callbacks])

