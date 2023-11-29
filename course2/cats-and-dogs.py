import os

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras



TRAINING_CATS_DIR = 'tmp/cats-and-dogs/training/cats'
TRAINING_DOGS_DIR = 'tmp/cats-and-dogs/training/dogs'

VALIDATION_CATS_DIR = 'tmp/cats-and-dogs/validation/cats'
VALIDATION_DOGS_DIR = 'tmp/cats-and-dogs/validation/dogs'

# ---------------建構模型---------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3, 3), activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])


