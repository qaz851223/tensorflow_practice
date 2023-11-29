import os

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator



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



TRAINING_DIR = 'tmp/cats-and-dogs/training'
VALIDATION_DIR = 'tmp/cats-and-dogs/validation'
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

history = model.fit(
    train_generator, 
    step_per_epoch=224, 
    epochs=20, 
    verbose=2, 
    validation_steps=25,
    validation_data=validation_generator)