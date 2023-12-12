import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
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

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

# ---------------預處理---------------
# ImageDataGenerator
# 利用現有的資料經過旋轉、翻轉、縮放…等方式增加更多的訓練資料
# 宣告兩個數據生成器，指定範圍0~1
TRAINING_DIR = 'tmp/cats-and-dogs/training'
VALIDATION_DIR = 'tmp/cats-and-dogs/validation'
train_datagen = ImageDataGenerator(rescale=1/255, 
                                    rotation_range=40,
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

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
    steps_per_epoch=225, 
    epochs=10, 
    verbose=2, 
    validation_steps=25,
    validation_data=validation_generator)

def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(12, 16))
    for i in range(min(3, len(name_list))):
        img = Image.open(name_list[i])
        sub_img = fig.add_subplot(131+i)
        sub_img.imshow(img)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and Validation accuracy')
plt.show()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()