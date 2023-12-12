import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import RMSprop

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print(tf.__version__)

# ---------------建構模型---------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16,(3, 3), activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(32,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])


# ---------ImageDataGenerator---------
# 利用現有的資料經過旋轉、翻轉、縮放…等方式增加更多的訓練資料
# 宣告兩個數據生成器，指定範圍0~1
TRAINING_DIR = 'tmp/cats-v-dogs/training/'
train_datagen = ImageDataGenerator(rescale=1/255)
TESTING_DIR = 'tmp/cats-v-dogs/testing/'
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    TESTING_DIR,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

history = model.fit(
    train_generator, 
    epochs=2, 
    verbose=1, 
    validation_data=validation_generator)

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
plt.title('Training and validation accuracy')
plt.show()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()

# Desired output. Charts with training and validation metrics. No crash :)


# ---------------predicting images---------------
# D:\tensorflow_practice\tmp\cats-v-dogs\testing\cats\105.jpg
file_path = input("Enter the file path: ")
if os.path.exists(file_path):
    # 打开图像
    img = Image.open(file_path)    
    # 调整图像大小
    img = img.resize((150, 150))   
    # 将图像转换为数组
    x = img_to_array(img)
    # 在第一个维度上扩展数组
    x = np.expand_dims(x, axis=0)
    # 将数组堆叠起来
    images = np.vstack([x]) 
    # 使用模型进行预测
    classes = model.predict(images, batch_size=10)
    
    # 输出预测结果
    print(classes[0])
    if classes[0] > 0.5:
        print("It's a dog!")
    else:
        print("It's a cat!")
else:
    print("File not found.")
