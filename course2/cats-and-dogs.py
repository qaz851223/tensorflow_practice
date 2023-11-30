import os
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
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
    steps_per_epoch=224, 
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
    return

# ========數據增強========
img_path = 'tmp/img/horse/*'
in_path = 'tmp/img/'
out_path = 'tmp/img/output/'
name_list = glob.glob(img_path)
print_result(img_path)

# 調整大小
if not os.path.exists(out_path + 'resize'):
    os.makedirs(out_path + 'resize')
datagen = image.ImageDataGenerator()
gen_data = datagen.flow_from_directory(in_path, 
                                       batch_size=1, 
                                       shuffle=False, 
                                       save_to_dir=out_path+'resize',
                                       save_prefix='gen', 
                                       target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'resize/*')

# 角度旋轉
if not os.path.exists(out_path + 'rotation_range'):
    os.makedirs(out_path + 'rotation_range')
datagen = image.ImageDataGenerator(rotation_range=45)
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'rotation_range',
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'rotation_range/*')                                        

# 平移變換
if not os.path.exists(out_path + 'shift'):
    os.makedirs(out_path + 'shift')
datagen = image.ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'shift', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'shift/*')

# 縮放
if not os.path.exists(out_path + 'zoom'):
    os.makedirs(out_path + 'zoom')
datagen = image.ImageDataGenerator(zoom_range=0.5)
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'zoom', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'zoom/*')

# channel_shift
if not os.path.exists(out_path + 'channel'):
    os.makedirs(out_path + 'channel')
datagen = image.ImageDataGenerator(channel_shift_range=15) 
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'channel', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'channel/*')

# 翻轉
if not os.path.exists(out_path + 'horizontal'):
    os.makedirs(out_path + 'horizontal')
datagen = image.ImageDataGenerator(horizontal_flip=True) 
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'horizontal', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'horizontal/*')

# rescale
if not os.path.exists(out_path + 'rescale'):
    os.makedirs(out_path + 'rescale')
datagen = image.ImageDataGenerator(rescale=1/255) 
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'rescale', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'rescale/*')

# 填充方法
# 有時旋轉 平移...等會有空隙
# fill_mode => constant nearest reflect wrap
if not os.path.exists(out_path + 'fill_mode'):
    os.makedirs(out_path + 'fill_mode')
datagen = image.ImageDataGenerator(fill_mode='wrap', zoom_range=[4,4]) 
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'fill_mode', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'fill_mode/*')


if not os.path.exists(out_path + 'nearest'):
    os.makedirs(out_path + 'nearest')
datagen = image.ImageDataGenerator(fill_mode='nearest', zoom_range=[4,4]) 
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size = 1,
                                class_mode=None, 
                                shuffle=True, 
                                target_size=(224,224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, 
                                        batch_size=1, 
                                        shuffle=False,
                                        save_to_dir=out_path+'nearest', 
                                        save_prefix='gen', 
                                        target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path + 'nearest/*')