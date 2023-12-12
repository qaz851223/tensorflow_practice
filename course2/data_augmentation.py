from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(12, 16))
    for i in range(min(3, len(name_list))):
        img = Image.open(name_list[i])
        sub_img = fig.add_subplot(131+i)
        sub_img.imshow(img)
    plt.show()

# ========數據增強========
img_path = 'tmp/img/horse/*'
in_path = 'tmp/img/'
out_path = 'tmp/img/output/'
# name_list = glob.glob(img_path)
print_result(img_path)

# 調整大小
if not os.path.exists(out_path + 'resize'):
    os.makedirs(out_path + 'resize')
datagen = ImageDataGenerator()
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
datagen = ImageDataGenerator(rotation_range=45)
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(zoom_range=0.5)
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(channel_shift_range=15) 
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(horizontal_flip=True) 
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(rescale=1/255) 
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(fill_mode='wrap', zoom_range=[4,4]) 
gen = ImageDataGenerator()
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
datagen = ImageDataGenerator(fill_mode='nearest', zoom_range=[4,4]) 
gen = ImageDataGenerator()
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