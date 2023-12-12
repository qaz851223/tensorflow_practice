import numpy as np
import os
import glob
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import layers
# import tensorflow.keras

# ==========圖項數據處理實例==========
image_labels = {
    '640px-Felis_catus-cat_on_snow': 0,
    'New_East_River_Bridge_from_Brooklyn_det.4a09796u': 1,
}

image_string = open('tmp/tfrecord/640px-Felis_catus-cat_on_snow.jpg', 'rb').read()
label = image_labels['640px-Felis_catus-cat_on_snow']

# ==========轉化方式==========
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


# ==========創建圖項數據的Example==========
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    # 創建tf.Example
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

image_example_proto = image_example(image_string, label)

for line in str(image_example_proto).split('\n')[:15]:
    print(line)
print("...")


img_path = 'tmp/tfrecord/'
images = glob.glob(img_path + '*.jpg')
recode_file = 'images.tfrecord'
counter = 0

with tf.io.TFRecordWriter(recode_file) as writer:
    for fname in images:
        with open(fname, 'rb') as f:
            image_string = f.read()
            # print(os.path.basename(fname).replace('.jpg', ''))
            label = image_labels[os.path.basename(fname).replace('.jpg', '')]
          
            tf_example= image_example(image_string, label)

            writer.write(tf_example.SerializeToString())
            counter += 1
            print("Processed {:d} of {:d} images".format(counter, len(images)))

print("Wrote {} images to {}".format(counter, recode_file)) 

# ==========加載tfrecord文件==========
# filenames = [recode_file]
# 讀取
raw_train_dataset = tf.data.TFRecordDataset('images.tfrecord')
print(raw_train_dataset)

# 解析之前寫入的序列化string
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64), 
    'width': tf.io.FixedLenFeature([], tf.int64), 
    'depth': tf.io.FixedLenFeature([], tf.int64), 
    'label': tf.io.FixedLenFeature([], tf.int64), 
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}

# 解析 TFRecord 数据集中的单个样本
def parse_tf_example(example_proto):
    # 解析出來
    parse_example = tf.io.parse_single_example(example_proto, image_feature_description)
    # 預處理
    x_train = tf.image.decode_jpeg(parse_example['image_raw'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))
    x_train /= 255.

    label = parse_example['label']
    y_train = label

    return x_train, y_train

# 透過map解析 應用於整個數據集，得到新的數據集train_dataset
train_dataset = raw_train_dataset.map(parse_tf_example)
print(train_dataset)

# ==========製作訓練集==========
num_epochs = 10
train_ds = train_dataset.shuffle(buffer_size=10000).batch(2).repeat(num_epochs)
print(train_ds)

for batch, (x, y) in enumerate(train_ds):
    print(batch, x.shape, y)

model = tf.keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['acc'])
model.fit(train_ds, epochs=num_epochs)