import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


# dataset = tf.data.Dataset.range(10)
# dataset = dataset.window(5, shift=1, drop_remainder=True)
# for window_dataset in dataset:
#     for val in window_dataset:
#         print(val.numpy(), end=" ")
#     print()

# 轉為numpy列表
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())


# 打散數據
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
    print(x.numpy(), y.numpy())

print("----------------------------")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10) # 打亂順序
for x,y in dataset:
    print(x.numpy(), y.numpy())


print("----------------------------")
# 設定數據批量，每兩個數據為一個批次
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10) # 打亂順序
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())