import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from keras.preprocessing import image
from keras.utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
# from keras_tuner.tuners import Hyperband
# from keras_tuner.engine.hyperparameters import HyperParameters

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_horse_dir = 'tmp/horse-or-human/train/horses'
train_human_dir = 'tmp/horse-or-human/train/humans'

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
# print('total training horse images:', len(os.listdir(train_horse_dir)))
# print('total training human images:', len(os.listdir(train_human_dir)))

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4
# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

# pic_index += 8
# # 顯示前8張
# next_horse_pix = [os.path.join(train_horse_dir, fname) 
#                 for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname) 
#                 for fname in train_human_names[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_horse_pix + next_human_pix):
#     # Set up subplot; subplot indices start at 1
#     sp = plt.subplot(nrows, ncols, i + 1)
#     sp.axis('Off') # Don't show axes (or gridlines)

#     img = mpimg.imread(img_path)
#     plt.imshow(img)
# plt.show()


# ---------------建構模型---------------
model = keras.Sequential()
model.add(keras.layers.Conv2D(16,(3, 3), activation='relu', input_shape=(300,300, 3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32,(3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])
model.summary()

# ---------ImageDataGenerator---------
# 利用現有的資料經過旋轉、翻轉、縮放…等方式增加更多的訓練資料
# 宣告兩個數據生成器，指定範圍0~1
TRAINING_DIR = 'tmp/horse-or-human/train/'
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary')

history = model.fit(
    train_generator, 
    steps_per_epoch=8,
    epochs=2, 
    verbose=1)

for data_batch, labels_batch in train_generator:
    print('Data batch shape:', data_batch.shape)
    print('Labels batch shape:', labels_batch.shape)
    break

# ---------------predicting images---------------
# D:\tf_practice\tmp\horse-or-human\train\humans\human01-03.png
# file_path = input("Enter the file path: ")
# if os.path.exists(file_path):
#     # 打开图像
#     img = Image.open(file_path)    
#     # 调整图像大小
#     img = img.convert("RGB")
#     img = img.resize((300, 300))   
#     # 将图像转换为数组
#     x = img_to_array(img)
#     # 在第一个维度上扩展数组
#     x = np.expand_dims(x, axis=0)
#     # 将数组堆叠起来
#     images = np.vstack([x]) 
#     # 使用模型进行预测
#     classes = model.predict(images, batch_size=128)
    
#     # 输出预测结果
#     print(classes[0])
#     if classes[0] > 0.5:
#         print("It's a human!")
#     else:
#         print("It's a horse!")
# else:
#     print("File not found.")



# 创建可视化模型： 创建了一个新模型（visualization_model），该模型以图像为输入，并输出原始模型中除第一层外所有层的中间表示。
# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

# 预处理图像： 加载选择的图像，将其转换为数组，并调整其形状以匹配模型的输入形状。
# 然后通过除以255对其进行归一化。
img = load_img(img_path, target_size=(300,300))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
y = x.reshape((1,) + x.shape)
x /= 255

# 生成特征图
successive_feature_maps = visualization_model.predict(x)

# 可视化特征图
layer_names = [layer.name for layer in model.layers] # 模型中所有层的名称
print(layer_names)

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:  # 检查特征图的维度
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map
        # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[1]
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]
            if x.std() != 0:  # 添加标准差不为零的检查
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x
            else:
                # 如果标准差为零，将图像置为全零
                display_grid[:, i * size: (i + 1) * size] = 0

        # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()