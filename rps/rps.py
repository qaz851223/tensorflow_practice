import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog

import tensorflow as tf
# import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

rock_dir = os.path.join('tmp/Rock-Paper-Scissors/train/rock')
paper_dir = os.path.join('tmp/Rock-Paper-Scissors/train/paper')
scissors_dir = os.path.join('tmp/Rock-Paper-Scissors/train/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
# print(rock_files[:10])
paper_files = os.listdir(paper_dir)
# print(paper_files[:10])
scissors_files = os.listdir(scissors_dir)
# print(scissors_files[:10])

pic_index = 2

# next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index-2:pic_index]]
# next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index-2:pic_index]]
# next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[pic_index-2:pic_index]]

# for i, img_path in enumerate(next_rock+next_paper+next_scissors):
#     #print(img_path)
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#     plt.axis('Off')
#     plt.show()

# =============預處理=============
TRAINING_DIR = 'tmp/Rock-Paper-Scissors/train'
train_datagen = ImageDataGenerator(
    rescale= 1/255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, 
    fill_mode="nearest")

VALIDATION_DIR = 'tmp/Rock-Paper-Scissors/test'
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, target_size=(150,150), class_mode='categorical'
)
validation_generator =validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=(150,150), class_mode='categorical'
)

# =============建構模型=============
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3, 3), activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
              loss= 'categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose=1)
# model.save('rps.h5') 

# =============畫圖=============
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()
# plt.show()

# 隐藏主窗口
root = tk.Tk()
root.withdraw()

# 通过文件选择对话框获取文件路径
file_paths = filedialog.askopenfilenames(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

if file_paths:
    for file_path in file_paths:
        # 加载图像并进行预处理
        img = load_img(file_path, target_size=(150, 150))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # 根据模型的训练进行归一化

        # 进行预测
        classes = model.predict(x, batch_size=1)

    print("Predicted classes:", classes)
else:
    print("No file selected.")