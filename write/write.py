import csv
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter= ',')
        first_line = True # 弟一行不是數據 是表頭
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels

training_images, training_labels = get_data('tmp/sign/sign_mnist_train.csv')
testing_images, testing_labels = get_data('tmp/sign/sign_mnist_test.csv')

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# =============預處理=============
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale= 1/255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale= 1/255)

print(training_images.shape)
print(testing_images.shape)

# ---------------建構模型---------------
model = keras.Sequential()
model.add(keras.layers.Conv2D(64,(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(26, activation='softmax'))
model.summary()

model.compile(optimizer=tf.optimizers.Adam(),
              loss= 'sparse_categorical_crossentropy',
              metrics=['acc'])


history = model.fit(
    train_datagen.flow(training_images, training_labels, batch_size=32), 
    steps_per_epoch=len(training_images)/32,  #27455/32=857
    epochs=15, 
    validation_data=validation_datagen.flow(testing_images, testing_labels,batch_size=32), 
    validation_steps=len(testing_images)/32)  #7127/32=225左右

model.evaluate(testing_images, testing_labels)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




















