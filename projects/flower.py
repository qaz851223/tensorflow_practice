from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical


dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
y = to_categorical(y) # 讀熱編碼
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(4,)))
model.add(tf.keras.layers.Dense(4, activation='relu', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.Dense(6, activation='relu', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='glorot_uniform'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(train_x, train_y, batch_size=16, epochs=300)

train_loss, train_accuracy = model.evaluate(train_x, train_y)
print('Train Loss:', train_loss)
print('Train Accuracy:', train_accuracy)

test_loss, test_accuracy = model.evaluate(test_x, test_y)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

predictions = model.predict(test_x)
for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    actual_class = np.argmax(test_y[i])
    print(f'Sample {i+1}: Predicted class = {predicted_class}, Actual class = {actual_class}')