import gzip
import pickle

import tensorflow as tf
# import tensorflow.keras

# read the file in read mode as binary 
with gzip.open('tmp/mnist.pkl.gz', 'rb') as file_contents:
    ((x_train, y_train,), (x_valid, y_valid ), _)= pickle.load(file_contents, encoding='latin1')

# =========建構模型=========
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_valid, y_valid))
model.summary()

# =========重新訓練=========
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.batch(32)
train = train.repeat()

valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid = train.batch(32)
valid = train.repeat()

model.fit(train, epochs=5, 
                 steps_per_epoch=100, 
                 validation_data=valid, 
                 validation_steps=100)
