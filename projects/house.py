import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

mean = train_x.mean(axis=0)
train_x -= mean
std = train_x.std(axis=0)
train_x /= std

test_x -= mean
test_x /= std

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(train_x.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

epochs = 500
history = model.fit(train_x,
                    train_y,
                    batch_size=16,
                    epochs=epochs,
                    validation_data=(test_x, test_y),
                    shuffle=True)

predictions = model.predict(test_x)
plt.figure(figsize=(10, 5))
plt.scatter(test_y, predictions)
plt.xlabel('real')
plt.ylabel('predict')
plt.title('real vs. predict')
plt.show()

plt.plot(np.arange(epochs), history.history['mae'], c='b', label='train_mae')
plt.plot(np.arange(epochs), history.history['val_mae'], c='y', label='val_mae')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('mae')
plt.show()