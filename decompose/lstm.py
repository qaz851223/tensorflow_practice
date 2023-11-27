import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras


# --------------模擬生成時間序列--------------
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi), 
                    1 / np.exp(3 * season_time))

def seasonality(time, peroid, amplitude=1, phase=0):
    season_time = ((time + phase) % peroid) / peroid
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# --------------create the series--------------
# 沒有誤差的序列
series = baseline + trend(time, slope) + seasonality(time, peroid=365, amplitude=amplitude)
# 有誤差的序列(noise)
series += noise(time, noise_level, seed=42)

# 劃分數據 train valid
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20 
batch_size = 32
shuffle_buffer_size = 1000

# --------------模擬生成數據集--------------
# parameters :序列數據，窗口大小，批次大小，隨機緩存大小
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# --------------搭建LSTM神經網路 使用LearningRateScheduler機制調整學習率--------------
# tf.keras.backend.clear_session()
# tf.random.set_seed(51)
# np.random.seed(51)

# tf.keras.backend.clear_session()
# dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.Lambda(lambda x: x * 100.0))

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-8 * 10**(epochs / 20))
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9) # SGD隨機梯度下降法
# model.compile(optimizer=optimizer, loss='Huber', metrics=['mae'])
# model.summary()
# history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

# plt.semilogx(history.history['lr'], history.history['loss']) # 取對數
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show() # 相較RNN，雙向LSTM的誤差曲線平滑許多


# --------------搭建LSTM神經網路 不調整學習率--------------
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))) #太多層有時效果不好
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Lambda(lambda x: x * 100.0))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9) # SGD隨機梯度下降法
model.compile(optimizer=optimizer, loss='Huber', metrics=['mae'])
model.summary()
history = model.fit(dataset, epochs=100)

forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

print(keras.metrics.mean_absolute_error(x_valid, results).numpy())

mae = history.history['mae']
loss = history.history['loss']
epochs = range(len(loss))

plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('Mae and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Mae", "Loss"])
plt.show()

epochs_zoom = epochs[20:]
mae_zoom = mae[20:]
loss_zoom = loss[20:]

plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('Mae and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Mae", "Loss"])
plt.show()