import tensorflow as tf
import tensorflow.keras 
import matplotlib.pyplot as plt
import numpy as np
import os


# ---------資料下載---------
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# 數據及布包刮名稱類，之後繪製圖像時使用
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_labels[:5])
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# print(test_images.shape)
# print(len(test_labels))
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# ---------預處理---------
train_images = train_images / 255
test_images = test_images / 255

# 查看驗證數據的格式是否正確
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# ---------建構模型---------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

# ---------訓練&評估模型---------
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
# ---------評估準確率---------
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

prediction = model.predict(test_images)
print(prediction.shape)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(np.argmax(probability_model.predict([[test_images]])[0]))
print(test_images[0])
plt.imshow(test_images[0])
plt.show()

# ---------保存模型---------
model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save(filepath='model/fashion_model.h5')
# ---------只保存網路架構---------
config = model.to_json()
print(config)
with open('model/config.json', 'w') as json:
    json.write(config)

with open('model/config.json', 'r') as json:
    json_config = json.read()

model = tf.keras.models.model_from_json(json_config)
model.summary()
# ---------權重參數---------
weights = model.get_weights()
print(weights)
model.save_weights('model/weights.h5')
model.load_weights('model/weights.h5')