import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt

# ---------資料下載---------
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()


# ---------建構模型---------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.summary()
# 640 = (3*3+1)*64
# 36928 = (3*3*64+1)*64

# ---------訓練&評估模型---------
train_images_scaled = train_images/255
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(train_images_scaled, train_labels, epochs=5)

# ---------查看CNN裡的結構---------
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
pred = activation_model.predict(test_images[0].reshape(1, 28, 28, 1))
# plt.imshow(pred[0][0,:,:,2])
# plt.show()

layer_to_visualize = 0  # 選擇要視覺化的層
filter_index = 0  # 初始化濾波器索引

# 視覺化所選層中多個濾波器的激活情況
for filter_index in range(8):  # 視覺化8個濾波器
    plt.imshow(pred[layer_to_visualize][0, :, :, filter_index], cmap='viridis')
    plt.title(f'L {layer_to_visualize}, filter {filter_index}')
    plt.show()