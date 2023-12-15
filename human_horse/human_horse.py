'''
還沒寫好
'''

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras_tuner.tuners import Hyperband
from tensorflow.keras_tuner.engine.hyperparameters import HyperParameters


# class myCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('loss')<0.8):
#             print("\nLoss is low so cancelling training!")
#             self.model.stop_training = True
# callbacks = myCallback()

# ---------ImageDataGenerator---------
# 利用現有的資料經過旋轉、翻轉、縮放…等方式增加更多的訓練資料
# 宣告兩個數據生成器，指定範圍0~1
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'tmp/horse-or-human/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    'tmp/horse-or-human/validation',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')

# ---------建構模型---------
# 調參數
hp = HyperParameters()
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Choice('num_filters_layer0', values=[16, 64], default=16), (3, 3), activation='relu', input_shape=(300,300,3)))
    model.add(tf.keras.layers.MaxPool2D(2,2))
    for i in range(hp.Int("num_conv_layers",1,3)):
        model.add(tf.keras.layers.Conv2D(hp.Choice(f'num_filters_layer{i}',values=[16, 64], default=16), (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2,2))  
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hp.Int("hidden_units", 128, 512, step=32), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['acc'])
    model.summary()

    return model

tuner = Hyperband(
    build_model, 
    objective = 'val_acc',
    max_epochs= 15,
    directory='human_horse_params', 
    hyperparameters=hp,
    project_name='my_horse_human_project'
)

tuner.search(train_generator, epochs=10, validation_data=validation_generator)
# history = model.fit(
#     train_generator, 
#     epochs=15, 
#     verbose=1, 
#     validation_data=validation_generator, 
#     validation_steps=8)

best_hps = tuner.get_best_hyperparameters(1)[0]
print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
model.summary()
