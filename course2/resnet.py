import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet101

TRAINING_DIR = 'tmp/cats-and-dogs/training'
VALIDATION_DIR = 'tmp/cats-and-dogs/validation'


TRAINING_CATS_DIR = 'tmp/cats-and-dogs/training/cats'
TRAINING_DOGS_DIR = 'tmp/cats-and-dogs/training/dogs'

VALIDATION_CATS_DIR = 'tmp/cats-and-dogs/validation/cats'
VALIDATION_DOGS_DIR = 'tmp/cats-and-dogs/validation/dogs'


pre_trained_model = ResNet101(input_shape=(224,224,3),
                                include_top=False, # 不要最後的全連接層
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')<0.8):
            print("\nReached '80%' accuracy so cancelling training!")
            self.model.stop_training = True

# 為全連接層做準備
x = layers.Flatten()(pre_trained_model.output)
# 加入全連接層，這個需要從頭訓練
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
# 輸出層
x = layers.Dense(1, activation='sigmoid')(x)
# 建構模型序列
model = Model(pre_trained_model.input, x)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', 
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255, 
                                    rotation_range=40,
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224, 224),
    batch_size=100,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=100,
    class_mode='binary')

callbacks = myCallback()
history = model.fit(
    train_generator, 
    steps_per_epoch=len(train_generator), 
    epochs=10, 
    verbose=2, 
    validation_steps=25,
    validation_data=validation_generator,
    callbacks=[callbacks])

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs
#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and Validation accuracy')
plt.show()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()