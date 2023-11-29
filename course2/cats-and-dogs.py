import os

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras

base_dir = 'tmp\cats-and-dogs'
train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')


