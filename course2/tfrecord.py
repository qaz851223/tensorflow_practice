import numpy as np

import tensorflow as tf

# ==========轉化方式==========
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


print(_bytes_feature(b'test_string'))
print(_bytes_feature('test_string'.encode('utf-8')))
print(_float_feature(np.exp(1)))
print(_int64_feature(True))
print(_int64_feature(1))

# ==========製作方法==========
def serialize_example(feature0, feature1, feature2, feature3):
    # 創建tf.Example
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3)
    }
    # 使用tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    # SerializeToString方法轉換為二進制字符串
    return example_proto.SerializeToString()

n_observations = int(1e4)
feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings =np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

filename = 'tfrecord-1.tfrecord'
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

# ==========加載tfrecord文件==========
filenames = [filename]
# 讀取
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)