import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tmp/sarcasm.json','r') as f:
    datastore = json.load(f)

sentences = []
labels = []
# urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    # urls.append(item['article_link'])

# hyperparameter
vocab_size = 1000 # 10000(有overfitting現象 調成1000，減少字典的單字量)
embedding_dim = 16 # 第二次調成32 結果影響比較小
max_len = 16  # 32
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"
training_size = 20000 # total:27000，test=7000

# 拆分訓練資料及測試資料
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# 创建 Tokenizer 并对文本序列进行转换和填充
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index # 查看字典中的數據

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, 
    maxlen=max_len, 
    padding=padding_type, 
    truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, 
    maxlen = max_len, 
    padding=padding_type, 
    truncating=trunc_type)
# print(len(word_index))
# print("\nWord Index= ", list(word_index.items())[:10])
# print(sentences[2])
# print(padded[2])
# print(padded.shape)

# ---------------建構模型---------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

history = model.fit(training_padded,
                    training_labels, 
                    epochs=30,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
plot_graphs(history, "acc")
plot_graphs(history, "loss")