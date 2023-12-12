import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
data = open('tmp/irish-lyrics-eof.txt', encoding='utf-8').read()

corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0] # 對每一行進行序列化
    for i in range(1, len(token_list)): # 以下 對每個句子產成不同長度的序列
        n_gram_sequence = token_list[:i+1]  #ex 某句有6個單字 從1.2/1.2.3/....
        input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences]) # 找到句子中最長的長度
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# 除最後一個字符以外的所有字符作為輸入x，最後一個字符作為標籤y => 產生神經網路的input label
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
# 獨熱編碼
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words) 

# ---------------建構模型---------------
# Hyperparameters
embedding_dim = 150
lstm_units = 150
learning_rate = 0.01
epochs = 200

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, embedding_dim, input_length=max_sequence_len - 1)) #-1 是因為最後一個是標籤y
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))) # 雙向LSTM層
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(xs, ys, epochs=epochs, verbose=1)



def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
  
plot_graphs(history, "acc")
# plot_graphs(history, "loss")

seed_text = "I've got a bad feeling about this"
# seed_text = "Help Me Obi-Wan Kenobi, you're my only hope"
next_words = 150

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_classes = np.argmax(predicted, axis=1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_classes[0]:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)