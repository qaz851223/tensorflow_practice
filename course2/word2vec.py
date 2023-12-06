import collections
import os
import random
import urllib
import zipfile
import numpy as np

import tensorflow as tf

learning_rate = 0.1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 20000


eval_words = ['nine', 'of', 'going', 'american', 'britain']
# word2vec 參數
embedding_size = 200 # 詞向量維度
max_vocabulary_size = 50000 # 語料庫詞語數
min_occurrence = 100 # 最小詞頻
skip_window = 3 # 左右窗口大小
num_skips = 2 # 一次製作多少輸入輸出對
num_sampled = 64 # 負採樣

data_path = 'tmp/text8.zip'
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()
print(len(text_words))
# 創建一個計數器，計算每個詞出現多少次
count = [('UNK', -1)]
# 基於詞頻返回max_vocabulary_size個常用詞
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size-1))
print(count[0:10])

# 剔除出現少於min_occurrence次的詞
for i in range(len(count)-1, -1, -1):
    if count[i][1] < min_occurrence: # 從start(49999)到end每次step多少
        count.pop(i)
    else:
        # 判斷時，從小到大排序的，所以跳出時剩下的都是滿足條件的
        break

# 詞ID映射
vocabulary_size = len(count)
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i
# print(word2id)

# 所有詞換成ID
data = list()
unk_count = 0
for word in text_words:
    # 全部轉換成ID
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))

print("words count:", len(text_words))
print("unique words:", len(set(text_words)))
print("vocabulary_size:", vocabulary_size)
print("most common words", count[:10])


# ==========建構所需訓練數據==========
data_index = 0

def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1 # 7為窗口，左3右3中間1
    buffer = collections.deque(maxlen=span) #創建一個長度為7的隊獵
    if data_index + span > len(data): # 如果數據被滑完一遍
        data_index = 0
    buffer.extend(data[data_index : data_index + span]) # 隊列裡存的是當前窗口，例如deque([5234, 3081, 12, 6, 195, 2, 3134], maxlen=7)
    data_index += span
    for i in range(batch_size // num_skips): #num_skips 表示取多少組不同詞作為輸出，此例為2
        context_words = [w for w in range(span) if w != skip_window] #上下文就是[0, 1, 2, 4, 5, 6]
        words_to_use = random.sample(context_words, num_skips) #在上下文裡隨機挑選2個候選詞
        for j, context_words in enumerate(words_to_use): #遍歷每一個候選詞，用其當作輸出友就是標籤
            batch[i * num_skips + j] = buffer[skip_window] # 輸入都為當前窗口的中間詞，即3
            labels[i * num_skips + j, 0] = buffer[context_words] #用當前候選詞當作標籤
        if data_index == len(data):
            buffer.extend(data[0 : span])
            data_index = span
        else:
            buffer.append(data[0:span])
            data_index += 1
    data_index = (data_index + len(data) - sapn) % len(data)
    return batch, labels

with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# ==========通過tf.nn.embedding_lookup函數將索引轉換成詞向量==========
def get_embedding(x):
    with tf.device('/cpu:0'):
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed

# ==========損失函數定義==========
def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, 
                                             biases=nce_biases, 
                                             labels=y, 
                                             inputs=x_embed, 
                                             num_sampled=num_sampled, #採樣出多少個負樣本
                                             num_classes=vocabulary_size))
        return loss

# ==========測試觀察模塊==========
# Evaluation
def evaluate(x_embed):
    with tf.device('/cpu:0'):
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed))) # 歸一化
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32) # 全部向量的
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True) # 計算餘弦相似度
        return cosine_sim_op
# SGD
optimizer = tf.optimizers.SGD(learning_rate)
# 迭代優化
def run_optimization(x, y):
    with tf.device('/cpu:0'):
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)
        # 計算梯度
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])
        # 更新
        optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))


