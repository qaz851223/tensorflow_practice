import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

data = "In the town of Athy one Jeremy Lanigan\n"
data += "Battered away 'til he hadn't a pound.\n"
data += "His father he died and made him a man again\n"
data += "Left him a farm and ten acres of ground.\n"
data += "He gave a grand party to friends and relations\n"
data += "Who didn't forget him when it comes to the will,\n"
data += "If you'll but listen I'll make your eyes glisten\n"
data += "Of the rows and the ructions of Lanigan's Ball.\n"
data += "Myself to be sure got free invitation, \n"
data += "For all the nice girls and boys I might ask, \n"
data += "And just in a minute both friends and relations\n"
data += "Were dancing round merry as bees 'round a cask.\n"
data += "Judy O'Daly, that nice little milliner, \n"
data += "She tipped me a wink for to give her a call, \n"
data += "And I soon arrived with Peggy McGilligan\n"
data += "Just in time for Lanigan's Ball.\n"
data += "There were lashings of punch and wine for the ladies,\n"
data += "Potatoes and cakes; there was bacon and tea,\n"
data += "There were the Nolans, Dolans, O'Gradys\n"
data += "Courting the girls and dancing away.\n"
data += "Songs they went round as plenty as water,\n"
data += "'The harp that once sounded in Tara's old hall,'\n"
data += "'Sweet Nelly Gray' and 'The Rat Catcher's Daughter,'\n"
data += "All singing together at Lanigan's Ball.\n"
data += "Boys were all merry and the girls they were hearty \n"
data += "And danced all around in couples and groups, \n"
data += "'Til an accident happened, young Terrance McCarthy \n"
data += "Put his right leg through Miss Finnerty's hoops.\n"
data += "Poor creature fainted and cried, 'Meelia murther', \n"
data += "Called for her brothers and gathered them all.\n"
data += "Carmody swore that he'd go no further \n"
data += "'Til he had satisfaction at Lanigan's Ball.\n"
data += "In the midst of the row miss Kerrigan fainted, \n"
data += "Her cheeks at the same time as red as a rose.\n"
data += "Some of the lads declared she was painted, \n"
data += "She took a small drop too much, I suppose.\n"
data += "Her sweetheart, Ned Morgan, so powerful and able, \n"
data += "When he saw his fair colleen stretched out by the wall, \n"
data += "Tore the left leg from under the table\n"
data += "And smashed all the Chaneys at Lanigan's Ball.\n"
data += "Boys, oh boys, 'twas then there were runctions.\n"
data += "Myself got a lick from big Phelim McHugh.\n"
data += "I soon replied to his introduction \n"
data += "And kicked up a terrible hullabaloo.\n"
data += "Ould Casey, the piper, was near being strangled.\n"
data += "They squeezed up his pipes, bellows, chanters and all.\n"
data += "The girls, in their ribbons, they got all entangled \n"
data += "And that put an end to Lanigan's Ball.\n"

# data = "Laurence went to dublin think and wine for lanigans ball entangled in nonsence me "
# data += "Laurence went to dublin his pipes bellows chanters and all all entangled all kinds "
# data += "Laurence went to dublin how the room a whirligig ructions long at brooks fainted"

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
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len - 1)) #-1 是因為最後一個是標籤y
# model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20))) # 雙向LSTM層
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(xs, ys, epochs=500, verbose=1)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
  
plot_graphs(history, "acc")
# plot_graphs(history, "loss")

seed_text = "Laurence went to dublin"
next_words = 100

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
