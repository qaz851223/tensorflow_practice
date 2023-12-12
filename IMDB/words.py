import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I, love my cat',
    'You love my dog!', 
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") # 建立100個單字的字典
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index # 查看字典中的數據
sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=8)
print("\nWord Index= ", word_index)
print("\nSequences = ", sequences)
print("\nPadded Sequences:")
print(padded)

test_data = [
    'i really love my dog', 
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=5)
print("\nPadded Test Sequence: ")
print(padded)