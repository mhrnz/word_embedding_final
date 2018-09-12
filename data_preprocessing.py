
# coding: utf-8

# In[10]:


import tensorflow as tf
import numpy as np
import tensorflow.keras.preprocessing as preprocessing
from collections import Counter
import random


# In[11]:


def text_to_word_sequence(text):
    words=preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n', lower=True, split=' ')
    return words


def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) #sorted_vocab : list of words
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

# p_drop(word)=1-sqrt(treshold/freq(word)) , frequent words are more likely to be removed from the dataset
def subsampling(int_words):
    word_counts=Counter(int_words) # a dictionary from int_word to number of times it appeared in the text
    total_count=len(int_words)
    p_drops={word:1-np.sqrt(1e-5/(count/total_count)) for word,count in word_counts.items()}
    train_words=[word for word in int_words if random.random()<(1-p_drops[word])] # the bigger p_drop the less likely to be chosen
    return train_words

def get_int_words():
    text=""
    #reading at most 100 lines
    with open('data.txt') as f:
        i=0
        for line in f.readlines():
            text+=line    
            i+=1
            if i>=100:
                break
    words=text_to_word_sequence(text)
    vocab_to_int, int_to_vocab = create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]
    return int_words,vocab_to_int
#     print(int_words)
#     print(len(int_words))
#     train_words=subsampling(int_words)

