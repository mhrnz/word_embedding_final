
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import tensorflow.keras.preprocessing as preprocessing
from collections import Counter
import random
import import_ipynb
import data_preprocessing


# In[2]:


def get_contexts(batch,i,window_size):
    n=window_size//2
    return list(set(batch[max(0,i-n):i]+batch[i+1:min(len(batch),i+n+1)]))

#generator for batches
def get_batch_sg(words,batch_size,window_size):
    n_batches=len(words)//batch_size
    words=words[:n_batches*batch_size]
    for batch_start in range(0,len(words),batch_size):
        batch=words[batch_start:batch_start+batch_size]
        x,y=[],[]
        for i in range(len(batch)):
            center=batch[i]
            contexts=get_contexts(batch,i,window_size)
            y.extend(contexts)
            x.extend([center]*len(contexts))
        yield x,y 
def train_sg(int_words,vocab_to_int):
    vocab_size=len(vocab_to_int)
    # hyperparameters
    epochs=300
    batch_size=100
    window_size=5
    word_dimension=300
    n_samples=10
    
    inputs=tf.placeholder(tf.int32,[None],name='inputs') # size is variable , inputs are indexes of words in the batch
    labels=tf.placeholder(tf.int32,[None,None],name='labels')
    with tf.variable_scope("skip_gram"):
        embedding_V=tf.Variable(tf.random_uniform((vocab_size,word_dimension),-1,1))
        embed=tf.nn.embedding_lookup(embedding_V,inputs) # chooses the given rows
        embedding_U=tf.Variable(tf.random_normal((vocab_size,word_dimension)))
        softmax_biases=tf.Variable(tf.zeros(vocab_size))
        #loss with negative_sampling
        loss=tf.nn.sampled_softmax_loss(weights=embedding_U,biases=softmax_biases,inputs=embed,labels=labels,num_sampled=n_samples,num_classes=vocab_size)
        cost=tf.reduce_mean(loss)
        optimizer=tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            cost_value=0
            batch_generator=get_batch_sg(int_words,batch_size,window_size)
            for x,y in batch_generator:
                feed_dict={inputs:x,labels:np.array(y)[:,None]}  #labels:np.array(y)[:,None] adds a dimmension, is like squeeze(1)
                _,cost_val=sess.run([optimizer,cost],feed_dict)
                cost_value+=cost_val
            print('epoch_{}'.format(epoch),'cost_value: ',cost_value)
    return embedding_V,embedding_U

if __name__=='__main__':
    int_words,vocab_to_int=data_preprocessing.get_int_words()
    center_embeds_sg,context_embeds_sg=train_sg(int_words,vocab_to_int)
    word_embeddings=center_embeds_sg+context_embeds_sg
    

