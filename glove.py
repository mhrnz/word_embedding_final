
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


def cooccurence_mat(int_words,vocab_size,window_size):
    skip_window=window_size//2
    matrix=np.zeros((vocab_size,vocab_size),np.float32)
    # we go through the dataset and count cooccurence for every (center,context) pairs in window
    for center_ind in range(skip_window,len(int_words)-skip_window,1):
        for j in range(window_size):
            if j!=skip_window:
                matrix[int_words[center_ind],int_words[center_ind-skip_window+j]]+=1.0/abs(skip_window-j)
                matrix[int_words[center_ind-skip_window+j],int_words[center_ind]]+=1.0/abs(skip_window-j)
    return matrix

def getContexts(batch,i,window_size):
    n=window_size//2
    return list(set(batch[max(0,i-n):i]+batch[i+1:min(len(batch),i+n+1)]))

def get_batch_glove(int_words,vocab_size,batch_size,window_size):
    n_batches=len(int_words)//batch_size
    int_words=int_words[:n_batches*batch_size]
    cooccur_mat=cooccurence_mat(int_words,vocab_size,window_size)
    for batch_start in range(0,len(int_words),batch_size):
        batch=int_words[batch_start:batch_start+batch_size]
        x,y,freq=[],[],[]
        for i in range(len(batch)):
            center=batch[i]
            contexts=getContexts(batch,i,window_size)
            for i in range(len(contexts)):
                if cooccur_mat[contexts[i],center]>0:
                    freq.append(cooccur_mat[contexts[i],center])
                    x.append(center)
                    y.append(contexts[i])
        # x: indexes of centers, y: indexes of contexts, freq : number of times (center,context) cooccur
        yield x,y,freq
    
    
def train_glove(int_words,vocab_to_int):
    
    # hyperparameters
    epochs=100
    batch_size=100
    window_size=5
    dimension=300
    n_samples=20
    
    vocab_size=len(vocab_to_int)
    
    inputs=tf.placeholder(tf.int32,[None])
    labels=tf.placeholder(tf.int32,[None])
    freqs=tf.placeholder(tf.float32,[None])
    
    embedding_V=tf.Variable(tf.random_uniform([vocab_size,dimension],-1,1))
    embedding_U=tf.Variable(tf.random_uniform([vocab_size,dimension],-1,1))
    
    center_embeds=tf.nn.embedding_lookup(embedding_V,inputs)
    contexts_embeds=tf.nn.embedding_lookup(embedding_U,labels)
    
    # 2 hypermarameters
    alpha=100
    beta=3/4
    #f(freq_i)=min((freq_i/alpha)^beta,1)
    loss=tf.multiply(tf.square(tf.reduce_sum(tf.multiply(center_embeds,contexts_embeds),axis=1)-tf.log(freqs)),tf.minimum(1.0,tf.pow(tf.div(freqs,alpha),beta)))
    cost=tf.reduce_mean(loss)
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            batch_generator=get_batch_glove(int_words,vocab_size,batch_size,window_size)
            cost_value=0
            for x,y,freq in batch_generator:
                feed_dic={inputs:x , labels:y , freqs: freq}
                _,cost_val=sess.run([optimizer,cost],feed_dic)
                cost_value+=cost_val
            print('epoch_{}'.format(epoch),'cost_value: ',cost_value)
    return embedding_V,embedding_U

if __name__=='__main__':
    int_words,vocab_to_int=data_preprocessing.get_int_words()
    center_embeds_glove,context_embeds_glove=train_glove(int_words,vocab_to_int)
    embeddings_glove=center_embeds_glove+context_embeds_glove

