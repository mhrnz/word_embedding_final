
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


def get_batch_cbow(int_words,batch_size,window_size):
    n_batches=len(int_words)//batch_size
    int_words=int_words[:n_batches*batch_size]
    center_ind=window_size//2
    
    for bath_start in range(0,len(int_words),batch_size):
        batch=int_words[bath_start:bath_start+batch_size]
        surroundings=np.ndarray((batch_size-(2*center_ind),window_size-1),np.int32)
        labels=np.ndarray((batch_size-(2*center_ind),1),np.int32)
        for i in range(center_ind,batch_size-center_ind,1):    
            center=batch[i]
            col_idx=0
            for j in range(window_size):
                if j==window_size//2:
                    continue
                else:
                    surroundings[i-center_ind,col_idx]=batch[i-center_ind+j]
                    col_idx+=1
            labels[i-center_ind,0]=center
            
        yield surroundings,labels
        

def train_cbow(int_words,vocab_to_int):
    
    # hyperparameters
    epochs=200
    batch_size=100
    window_size=5
    dimension=300
    n_samples=20
    
    half_window=window_size//2
    vocab_size=len(vocab_to_int)
    
    inputs=tf.placeholder(tf.int32,[batch_size-(2*half_window),window_size-1])
    labels=tf.placeholder(tf.int32,[batch_size-(2*half_window),1])
    with tf.variable_scope('cbow'):
    
        embeddings=tf.Variable(tf.random_uniform((vocab_size,dimension),-1,1))

        #get_avg_embed
        embeds=None
        for i in range(window_size-1):
            embedding_i=tf.nn.embedding_lookup(embeddings,inputs[:,i])
            emb_x,emb_y = embedding_i.get_shape().as_list()
            if embeds is None:
                embeds=tf.reshape(embedding_i,[emb_x,emb_y,1])
            else:
                embeds=tf.concat([embeds,tf.reshape(embedding_i,[emb_x,emb_y,1])],2)
        avg_embed=tf.reduce_mean(embeds,2,keepdims=False)

        softmax_w=tf.Variable(tf.random_normal((vocab_size,dimension)))
        softmax_b=tf.Variable(tf.zeros(vocab_size))

        loss=tf.nn.sampled_softmax_loss(weights=softmax_w,biases=softmax_b,inputs=avg_embed,labels=labels,num_sampled=n_samples,num_classes=vocab_size)
        cost=tf.reduce_mean(loss)
        optimizer=tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            batch_generator=get_batch_cbow(int_words,batch_size,window_size)
            cost_value=0
            
            for x,y in batch_generator:
                feed_dic={inputs:x, labels:y}
                _,cost_val=sess.run([optimizer,cost],feed_dict=feed_dic)
                cost_value+=cost_val
            print('epoch_{}'.format(epoch),'cost_value: ',cost_value)
            
    return embeddings,softmax_w

if __name__=='__main__':
    int_words,vocab_to_int=data_preprocessing.get_int_words()
    center_embeds_cbow,context_embeds_cbow=train_cbow(int_words,vocab_to_int)
    word_embeddings_cbow=center_embeds_cbow+context_embeds_cbow

