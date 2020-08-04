#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:44:30 2020

@author: deniss
"""
import pandas as pd
import numpy as np
import gensim 
from gensim.models import Word2Vec
from ast import literal_eval
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense,Input,Embedding,Flatten, AveragePooling2D, Conv2D,Reshape, SpatialDropout1D
from keras.models import Sequential,Model
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import Concatenate
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.models import load_model
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation
from gensim.models.doc2vec import Doc2Vec

df = pd.read_csv('final_lemmatized.csv',low_memory=False)
df['description_text'] = df.description_text.apply(literal_eval)

g = df.groupby('harmonized_number')
df = g.filter(lambda x: len(x) > 1)

df = df.reset_index()
df = df.drop(['index'],axis=1)

uniques = list(df['harmonized_number'].unique())
df['index'] = df['harmonized_number']
df['index'] = df['index'].apply(lambda x: str(uniques.index(x)))

X = df['description_text']
y = pd.to_numeric(df["index"], downcast="float")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.31, random_state=777,stratify=y)

def loadData_Tokenizer(X_train, X_test,MAX_SEQUENCE_LENGTH=22):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train), ]
    X_test = text[len(X_train):, ]
    # Make a dictionary
    embeddings_index = {}
    
    
    wv = KeyedVectors.load("word2vec_lemmatized_skipgram.kv", mmap='r')
    words = wv.vocab.keys()
    embeddings_index = {word:wv[word] for word in words}
    """
    f = open("vectors.txt", encoding="utf8") 
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    f.close()
    """
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, X_test, word_index,embeddings_index)

# CNN MODEL
def Build_Model_CNN_Text(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=22, EMBEDDING_DIM=100, dropout=0.5):

    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                exit(1)
            
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    
    # applying a more complex convolutional approach
    convs = []
    filter_sizes = []
    layer = 5
    print("Filter  ",layer)
    for fl in range(0,layer):
        filter_sizes.append((fl+2,fl+2))
    
    node = 128
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    emb = Reshape((22,10,10), input_shape=(500,100))(embedded_sequences)
    
    for fsz in filter_sizes:
        l_conv = Conv2D(node, padding="same", kernel_size=fsz, activation='relu')(emb)
        l_pool = AveragePooling2D(pool_size=(5,1), padding="same")(l_conv)
        #l_pool = Dropout(0.25)(l_pool)
        convs.append(l_pool)
    
    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv2D(node, (5,5), padding="same", activation='relu')(l_merge)
    l_cov1 = AveragePooling2D(pool_size=(5,2), padding="same")(l_cov1)
    l_cov2 = Conv2D(node, (5,5), padding="same", activation='relu')(l_cov1)
    l_pool2 = AveragePooling2D(pool_size=(5,2), padding="same")(l_cov2)
    l_cov2 = Dropout(dropout)(l_pool2)
    l_flat = Flatten()(l_cov2)
    l_dense = Dense(128, activation='relu')(l_flat)
    l_dense = Dropout(dropout)(l_dense)
    
    preds = Dense(nclasses, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#word embeddings
X_train_Word2Vec,X_test_Word2Vec, word_index,embeddings_index = loadData_Tokenizer(X_train,X_test)

model_CNN = Build_Model_CNN_Text(word_index,embeddings_index, 3243)
model_CNN.summary()

model_CNN.fit(X_train_Word2Vec, y_train,
                              validation_data=(X_test_Word2Vec, y_test),
                              epochs=20,
                              batch_size=128,
                              verbose=2)

#model_CNN.save("cnn_cbow_non_lemmatized.h5")
#model_CNN.save("cnn_cbow.h5")
#model_CNN.save("cnn_pv_dm.h5")
#model_CNN.save("cnn_pv_dbow.h5")
#model_CNN.save("cnn_skipgram.h5")
#model_CNN.save("cnn_glove.h5")

#model_CNN = load_model('cnn_glove.h5')
#model_CNN = load_model('cnn_skipgram.h5')
#model_CNN = load_model('cnn_cbow.h5')
#model_CNN = load_model('cnn_pv_dm.h5')
#model_CNN = load_model('cnn_pv_dbow.h5')

predicted = model_CNN.predict(X_test_Word2Vec)

predicted = np.argmax(predicted, axis=1)

print(metrics.classification_report(y_test, predicted))