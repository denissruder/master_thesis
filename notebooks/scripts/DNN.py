#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:45:09 2020

@author: deniss
"""

from keras.layers import  Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from ast import literal_eval
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import scipy.sparse
from nltk import word_tokenize
import time

df = pd.read_csv('final_lemmatized.csv', dtype=object, low_memory=False)
df['description_text'] = df.description_text.apply(literal_eval)

df['description_text'] = df['description_text'].apply(lambda x: ' '.join(x))
df['description_text'] = df['description_text'].str.replace(',', '').str.replace('[', '').str.replace(']', '')

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

#data = pd.DataFrame({"description_text":X_train,"harmonized_number":y_train})
#data.to_csv("train.csv")
#data.to_csv("test.csv")

def TFIDF(X_train, X_test):
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(df['description_text'])
    vectorizer_x = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
    X_train = vectorizer_x.fit_transform(X_train)
    X_test = vectorizer_x.transform(X_test)
    #print("tf-idf with", str(len(vectorizer_x.get_feature_names())),"features")
    #print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Text(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 1024 # number of nodes
    nLayers = 1 # number of  hidden layer

    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy']
                )

    return model

model_DNN = Build_Model_DNN_Text(17464,3243)

model_DNN.summary()
model_DNN.save("dnn.h5")

X_train_tfidf_sparse,X_test_tfidf_sparse = TFIDF(X_train,X_test)

def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0


batch_size = 10000
steps_per_epoch= X_train_tfidf_sparse.shape[0]/batch_size
validation_steps=X_train_tfidf_sparse.shape[0]/batch_size

train_batch_generator = nn_batch_generator(X_train_tfidf_sparse, to_categorical(y_train), batch_size)
test_batch_generator = nn_batch_generator(X_test_tfidf_sparse, to_categorical(y_test), batch_size)

for x in range(1):
    model_DNN = load_model("dnn.h5")

    model_DNN.fit(train_batch_generator,
                  validation_data=test_batch_generator,
                  epochs=1,
                  steps_per_epoch=steps_per_epoch,
                  validation_steps=validation_steps,
                  verbose=1)
    
    model_DNN.save("dnn.h5")
    del model_DNN
    gc.collect()

predicted1 = model_DNN.predict_classes(X_test_tfidf_sparse.todense()[:116237])
gc.collect()
time.sleep(10)
predicted2 = model_DNN.predict_classes(X_test_tfidf_sparse.todense()[116237:232474])
gc.collect()
time.sleep(10)
predicted3 = model_DNN.predict_classes(X_test_tfidf_sparse.todense()[232474:])
gc.collect()

predicted = np.concatenate([predicted1, predicted2,predicted3])
print(metrics.classification_report(y_test, predicted))