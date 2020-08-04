#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:58:53 2020

@author: deniss
"""

from RMDL import RMDL_Text
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

RMDL_Text.Text_Classification(X_train, y_train, X_test,  y_test, batch_size=128,
                 EMBEDDING_DIM=100,MAX_SEQUENCE_LENGTH = 22, MAX_NB_WORDS = 17476,
                 GloVe_dir="GloVe-master/", GloVe_file = "vectors.txt",
                 sparse_categorical=False, random_deep=[50, 0, 10], epochs=[20, 0, 20],  plot=True,
                 min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                 min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32,  max_nodes_rnn=128,
                 min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                 random_state=42, random_optimizor=True, dropout=0.05,no_of_classes=4119)