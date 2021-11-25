# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:01:05 2021

@author: Admin
"""
import numpy as np
import pandas as pd
#from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import keras
import itertools

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score

'''Step 1: Read'''

#r_cols = ['age', 'sex', 'chest_pain_type', 'resting_BP', 'cholesterol', 'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 'old_peak', 'st_slop', 'heart_dease']
heart = pd.read_csv('heart.csv', sep=',', encoding='latin-1')
heart.head()
'''Step 2: Flatten Variable'''
X = pd.get_dummies(heart)

'''Step 3: X, Y split'''
Y = np.array(X['HeartDisease'])
X = X.drop(['HeartDisease'], axis = 1)

'''Step 4: Min Max normalization'''
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

'''Step 5: Train test split'''
X_train ,X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)

'''Step 6: Transpose'''
'''
X_train = np.ndarray.transpose(X_train)
X_test = np.ndarray.transpose(X_test)
Y_train = np.ndarray.transpose(Y_train)
Y_test = np.ndarray.transpose(Y_test)
'''
'''Step 6: Create a checkpoint to save best value'''
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='weight.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_accuracy', mode = 'max', save_best_only=True)

'''Step 7: Call wanted model'''

model = Sequential([
    Dense(700, activation = 'relu'),
    Dense(700, activation = 'relu'),
    Dense(200, activation = 'relu'),
    Dense(100, activation = 'relu'),
    Dense(1, activation = 'sigmoid'),])

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['accuracy'])

'''Step 8: Store history for reverse in case of old value is better''' 
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs = 10, callbacks=[model_checkpoint_callback], verbose=2)

'''Step 9: Predict'''
Y_preds = model.predict(X_test)

'''Step 10: Check acuracy'''
acs = accuracy_score(Y_test, tf.round(Y_preds))

print("Accuracy: ", acs)