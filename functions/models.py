import numpy as np
import pandas as pd
import os
import sys
from tensorflow.keras import models, Sequential, layers, optimizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN,MaxPooling1D, Conv1D, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from functions.path_functions import find_parent_filepath
from functions.feature_extraction import ravdess_generation, ravdess_preprocessing, ravdess_feature_extraction
from functions.feature_extraction import tess_export_process
import glob
import os.path
from pathlib import Path
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)



#Instantiate CNN 4 Layer Model
def cnn_4_model(X_train, loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    Model=Sequential()
    Model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
    Model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    Model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    Model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    Model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    Model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    Model.add(Dropout(0.2))

    Model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
    Model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    Model.add(Flatten())
    Model.add(Dense(units=32, activation='relu'))
    Model.add(Dropout(0.3))

    Model.add(Dense(units=14, activation='softmax'))

    Model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

<<<<<<< HEAD
    return model
=======
    return Model

>>>>>>> 5a7b74149372ba6e339a3cb4ebd721db3ac79eef


#Instantiate the CNN 7 Layer Model
def cnn_7_model(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    """
    Instantiate the CNN 7 Layer Model and Compile
    """

    #Create Model
    model = Sequential()
    model.add(Conv2D(256, 5,padding='same',input_shape=(25, 448, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, 5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(8)))
    model.add(Conv2D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(9))
    model.add(Activation('softmax'))


    #Compile Model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def lstm_4_model(loss = 'categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'], X=X):

    #Create Model Structure

    model = Sequential()
    model.add(layers.LSTM(64, return_sequences = True, input_shape=(X.shape[1:3])))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(8, activation = 'softmax'))

    #Compile Model
    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)

    return model



#Fit a Model
def fit_model(model, X_train, y_train, epochs = 300, batch_size = 256, validation_split = 0.3, patience = 20, verbose = 1):
    """
    Takes a previously created model and fits it to the training data.  Pass a previously
    created model with the training data to fit the model
    """

    #Create Early Stopping Metric
    es = EarlyStopping(patience=patience, restore_best_weights=True)

    model.fit(X_train,
              y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[es]
            )

    return model

if __name__=="__main__":
    smth = ravdess_x_y_train_test()
    print(smth)
