import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import gc
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tqdm import tqdm

USE_FOLDS = True
FOLDS = 5


def build_model():
    model=Sequential()
    #adding convolution layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(300,16, 1)))
    #adding pooling layer
    model.add(MaxPool2D(2,2))
    #adding fully connected layer
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    #adding output layer
    model.add(Dense(1,activation='sigmoid'))
    #compiling the model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

X = []
Y = []
print("Loading data...")
if USE_FOLDS:
    for i in tqdm(range(FOLDS)):
        X.append(np.load(f"../../data/folds/X{i}.npy"))
        Y.append(np.load(f"../../data/folds/Y{i}.npy"))
else:
    X.append(np.load("../../data/X_train.npy"))
    Y.append(np.load("../../data/Y_train.npy"))
       
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = np.expand_dims(X[i], axis=-1)


if USE_FOLDS:
    print("Fitting models...")
    for i in range(FOLDS):
        print(f"Building model {i}...")
        model = build_model()
        print(f"Preparing learning folds...")
        X_train = np.concatenate((X[:i] + X[i+1:]))
        Y_train = np.concatenate((Y[:i] + Y[i+1:]))
        print(X_train.shape)
        print(Y_train.shape)
        print(X[i].shape)
        print(Y[i].shape)
        print(f"Fitting model {i}/{FOLDS}")
        model.fit(X_train, Y_train, epochs=10, validation_data=(X[i], Y[i]))
        model.save(f"../../models/CNN/f{i}_CNN.h5")

else:
    print("Building model...")
    model = build_model()
    print("Fitting model...")
    history = model.fit(X[0], Y[0], epochs=10)
    model.save("../../models/CNN.h5")




