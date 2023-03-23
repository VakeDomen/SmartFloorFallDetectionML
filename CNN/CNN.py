import numpy as np
import gc
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


print("Loading data...")
X_train = np.load("../data/X_train.npy")
Y_train = np.load("../data/Y_train.npy")
X_test = np.load("../data/X_test.npy")
Y_test = np.load("../data/Y_test.npy")
       
print("Reshaping data...")
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Building model...")

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

print("Fitting model...")
history = model.fit(X_train,Y_train,epochs=10, validation_data=(X_test, Y_test))


model.save('../models/CNN')




