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
import configparser

config = configparser.ConfigParser()
config.read('../../config.ini')

SENSOR_KERNEL_SHAPE         = eval(config.get('data', 'shape_kernel'))
WINDOW_SIZE                 = int(config.get('data-preprocess', 'window_size'))
FOLDS                       = int(config.get('data-preprocess', 'folds'))
N_CONV_FILTERS              = int(config.get('model-cnn', 'n_conv_filters'))
CONV_ACTIVATION             = config.get('model-cnn', 'conv_activation')
SHAPE_KERNEL                = eval(config.get('model-cnn', 'shape_kernel'))
SHAPE_MAX_POOL              = eval(config.get('model-cnn', 'shape_max_pool'))
DENSE_LAYERS_UNITS          = eval(config.get('model-cnn', 'dense_layers_units'))
DENSE_LAYERS_ACTIVATIONS    = eval(config.get('model-cnn', 'dense_layers_activations'))
LOSS_FUNCTION               = config.get('model-cnn', 'loss_function')
OPTIMIZER                   = config.get('model-cnn', 'optimizer')
METRICS                     = eval(config.get('model-cnn', 'metrics'))
EPOCHS                      = int(config.get('model-cnn', 'epochs'))

if not len(SHAPE_MAX_POOL) == 2:
    print("Incorrect SHAPE_MAX_POOL")
    exit(1)
if not len(DENSE_LAYERS_UNITS) == len(DENSE_LAYERS_ACTIVATIONS):
    print("Lenth of DENSE_LAYERS_UNITS and DENSE_LAYERS_ACTIVATIONS should match")
    exit(1)

def build_model():
    model=Sequential()
    #adding convolution layer
    model.add(Conv2D(
        N_CONV_FILTERS, 
        SHAPE_KERNEL, 
        activation=CONV_ACTIVATION, 
        input_shape=(
            WINDOW_SIZE, 
            SENSOR_KERNEL_SHAPE[0] * SENSOR_KERNEL_SHAPE[1], 
            1
        )
    ))
    #adding pooling layer
    model.add(MaxPool2D(SHAPE_MAX_POOL[0], SHAPE_MAX_POOL[1]))
    #adding fully connected layer
    model.add(Flatten())
    for i in range(len(DENSE_LAYERS_UNITS)):
        model.add(Dense(DENSE_LAYERS_UNITS[i], activation=DENSE_LAYERS_ACTIVATIONS[i]))
    #adding output layer
    model.add(Dense(1,activation='sigmoid'))
    #compiling the model
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    return model

X = []
Y = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../../data/folds/X{i}.npy"))
    Y.append(np.load(f"../../data/folds/Y{i}.npy"))
       
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = np.expand_dims(X[i], axis=-1)


print("Fitting models...")
for i in range(FOLDS):
    print(f"Building model {i+1}...")
    model = build_model()
    print(f"Preparing learning folds...")
    X_train = np.concatenate((X[:i] + X[i+1:]))
    Y_train = np.concatenate((Y[:i] + Y[i+1:]))
    print(f"Fitting model {i+1}/{FOLDS}")
    model.fit(X_train, Y_train, epochs=EPOCHS, validation_data=(X[i], Y[i]))
    model.save(f"../../models/CNN/f{i}_CNN.h5")
    gc.collect()

