# CNN.py
# -------
# This script trains a Convolutional Neural Network (CNN) on preprocessed data and
# saves the resulting model for each fold. The model's parameters are read from a
# config.ini file.
#
# Author: Domen Vake
#
# MIT License
# Copyright (c) 2023 Domen Vake
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed
from tqdm import tqdm
import configparser


# Read model configuration from the config.ini file
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
SEED                        = int(config.get('general', 'random_seed'))

# Validate the configuration parameters
if not len(SHAPE_MAX_POOL) == 2:
    print("Incorrect SHAPE_MAX_POOL")
    exit(1)
if not len(DENSE_LAYERS_UNITS) == len(DENSE_LAYERS_ACTIVATIONS):
    print("Lenth of DENSE_LAYERS_UNITS and DENSE_LAYERS_ACTIVATIONS should match")
    exit(1)

# Set random seeds for reproducibility
set_random_seed(SEED)

# Function to build a CNN model with the given configuration
def build_model():
    model=Sequential()
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
    model.add(MaxPool2D(SHAPE_MAX_POOL[0], SHAPE_MAX_POOL[1]))
    model.add(Flatten())
    for i in range(len(DENSE_LAYERS_UNITS)):
        model.add(Dense(DENSE_LAYERS_UNITS[i], activation=DENSE_LAYERS_ACTIVATIONS[i]))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    return model


# Load the preprocessed data for each fold
X = []
Y = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../../data/folds/X{i}.npy"))
    Y.append(np.load(f"../../data/folds/Y{i}.npy"))


# Reshape the data to fit the input layer     
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = np.expand_dims(X[i], axis=-1)


# Train a model for each fold and save the resulting model
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
print("Done!")
