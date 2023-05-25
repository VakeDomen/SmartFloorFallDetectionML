# Transformer.py
# ----------
# This script trains a Transformer classifier on preprocessed data and saves the
# resulting model for each fold. The model's parameters are read from a
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.backend import manual_variable_initialization 
from tqdm import tqdm
import gc
import configparser

# Read model configuration from the config.ini file
config = configparser.ConfigParser()
config.read('../../config.ini')

FOLDS                   = int(config.get('data-preprocess', 'folds'))
HIDDEN_SIZE             = eval(config.get('model-transformer', 'hidden_size'))
NUM_LAYERS              = eval(config.get('model-transformer', 'num_layers'))
NUM_HEADS               = eval(config.get('model-transformer', 'num_heads'))
ENCODER_ACTIVATION      = config.get('model-transformer', 'encoder_layer_activation')
DROPOUT_RATE            = eval(config.get('model-transformer', 'dropout_rate'))
BATCH_SIZE              = eval(config.get('model-transformer', 'batch_size'))
LEARNING_RATE           = eval(config.get('model-transformer', 'learning_rate'))
EPOCHS                  = eval(config.get('model-transformer', 'epochs'))
METRICS                 = eval(config.get('model-transformer', 'metrics'))
WINDOW_SIZE             = eval(config.get('data-preprocess', 'window_size'))
SENSORS_KERNEL_SHAPE    = eval(config.get('data', 'shape_kernel'))
SEED                    = int(config.get('general', 'random_seed'))

# Set random seeds for reproducibility
manual_variable_initialization(True)
tf.random.set_seed(SEED)

# Function to build a CatBoost classifier model with the given configuration
def build_model():
    inputs = keras.Input(shape=(WINDOW_SIZE, SENSORS_KERNEL_SHAPE[0] * SENSORS_KERNEL_SHAPE[1]))

    # Add the Transformer Encoder layers
    x = layers.Dropout(DROPOUT_RATE)(inputs)
    for i in range(NUM_LAYERS):
        x = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=HIDDEN_SIZE*2)(x, x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.Dense(HIDDEN_SIZE*4, activation=ENCODER_ACTIVATION)(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.Dense(HIDDEN_SIZE*2)(x)
        x = layers.LayerNormalization()(x)

    # Take the mean of the output features along the time axis
    x = layers.GlobalAveragePooling1D()(x)
    # Add the output layer
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load the preprocessed data for each fold
X = []
Y = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../../data/folds/X{i}.npy"))
    Y.append(np.load(f"../../data/folds/Y{i}.npy"))


# Train a model for each fold and save the resulting model
for i in range(FOLDS):
    print(f"Building model {i+1}...")
    model = build_model()
    print(f"Preparing learning folds...")
    X_train = np.concatenate((X[:i] + X[i+1:]))
    Y_train = np.concatenate((Y[:i] + Y[i+1:]))
        
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(10000, seed=SEED)

    # Define the loss function and optimizer
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(LEARNING_RATE)
        
    print("Compiling model...")
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=METRICS)

    print(f"Fitting model {i+1}/{FOLDS}")
    # Train the model
    model.fit(train_dataset, epochs=EPOCHS, verbose=1)
    model.save(f"../../models/Transformer/f{i}_Transformer.h5")
    gc.collect()

print("Done!")
