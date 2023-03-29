import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
from tqdm import tqdm
import gc


USE_FOLDS = True
FOLDS = 5


# Define the Transformer model architecture
def transformer_model(hidden_size, num_layers, num_heads, dropout_rate):
    inputs = keras.Input(shape=(300, 16))

    # Add the Transformer Encoder layers
    x = layers.Dropout(dropout_rate)(inputs)
    for i in range(num_layers):
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size*2)(x, x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(hidden_size*4, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(hidden_size*2)(x)
        x = layers.LayerNormalization()(x)

    # Take the mean of the output features along the time axis
    x = layers.GlobalAveragePooling1D()(x)

    # Add the output layer
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

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
       

# Define the hyperparameters
hidden_size = 128
num_layers = 2
num_heads = 4
dropout_rate = 0.2
batch_size = 32
learning_rate = 1e-4
num_epochs = 10



if USE_FOLDS:
    for i in range(FOLDS):
        print(f"Building model {i+1}...")
        model = transformer_model(hidden_size, num_layers, num_heads, dropout_rate)
        print(f"Preparing learning folds...")
        X_train = np.concatenate((X[:i] + X[i+1:]))
        Y_train = np.concatenate((Y[:i] + Y[i+1:]))
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.batch(batch_size).shuffle(10000)

        # Define the loss function and optimizer
        loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate)
        
        print("Compiling model...")
        # Compile the model
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["binary_accuracy"])


        print(f"Fitting model {i+1}/{FOLDS}")
        # Train the model
        model.fit(train_dataset, epochs=num_epochs, verbose=1)

        model.save(f"../../models/Transformer/f{i}_Transformer.h5")
        gc.collect()
else:
    print("Building model...")
    model = transformer_model(hidden_size, num_layers, num_heads, dropout_rate)
    print("Fitting model...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X[0], Y[0]))
    train_dataset = train_dataset.batch(batch_size).shuffle(10000)
    # Define the loss function and optimizer
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate)
    print("Compiling model...")
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["binary_accuracy"])
    history = model.fit(train_dataset, epochs=num_epochs, verbose=1)
    model.save("../../models/Transformer.h5")

print("Done!")
