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



print("Loading data...")
X_train = np.load("../data/X_train.npy")
Y_train = np.load("../data/Y_train.npy")
X_test = np.load("../data/X_test.npy")
Y_test = np.load("../data/Y_test.npy")
       

# Define the hyperparameters
hidden_size = 128
num_layers = 2
num_heads = 4
dropout_rate = 0.2
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

print("Build dataset...")
# Create a TensorFlow Dataset for the training data
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.batch(batch_size).shuffle(10000)

print("Build model...")
# Initialize the Transformer model
model = transformer_model(hidden_size, num_layers, num_heads, dropout_rate)

# Define the loss function and optimizer
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate)

print("Compiling model...")
# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["binary_accuracy"])


print("Fitting model...")
# Train the model
model.fit(train_dataset, epochs=num_epochs, verbose=1)

print("Saving model...")
model.save('../models/transformer.h5')

print("Done!")
