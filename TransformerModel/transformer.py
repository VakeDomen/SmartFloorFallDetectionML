import numpy as np
import gc
import pandas as pd
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 300

#!gdown --id 1c69vPJQChbqEWIvQI3sJ6osO-tLVXXPj
#!gdown --id 1HEeF49zrKyxobAJl1JX7UonVs_F2v69S

#load both as pandas df
positive_set = pd.read_csv('positiveSet.csv')
negative_set = pd.read_csv('negativeSet.csv')

# Drop columns that are not needed
# only keep sample ID and sensor info
positive_set = positive_set.drop(columns=["person_ID", "fall_category", "tick"])
negative_set = negative_set.drop(columns=["neg_category", "person_ID", "tick"])

def extract_windows(data, col):
  for val in data[col].unique():
    sample_data = data.loc[data[col] == val]
    sample_data = sample_data.drop(columns=[col]).to_numpy()
    for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
      window = sample_data[ window_offset : window_offset + WINDOW_SIZE, : ]
      yield window / 65537


pdata = list(extract_windows(positive_set, "fall_ID"))
ndata = list(extract_windows(negative_set, "neg_ID"))
        
indices = np.random.choice(len(pdata), size=(len(pdata) - len(ndata)), replace=False)
pdata = np.delete(pdata, indices, axis=0)


X_train, X_test, Y_train, Y_test = train_test_split(
  np.concatenate((
    pdata,
    ndata
  )),
  np.concatenate((
    np.ones(len(pdata)),
    np.zeros(len(ndata))
  )),
  test_size=0.2,
  random_state=4,
)
del pdata, ndata
        

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

# Define the hyperparameters
hidden_size = 128
num_layers = 2
num_heads = 4
dropout_rate = 0.2
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

# Create a TensorFlow Dataset for the training data
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.batch(batch_size).shuffle(10000)

# Initialize the Transformer model
model = transformer_model(hidden_size, num_layers, num_heads, dropout_rate)

# Define the loss function and optimizer
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# Train the model
model.fit(train_dataset, epochs=num_epochs, verbose=1)

# Evaluate the model on the test data
test_loss, test_mse = model.evaluate(X_test, Y_test, verbose=1)

# Print the test loss and mean squared error
print("Test loss:", test_loss)
print("Test MSE:", test_mse)
