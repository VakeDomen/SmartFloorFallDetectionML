# HoeffdingTreeClassifier.py
# ----------
# This script trains a HoeffdingTree classifier on preprocessed data and saves the
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


import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from tqdm import tqdm
import pickle
import gc
from tqdm import tqdm
import configparser

# Read model configuration from the config.ini file
config = configparser.ConfigParser()
config.read('../../config.ini')

FOLDS                       = int(config.get('data-preprocess', 'folds'))
MAX_BYTE_SIZE               = eval(config.get('model-hoeffding-tree', 'max_byte_size'))
MEMORY_ESTIMATE_PERIOD      = eval(config.get('model-hoeffding-tree', 'memory_estimate_period'))
GRACE_PERIOD                = eval(config.get('model-hoeffding-tree', 'grace_period'))
SPLIT_CRITERION             = config.get('model-hoeffding-tree', 'split_criterion')
SPLIT_CONFIDENCE            = eval(config.get('model-hoeffding-tree', 'split_confidence'))
TIE_THRESHOLD               = eval(config.get('model-hoeffding-tree', 'tie_threshold'))
STOP_MEM_MANAGEMENT         = bool(config.get('model-hoeffding-tree', 'stop_mem_management'))
REMOVE_POOR_ATTS            = bool(config.get('model-hoeffding-tree', 'remove_poor_atts'))
NO_PREPRUNE                 = bool(config.get('model-hoeffding-tree', 'no_preprune'))
LEAF_PREDICTION             = config.get('model-hoeffding-tree', 'leaf_prediction')
NB_THRESHOLD                = eval(config.get('model-hoeffding-tree', 'nb_threshold'))
BATCH_SIZE                  = int(config.get('model-hoeffding-tree', 'batch_size'))

# Function to build a CatBoost classifier model with the given configuration
def build_model():
    return HoeffdingTreeClassifier(
            leaf_prediction=LEAF_PREDICTION, 
            split_criterion=SPLIT_CRITERION,
            max_byte_size=MAX_BYTE_SIZE,
            memory_estimate_period=MEMORY_ESTIMATE_PERIOD,
            grace_period=GRACE_PERIOD,
            split_confidence=SPLIT_CONFIDENCE,
            tie_threshold=TIE_THRESHOLD,
            stop_mem_management=STOP_MEM_MANAGEMENT,
            no_preprune=NO_PREPRUNE,
            nb_threshold=NB_THRESHOLD            
    )


# Load the preprocessed data for each fold
X = []
Y = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../../data/folds/X{i}.npy"))
    Y.append(np.load(f"../../data/folds/Y{i}.npy"))


# Reshape the data by combining the last two dimensions   
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = X[i].reshape(*X[i].shape[:-2], -1)


# Train a model for each fold and save the resulting model
print("Fitting models...")
for i in range(FOLDS):
    print(f"Building model {i+1}...")
    model = build_model()
    print(f"Preparing learning folds...")
    X_train = np.concatenate((X[:i] + X[i+1:]))
    Y_train = np.concatenate((Y[:i] + Y[i+1:]))
    y_pred = np.empty(len(X_train))
    
    print(f"Fitting model {i+1}/{FOLDS}")
    num_batches = len(X_train) // BATCH_SIZE
    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = (batch_idx + 1) * BATCH_SIZE
        X_batch = X_train[batch_start:batch_end]
        Y_batch = Y_train[batch_start:batch_end]

        for index in range(len(X_batch)):
            y_pred[batch_start + index] = model.predict([X_batch[index]])[0]
            model.partial_fit([X_batch[index]], [Y_batch[index]])

    # Process remaining samples if the number of samples is not divisible by the batch size
    if len(X_train) % BATCH_SIZE != 0:
        remaining_start = num_batches * BATCH_SIZE
        X_remaining = X_train[remaining_start:]
        Y_remaining = Y_train[remaining_start:]

        for index in range(len(X_remaining)):
            y_pred[remaining_start + index] = model.predict([X_remaining[index]])[0]
            model.partial_fit([X_remaining[index]], [Y_remaining[index]])

            
            
    print("Saving model...")
    with open(f'../../models/HoeffdingTree/f{i}_HoeffdingTree.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()

print("Done!")
