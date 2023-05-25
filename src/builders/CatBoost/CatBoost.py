# CatBoost.py
# ----------
# This script trains a CatBoost classifier on preprocessed data and saves the
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
from tqdm import tqdm
import gc
import catboost as cb
import configparser

# Read model configuration from the config.ini file
config = configparser.ConfigParser()
config.read('../../config.ini')
FOLDS           = int(config.get('data-preprocess', 'folds'))
ITERATIONS      = int(config.get('model-cat-boost', 'iterations'))
LEARNING_RATE   = float(config.get('model-cat-boost', 'learning_rate'))
DEPTH           = int(config.get('model-cat-boost', 'depth'))
L2_LEAF_REG     = int(config.get('model-cat-boost', 'l2_leaf_reg'))
LOSS_FUNCTION   = config.get('model-cat-boost', 'loss_function')
TASK_TYPE       = config.get('model-cat-boost', 'task_type')
DEVICES         = config.get('model-cat-boost', 'devices')
SEED            = int(config.get('general', 'random_seed'))

# Function to build a CatBoost classifier model with the given configuration
def build_model():
    return cb.CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        l2_leaf_reg=L2_LEAF_REG,
        loss_function=LOSS_FUNCTION,
        random_seed=SEED,
        task_type=TASK_TYPE,
        devices=DEVICES
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
    print(f"Building model {i + 1}...")
    model = build_model()
    print(f"Preparing learning folds...")
    X_train = np.concatenate((X[:i] + X[i+1:]))
    Y_train = np.concatenate((Y[:i] + Y[i+1:]))
    print(f"Fitting model {i + 1}/{FOLDS}")
    model.fit(X_train, Y_train, verbose=True)
    print("Saving model...")
    model.save_model(f"../../models/CatBoost/f{i}_CatBoost.cbm", format="cbm")
    gc.collect()
print("Done!")
