import math
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as cb
import configparser

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


X = []
Y = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../../data/folds/X{i}.npy"))
    Y.append(np.load(f"../../data/folds/Y{i}.npy"))
       
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = X[i].reshape(*X[i].shape[:-2], -1)


print("Building model...")

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
