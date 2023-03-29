import math
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as cb

USE_FOLDS = True
FOLDS = 5

def build_model():
    return cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        random_seed=42,
        task_type='GPU',
        devices='0:1'
    )


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
       
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = X[i].reshape(*X[i].shape[:-2], -1)


print("Building model...")

if USE_FOLDS:
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
else:
    print("Building model...")
    model = build_model()
    print("Fitting model...")
    history = model.fit(X[0], Y[0], verbose=True)
    print("Saving model...")
    model.save("../../models/CatBoost.cbm", format="cbm")

