import math
import numpy as np
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as cb

print("Loading data...")
X_train = np.load("../data/X_train.npy")
Y_train = np.load("../data/Y_train.npy")
X_test = np.load("../data/X_test.npy")
Y_test = np.load("../data/Y_test.npy")

print("Reshaping data...")
X_train = X_train.reshape(*X_train.shape[:-2], -1)
X_test = X_test.reshape(*X_test.shape[:-2], -1)



print("Building model...")
        
# Define the CatBoost model
model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    random_seed=42,
    task_type='GPU',
    devices='0:1'
)

print("Fitting model...")
# Fit the model on the training data
model.fit(X_train, Y_train, verbose=True)

print("Saving model...")
model.save_model(
    "../models/CatBoostModel.cmb",
    format="cbm",
    export_parameters=None,
    pool=None
)

print("Done!")

"""
# Evaluate the model on the testing data
Y_pred = model.predict(X_test)

accuracy = np.mean(Y_test == Y_pred)
n = len(Y_test)
print(f'Testing Accuracy: {accuracy:.4f} on {n} data points')

Y_pred = model.predict(X_test_p)
accuracy = np.mean(([1]*len(Y_pred)) == Y_pred)
n = len(Y_test)
print(f'Positive test Accuracy: {accuracy:.4f} on {n} data points')
"""
