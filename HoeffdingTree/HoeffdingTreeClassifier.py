import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from tqdm import tqdm
import pickle

print("Loading data...")
X_train = np.load("../data/X_train.npy")
Y_train = np.load("../data/Y_train.npy")
X_test = np.load("../data/X_test.npy")
Y_test = np.load("../data/Y_test.npy")

print("Reshaping data...")
X_train = X_train.reshape(*X_train.shape[:-2], -1)
X_test = X_test.reshape(*X_test.shape[:-2], -1)

print("Building model...")
model = HoeffdingTreeClassifier(leaf_prediction='mc', split_criterion='gini')
y_pred = np.empty(len(X_train))


print("Fitting model...")
for index in tqdm(range(len(X_train))):
    y_pred[index] = model.predict([X_train[index]])[0]
    model.partial_fit([X_train[index]], [Y_train[index]])

print("Saving model...")
with open('../models/HoeffdingTreeModel.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")
