import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from tqdm import tqdm
import pickle
import gc
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read('../../config.ini')
FOLDS = int(config.get('data-prepocess', 'folds'))


def build_model():
    return HoeffdingTreeClassifier(leaf_prediction='mc', split_criterion='gini')


X = []
Y = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../../data/folds/X{i}.npy"))
    Y.append(np.load(f"../../data/folds/Y{i}.npy"))
       
print("Reshaping data...")
for i in tqdm(range(len(X))):
    X[i] = X[i].reshape(*X[i].shape[:-2], -1)


print("Fitting models...")
for i in range(FOLDS):
    print(f"Building model {i+1}...")
    model = build_model()
    print(f"Preparing learning folds...")
    X_train = np.concatenate((X[:i] + X[i+1:]))
    Y_train = np.concatenate((Y[:i] + Y[i+1:]))
    y_pred = np.empty(len(X_train))

    print(f"Fitting model {i+1}/{FOLDS}")
        
    for index in tqdm(range(len(X_train))):
        y_pred[index] = model.predict([X_train[index]])[0]
        model.partial_fit([X_train[index]], [Y_train[index]])

    print("Saving model...")
    with open(f'../../models/HoeffdingTree/f{i}_HoeffdingTree.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()

print("Done!")
