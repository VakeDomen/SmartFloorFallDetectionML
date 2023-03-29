import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from tqdm import tqdm
import pickle
import gc
from tqdm import tqdm

USE_FOLDS = True
FOLDS = 5



def build_model():
    return HoeffdingTreeClassifier(leaf_prediction='mc', split_criterion='gini')


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


if USE_FOLDS:
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

else:
    print("Building model...")
    model = build_model()
    y_pred = np.empty(len(X[0]))


    print("Fitting model...")
    for index in tqdm(range(len(X[0]))):
        y_pred[index] = model.predict([X[0][index]])[0]
        model.partial_fit([X[0][index]], [Y[0][index]])

    print("Saving model...")
    with open('../../models/HoeffdingTree.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")
