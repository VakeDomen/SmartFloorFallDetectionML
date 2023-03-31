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


def build_model():
    return HoeffdingTreeClassifier(
            leaf_prediction=LEAF_PREDICTION, 
            split_criterion=SPLIT_CRITERION,
            max_byte_size=MAX_BYTE_SIZE,
            memory_estimate_period=MEMORY_ESTIMATE_PERIOD,
            grace_period=MEMORY_ESTIMATE_PERIOD,
            split_confidence=SPLIT_CONFIDENCE,
            tie_threshold=TIE_THRESHOLD,
            stop_mem_management=STOP_MEM_MANAGEMENT,
            remove_poor_atts=REMOVE_POOR_ATTS,
            no_preprune=NO_PREPRUNE,
            nb_threshold=NB_THRESHOLD            
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
