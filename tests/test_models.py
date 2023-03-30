import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from tensorflow import keras
from catboost import CatBoostClassifier
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
from tqdm import tqdm
import matplotlib.pyplot as plt

FOLDS = 5 

####################### Setting up data ########################
print("Loading test data...")
X = []
Y = []
X_flat = []
X_deep = []
print("Loading data...")
for i in tqdm(range(FOLDS)):
    X.append(np.load(f"../data/folds/X{i}.npy"))
    Y.append(np.load(f"../data/folds/Y{i}.npy"))
print("\tDone!")

print("Reshaping data...")
for i in tqdm(range(FOLDS)):
    X_flat.append(X[i].reshape(*X[i].shape[:-2], -1))
    X_deep.append(np.expand_dims(X[i], axis=-1))

print("\tDone!")

####################### Loading models ########################

print("Loading HoeffdingTree models...")
models_hoef = []
for i in tqdm(range(FOLDS)):
    with open(f'../models/HoeffdingTree/f{i}_HoeffdingTree.pickle', 'rb') as handle:
        models_hoef.append(pickle.load(handle))
print("\tDone!")

print("Loading CNN models...")
models_cnn = []
for i in tqdm(range(FOLDS)):
    models_cnn.append(keras.models.load_model(f'../models/CNN/f{i}_CNN.h5'))
print("\tDone!")

print("Loading transformer models...")
models_transf = []
for i in tqdm(range(FOLDS)):
    models_transf.append(keras.models.load_model(f"../models/Transformer/f{i}_Transformer.h5"))
print("\tDone!")

print("Loading CatBoost models...")
models_cb = []
for i in tqdm(range(FOLDS)):
    models_cb.append(CatBoostClassifier().load_model(f"../models/CatBoost/f{i}_CatBoost.cbm"))
print("\tDone!")


####################### Calculating ROC AUC ########################

print("\nPredicting probabilities for ROC AUC...")
y_pred_ht = []
for i in tqdm(range(FOLDS)):
    y_pred_ht.append(models_hoef[i].predict_proba(X_flat[i]))
print("\tROC AUC Done! \t| HoeffdingTreeModel")
y_pred_cnn = []
for i in tqdm(range(FOLDS)):
    y_pred_cnn.append(models_cnn[i].predict(X_deep[i]))
print("\tROC AUC Done! \t| CNN")
y_pred_transf = []
for i in tqdm(range(FOLDS)):
    y_pred_transf.append(models_transf[i].predict(X[i]))
print("\tROC AUC Done! \t| Transformer")
y_pred_cb = []
for i in tqdm(range(FOLDS)):
    y_pred_cb.append(models_cb[i].predict_proba(X_flat[i]))
print("\tROC AUC Done! \t| CatBoost")

print("\tDone!")

print("Calculating AUC ROC...")
auc_ht = []
for i in tqdm(range(FOLDS)):
    auc_ht.append(roc_auc_score(Y[i], y_pred_ht[i][:,1]))
print(f"HoeffdingTree: \t{auc_ht}")
auc_cnn = []
for i in tqdm(range(FOLDS)):
    auc_cnn.append(roc_auc_score(Y[i], y_pred_cnn[i]))
print(f"CNN: \t\t{auc_cnn}")
auc_transf = []
for i in tqdm(range(FOLDS)):
    auc_transf.append(roc_auc_score(Y[i], y_pred_transf[i]))
print(f"Transformer: \t{auc_transf}")
auc_cb = []
for i in tqdm(range(FOLDS)):
    auc_cb.append(roc_auc_score(Y[i], y_pred_cb[i][:,1]))
print(f"CatBoost: \t{auc_cb}")
