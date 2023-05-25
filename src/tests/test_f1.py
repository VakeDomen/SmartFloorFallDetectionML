
"""
test_f1.py
-----------

This script calculates the F1 scores for the trained models for each fold 
and for the total dataset.


MIT License
Copyright (c) 2023 Domen Vake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
from tensorflow import keras
from catboost import CatBoostClassifier
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
from sklearn.metrics import f1_score
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
FOLDS               = int(config.get('data-preprocess', 'folds')) 
TEST_CAT_BOOST      = int(config.get('models', 'cat_boost'))
TEST_CNN            = int(config.get('models', 'CNN'))
TEST_TRANSFORMER    = int(config.get('models', 'transformer'))
TEST_HOEFTREE       = int(config.get('models', 'hoeffding_tree'))



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
if TEST_HOEFTREE == 1:
    print("Loading HoeffdingTree models...")
    models_hoef = []
    for i in tqdm(range(FOLDS)):
        with open(f'../models/HoeffdingTree/f{i}_HoeffdingTree.pickle', 'rb') as handle:
            models_hoef.append(pickle.load(handle))
    print("\tDone!")

if TEST_CNN == 1:
    print("Loading CNN models...")
    models_cnn = []
    for i in tqdm(range(FOLDS)):
        models_cnn.append(keras.models.load_model(f'../models/CNN/f{i}_CNN.h5'))
    print("\tDone!")

if TEST_TRANSFORMER == 1:
    print("Loading transformer models...")
    models_transf = []
    for i in tqdm(range(FOLDS)):
        models_transf.append(keras.models.load_model(f"../models/Transformer/f{i}_Transformer.h5"))
    print("\tDone!")

if TEST_CAT_BOOST == 1:
    print("Loading CatBoost models...")
    models_cb = []
    for i in tqdm(range(FOLDS)):
        models_cb.append(CatBoostClassifier().load_model(f"../models/CatBoost/f{i}_CatBoost.cbm"))
    print("\tDone!")

######################## Making predictions ########################
print("\nPredicting probabilities...")

if TEST_HOEFTREE == 1:
    y_pred_hoef = []
    for i in tqdm(range(FOLDS)):
        y_pred_hoef.append(models_hoef[i].predict(X_flat[i]))
    print("\tROC AUC Done! \t| HoeffdingTreeModel")

if TEST_CNN == 1:
    y_pred_cnn = []
    for i in tqdm(range(FOLDS)):
        y_pred_cnn.append(models_cnn[i].predict(X_deep[i]))
    y_pred_cnn = [np.where(y_pred > 0.5, 1, 0).flatten() for y_pred in y_pred_cnn]
    print("\tROC AUC Done! \t| CNN")

if TEST_TRANSFORMER == 1:
    y_pred_transf = []
    for i in tqdm(range(FOLDS)):
        y_pred_transf.append(models_transf[i].predict(X[i]))
    y_pred_transf = [np.where(y_pred > 0.5, 1, 0).flatten() for y_pred in y_pred_transf]
    print("\tROC AUC Done! \t| Transformer")

if TEST_CAT_BOOST == 1:
    y_pred_cb = []
    for i in tqdm(range(FOLDS)):
        y_pred_cb.append(models_cb[i].predict(X_flat[i]))
    print("\tROC AUC Done! \t| CatBoost")

print("\tDone!")


####################### Calculating F1-score ########################
print("Calculating F1-Score...")
# Initialize F1-score lists for each model
f1_scores_hoef = []
f1_scores_cnn = []
f1_scores_transf = []
f1_scores_cb = []
print(y_pred_cnn)
print(y_pred_hoef)
print(np.array(y_pred_cb).shape)
# Loop through the folds
for i in tqdm(range(FOLDS)):
    if TEST_HOEFTREE == 1:
        f1_hoef = f1_score(Y[i], y_pred_hoef[i], average='weighted')
        f1_scores_hoef.append(f1_hoef)
    if TEST_CNN == 1:   
        f1_cnn = f1_score(Y[i], y_pred_cnn[i], average='weighted')
        f1_scores_cnn.append(f1_cnn)

    if TEST_TRANSFORMER == 1:
        f1_transf = f1_score(Y[i], y_pred_transf[i], average='weighted')
        f1_scores_transf.append(f1_transf)

    if TEST_CAT_BOOST == 1:
        f1_cb = f1_score(Y[i], y_pred_cb[i], average='weighted')
        f1_scores_cb.append(f1_cb)

# Print the F1-scores for each model
if TEST_HOEFTREE == 1:
    print(f"HoeffdingTree F1-Scores: {f1_scores_hoef}")
    print(f"Average F1-Score: {np.mean(f1_scores_hoef)}")

if TEST_CNN == 1:
    print(f"CNN F1-Scores: {f1_scores_cnn}")
    print(f"Average F1-Score: {np.mean(f1_scores_cnn)}")

if TEST_TRANSFORMER == 1:
    print(f"Transformer F1-Scores: {f1_scores_transf}")
    print(f"Average F1-Score: {np.mean(f1_scores_transf)}")

if TEST_CAT_BOOST == 1:
    print(f"CatBoost F1-Scores: {f1_scores_cb}")
    print(f"Average F1-Score: {np.mean(f1_scores_cb)}")

print("\tDone!")

