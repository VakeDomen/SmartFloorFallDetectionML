
"""
test_auc_roc.sh
-----------

This script calculates the ROC AUC scores for the trained models and generates
ROC curve plots for each fold and for the total dataset.


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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from tensorflow import keras
from catboost import CatBoostClassifier
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
from tqdm import tqdm
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
FOLDS               = int(config.get('data-preprocess', 'folds')) 
TEST_CAT_BOOST      = int(config.get('models', 'cat_boost'))
TEST_CNN            = int(config.get('models', 'CNN'))
TEST_TRANSFORMER    = int(config.get('models', 'transformer'))
TEST_HOEFTREE       = int(config.get('models', 'hoeffding_tree'))


def calc_roc_auc(y_tests, y_preds, model_name):
    scores = []
    for i in tqdm(range(FOLDS)):
        score = roc_auc_score(y_tests[i], y_preds[i])
        scores.append(score)
        plt_roc_curve(y_tests[i], y_preds[i], score, f"auc_roc_{model_name}_f{i}")
    tests = np.concatenate(y_tests)
    preds =  np.concatenate(y_preds)
    total_score = roc_auc_score(tests, preds)
    scores.append(total_score)
    plt_roc_curve(tests, preds, total_score, f"auc_roc_{model_name}")
    return scores

def plt_roc_curve_open(y_test, y_pred, score, fname):
    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    plt.figure(2)
    #create ROC curve
    plt.plot(fpr,tpr,label=f"AUC {fname}="+str(score))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)

def save_plot():
    plt.figure(2)
    plt.savefig(f"../results/plots/auc_roc.png")
    plt.close()

def plt_roc_curve(y_test, y_pred, score, fname):
    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    plt.figure(1)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(score))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(f'../results/plots/{fname}.png')
    plt.close()

def roc_auc(Y_test, Y_pred, name):
    score = calc_roc_auc(Y_test, Y_pred, name)
    plt_roc_curve_open(np.concatenate(Y_test), np.concatenate(Y_pred), score[-1], name)
    return score


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


####################### Calculating ROC AUC ########################

print("\nPredicting probabilities for ROC AUC...")

if TEST_HOEFTREE == 1:
    y_pred_ht = []
    print("\tHoeffdingTreeModel")
    for i in tqdm(range(FOLDS)):
        y_pred_ht.append(models_hoef[i].predict_proba(X_flat[i])[:,1])

if TEST_CNN == 1:
    y_pred_cnn = []
    print("\tCNN")
    for i in tqdm(range(FOLDS)):
        y_pred_cnn.append(models_cnn[i].predict(X_deep[i]))

if TEST_TRANSFORMER == 1:
    y_pred_transf = []
    print("\tTransformer")
    for i in tqdm(range(FOLDS)):
        y_pred_transf.append(models_transf[i].predict(X[i]))

if TEST_CAT_BOOST == 1:
    y_pred_cb = []
    print("\tCatBoost")
    for i in tqdm(range(FOLDS)):
        y_pred_cb.append(models_cb[i].predict_proba(X_flat[i])[:,1])

print("\tDone!")


print("Calculating AUC ROC...")
if TEST_HOEFTREE == 1:
    score_ht = roc_auc(Y, y_pred_ht, "ht")
if TEST_CNN == 1:
    score_cnn = roc_auc(Y, y_pred_cnn, "cnn")
if TEST_TRANSFORMER == 1:
    score_transf = roc_auc(Y, y_pred_transf, "transf")
if TEST_CAT_BOOST == 1:
    score_cb = roc_auc(Y, y_pred_cb, "cb")

save_plot()

if TEST_HOEFTREE == 1:
    print(f"Hoeffding Tree: \t{score_ht}")
if TEST_CNN == 1:
    print(f"CNN: \t{score_cnn}")
if TEST_TRANSFORMER == 1:
    print(f"Transformer: \t{score_transf}")
if TEST_CAT_BOOST == 1:
    print(f"CatBoost: \t{score_cb}")


