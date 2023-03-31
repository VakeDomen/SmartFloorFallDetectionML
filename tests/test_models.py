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
FOLDS = int(config.get('data-preprocess', 'folds')) 

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

    #create ROC curve
    plt.plot(fpr,tpr,label=f"AUC {fname}="+str(score))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)

def save_plot():
    plt.savefig(f"../results/plots/auc_roc.png")
    plt.close()

def plt_roc_curve(y_test, y_pred, score, fname):
    fpr, tpr, _ = roc_curve(y_test,  y_pred)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(score))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(f'../results/plots/{fname}.png')
    plt.close()



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
    y_pred_ht.append(models_hoef[i].predict_proba(X_flat[i])[:,1])
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
    y_pred_cb.append(models_cb[i].predict_proba(X_flat[i])[:,1])
print("\tROC AUC Done! \t| CatBoost")

print("\tDone!")

print("Calculating AUC ROC...")
score_ht        = calc_roc_auc(Y, y_pred_ht, "ht")
score_cnn       = calc_roc_auc(Y, y_pred_cnn, "cnn")
score_transf    = calc_roc_auc(Y, y_pred_transf, "transf")
score_cb        = calc_roc_auc(Y, y_pred_cb, "cb")

plt_roc_curve_open(np.concatenate(Y), np.concatenate(y_pred_ht), score_ht[-1], "ht")
plt_roc_curve_open(np.concatenate(Y), np.concatenate(y_pred_cnn), score_cnn[-1], "cnn")
plt_roc_curve_open(np.concatenate(Y), np.concatenate(y_pred_transf), score_transf[-1], "transf")
plt_roc_curve_open(np.concatenate(Y), np.concatenate(y_pred_cb), score_cb[-1], "cb")
save_plot()
"""
y_preds_ht = []
for i in tqdm(range(FOLDS)):
    y_preds_ht.append(y_pred_ht[i][:,1]))
    plt_roc_curve(Y[i], y_pred_ht[i][:,1], auc_ht[i], f"auc_roc_hof_tree_f{i}")


print(f"HoeffdingTree: \t{auc_ht}")
auc_cnn = []
for i in tqdm(range(FOLDS)):
    auc_cnn.append(roc_auc_score(Y[i], y_pred_cnn[i]))
    plt_roc_curve(Y[i], y_pred_cnn[i], auc_cnn[i], f"auc_roc_cnn_f{i}")

print(f"CNN: \t\t{auc_cnn}")
auc_transf = []
for i in tqdm(range(FOLDS)):
    auc_transf.append(roc_auc_score(Y[i], y_pred_transf[i]))
    plt_roc_curve(Y[i], y_pred_transf[i], auc_transf[i], f"auc_roc_transf_f{i}")

print(f"Transformer: \t{auc_transf}")
auc_cb = []
for i in tqdm(range(FOLDS)):
    auc_cb.append(roc_auc_score(Y[i], y_pred_cb[i][:,1]))
    plt_roc_curve(Y[i], y_pred_cb[i][:,1], auc_cb[i], f"auc_roc_cb_f{i}")
"""
print(f"Hoeffding Tree: \t{score_ht}")
print(f"CNN: \t{score_cnn}")
print(f"Transformer: \t{score_transf}")
print(f"CatBoost: \t{score_cb}")


