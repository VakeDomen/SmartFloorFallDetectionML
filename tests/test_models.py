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


####################### Setting up data ########################
print("Loading test data...")
X_test = np.load("../data/X_test.npy")
Y_test = np.load("../data/Y_test.npy")
print("\tDone!")

print("Reshaping data...")
X_test_flat = X_test.reshape(*X_test.shape[:-2], -1)
X_test_deep = np.expand_dims(X_test, axis=-1)

print("\tDone!")

####################### Loading models ########################

print("Loading HoeffdingTree model...")
with open('../models/HoeffdingTreeModel.pickle', 'rb') as handle:
    model_hoef_tree = pickle.load(handle)
print("\tDone!")

print("Loading CNN model...")
model_cnn = keras.models.load_model('../models/CNN')
print("\tDone!")

print("Loading transformer model...")
model_transf = keras.models.load_model("../models/transformer.h5")
print("\tDone!")

print("Loading CatBoost model...")
model_cb = CatBoostClassifier().load_model("../models/CatBoostModel.cmb")
print("\tDone!")


####################### Calculating ROC AUC ########################

print("\nPredicting probabilities for ROC AUC...")
y_pred_ht = model_hoef_tree.predict_proba(X_test_flat)
print("\tROC AUC Done! \t| HoeffdingTreeModel")
y_pred_cnn = model_cnn.predict(X_test_deep)
print("\tROC AUC Done! \t| CNN")
y_pred_transf = model_transf.predict(X_test)
print("\tROC AUC Done! \t| Transformer")
y_pred_cb = model_cb.predict_proba(X_test_flat)
print("\tROC AUC Done! \t| CatBoost")

print("\tDone!")

print("Calculating AUC ROC...")
auc_ht = roc_auc_score(Y_test, y_pred_ht[:,1])
print(f"HoeffdingTree: \t{auc_ht}")
auc_cnn = roc_auc_score(Y_test, y_pred_cnn)
print(f"CNN: \t\t{auc_cnn}")
auc_transf = roc_auc_score(Y_test, y_pred_transf)
print(f"Transformer: \t{auc_transf}")
auc_cb = roc_auc_score(Y_test, y_pred_cb[:,1])
print(f"CatBoost: \t{auc_cb}")
