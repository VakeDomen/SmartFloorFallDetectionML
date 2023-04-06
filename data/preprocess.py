import math
import random
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
import catboost as cb
from sklearn.utils import shuffle
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

FOLDS               = int(config.get('data-preprocess', 'folds'))
WINDOW_SIZE         = int(config.get('data-preprocess', 'window_size'))
SEED                = int(config.get('general', 'random_seed'))
SENSOR_UPPER_BOUND  = int(config.get('data', 'sensor_upper_bound'))
SENSOR_LOWER_BOUND  = int(config.get('data', 'sensor_lower_bound'))
DATA_FILE_NAME      = config.get('data', 'data_file_name')
GROUP_COL           = config.get('data', 'group_col')
PREDICT_COL         = config.get('data', 'predict_col')


np.random.seed(SEED)


def split_list(numbers, m):
    # Shuffle the list randomly
    random.shuffle(numbers)
    # Create m empty sublists
    sublists = [[] for _ in range(m)]
    # Iterate through the shuffled list and append each number to the smallest sublist
    for num in numbers:
        smallest_list = min(sublists, key=len)
        if num not in smallest_list:
            smallest_list.append(num)
    return sublists


def extract_windows(data, col, vals=[]):
  for val in data[col].unique():
    if len(vals) > 0:
        if val not in vals:
            continue
    sample_data = data.loc[data[col] == val]
    sample_data = sample_data.drop(columns=[col]).to_numpy()
    for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
      window = sample_data[ window_offset : window_offset + WINDOW_SIZE, : ]
      yield (window - SENSOR_LOWER_BOUND) / (SENSOR_UPPER_BOUND - SENSOR_LOWER_BOUND)



print("Loading raw data...")
data_set = pd.read_csv(DATA_FILE_NAME)

print("Splitting data by prediction label...")
labels = data_set[PREDICT_COL].unique()
data = [[] for _ in range(len(labels)) ]
for i, lab in enumerate(labels):
    data[i] = data_set.loc[data_set[PREDICT_COL] == lab]

print("Folding data...")
folds = [[] for i in range(len(labels))]
for i, lab in enumerate(labels):
    folds[i] = split_list(data[i][GROUP_COL].unique(), FOLDS)

print("Extracting data from the sets...")
#ndata = list(extract_windows(negative_set, NEG_ID_COL))
#negative_folds = split_list([*range(len(ndata))], FOLDS)

windowed_data = [[] for _ in range(len(labels))]
for i, lab in enumerate(labels):
    for j in range(FOLDS):
        windowed_data[i].append(list(extract_windows(data_set, GROUP_COL, folds[i][j])))
        print(f"extracted {len(windowed_data[i])} windows for label {lab} in fold {j} of shape")

#positive_folds = split_list(positive_set[POS_ID_COL].unique(), FOLDS)
min_sample = len(windowed_data[0][0])
for i in range(len(labels)):
    for j in range(FOLDS):
        if len(windowed_data[i][j]) < min_sample:
            min_sample = len(windowed_data[i][j])

print(f"min: {min_sample}")
print("Balancing sets...")
for i in range(len(labels)):
    for j in tqdm(range(FOLDS)):
        indices = np.random.choice(len(windowed_data[i][j]), size=(len(windowed_data[i][j]) - min_sample), replace=False)
        windowed_data[i][j] = np.delete(windowed_data[i][j], indices, axis=0)
print(np.array(windowed_data[0]).shape)
"""
for i in tqdm(range(FOLDS)):
    X_p = list(extract_windows(positive_set, POS_ID_COL, positive_folds[i]))
    X_n = [ndata[j] for j in negative_folds[i]]

    len_diff = len(X_p) - len(X_n)
    if len_diff > 0:
        indices = np.random.choice(len(X_p), size=(len_diff), replace=False)
        X_p = np.delete(X_p, indices, axis=0)
    else:
        indices = np.random.choice(len(X_n), size=(-len_diff), replace=False)
        X_n = np.delete(X_n, indices, axis=0)

    positive_folds[i] = X_p
    negative_folds[i] = X_n
"""
X = []
Y = []

for i in tqdm(range(FOLDS)):
    X_fold = []
    Y_fold = []
    for j, lab in enumerate(labels):
        X_fold.append(windowed_data[j][i])
        Y_fold.append([lab] * len(windowed_data[j][i]))
    X.append(np.concatenate(X_fold))
    Y.append(np.concatenate(Y_fold))

"""
print("Merging data sets...")
for i in tqdm(range(FOLDS)):
    X.append(np.concatenate((
        positive_folds[i],
        negative_folds[i]
    )))
    Y.append(np.concatenate((
        np.ones(len(positive_folds[i])),
        np.zeros(len(negative_folds[i]))
    )))

"""
print(np.array(X[0]).shape)
print("Shuffling data...")
for i in tqdm(range(FOLDS)):
    X[i], Y[i] = shuffle(X[i], Y[i], random_state=SEED)

print("Saving data...")
for i in tqdm(range(FOLDS)):
    np.save(f"folds/X{i}.npy", X[i])
    np.save(f"folds/Y{i}.npy", Y[i])
    

