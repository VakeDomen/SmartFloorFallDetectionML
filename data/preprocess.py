import math
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import catboost as cb
from sklearn.utils import shuffle
import configparser
import csv
import os

config = configparser.ConfigParser()
config.read('../config.ini')

FOLDS               = int(config.get('data-preprocess', 'folds'))
WINDOW_SIZE         = int(config.get('data-preprocess', 'window_size'))
SEED                = int(config.get('general', 'random_seed'))
SENSOR_UPPER_BOUND  = int(config.get('data', 'sensor_upper_bound'))
SENSOR_LOWER_BOUND  = int(config.get('data', 'sensor_lower_bound'))
SENSORS_FILE        = config.get('data', 'sensor_matrix')
DATA_FILE_NAME      = config.get('data', 'data_file_name')
GROUP_COL           = config.get('data', 'group_col')
PREDICT_COL         = config.get('data', 'predict_col')
KERNEL              = eval(config.get('data', 'shape_kernel'))


np.random.seed(SEED)
random.seed(SEED)

def load_sensor_matrix():
    with open(f'../{SENSORS_FILE}', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        return [row for row in reader]


def strip_table(data, needed_cols):
    return data.filter(items=needed_cols)

def split_list(numbers, m):
    random.shuffle(numbers)
    sublists = [[] for _ in range(m)]
    for num in numbers:
        smallest_list = min(sublists, key=len)
        smallest_list.append(num)
    return sublists


def get_kernel_values(matrix, kernel):
    n, m = len(matrix), len(matrix[0])
    kx, ky = kernel

    for rot in range(4):
        rotated_matrix = np.rot90(matrix, rot)
        for i in range(n - kx + 1):
            for j in range(m - ky + 1):
                values = []
                for ki in range(kx):
                    for kj in range(ky):
                        values.append(rotated_matrix[i+ki][j+kj])
                yield values


def extract_windows(data, filter_col, included_vals=[], sensor_matrix=[]):
    data_array = data.to_numpy()  # Convert data to numpy array
    n_rows, n_cols = data.shape
    kernel_values = get_kernel_values(sensor_matrix, KERNEL)
    for val in data[filter_col].unique():
        if len(included_vals) > 0:
            if val not in included_vals:
                continue
        sample_data = data.loc[data[filter_col] == val]
        for mask in kernel_values:
            reordered_data = sample_data[sample_data.columns[sample_data.columns.isin(mask)]]
            reordered_data_array = reordered_data.to_numpy()
            for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
                window = reordered_data_array[ window_offset : window_offset + WINDOW_SIZE, : ]
                window = (window - SENSOR_LOWER_BOUND) / (SENSOR_UPPER_BOUND - SENSOR_LOWER_BOUND)
                yield window


def get_unique_strings(matrix):
    # Create a set to store unique strings
    unique_strings = {string for row in matrix for string in row}
    # Convert the set to a list and return it
    return list(unique_strings)


print("Loading raw data...")
data_set = pd.read_csv(DATA_FILE_NAME)
sensors = np.array(load_sensor_matrix())
cols_to_keep = get_unique_strings(sensors)
cols_to_keep.append(GROUP_COL)
cols_to_keep.append(PREDICT_COL)
data_set = strip_table(data_set, cols_to_keep) 

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
windowed_data = [[] for _ in range(len(labels))]
for i, lab in enumerate(labels):
    for j in tqdm(range(FOLDS)):
        windowed_data[i].append(list(extract_windows(data_set, GROUP_COL, folds[i][j], sensors)))

print("Balancing sets...")
min_sample = min(len(windowed_data[i][j]) for i in range(len(labels)) for j in range(FOLDS))

for i in range(len(labels)):
    for j in tqdm(range(FOLDS)):
        indices = np.random.choice(len(windowed_data[i][j]), size=(len(windowed_data[i][j]) - min_sample), replace=False)
        index_set = set(indices)
        windowed_data[i][j] = [x for idx, x in enumerate(windowed_data[i][j]) if idx not in index_set]


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

print("Shuffling data...")
for i in tqdm(range(FOLDS)):
    X[i], Y[i] = shuffle(X[i], Y[i], random_state=SEED)

print("Saving data...")
for i in tqdm(range(FOLDS)):
    np.save(f"folds/X{i}.npy", X[i])
    np.save(f"folds/Y{i}.npy", Y[i])
    

