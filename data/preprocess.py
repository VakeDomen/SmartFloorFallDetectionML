import math
import random
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
import catboost as cb
from sklearn.utils import shuffle
import configparser
import csv

config = configparser.ConfigParser()
config.read('../config.ini')

FOLDS               = int(config.get('data-preprocess', 'folds'))
WINDOW_SIZE         = int(config.get('data-preprocess', 'window_size'))
SEED                = int(config.get('general', 'random_seed'))
SENSOR_UPPER_BOUND  = int(config.get('data', 'sensor_upper_bound'))
SENSOR_LOWER_BOUND  = int(config.get('data', 'sensor_lower_bound'))
SESNSORS_FILE       = config.get('data', 'sensor_matrix')
DATA_FILE_NAME      = config.get('data', 'data_file_name')
GROUP_COL           = config.get('data', 'group_col')
PREDICT_COL         = config.get('data', 'predict_col')
KERNEL              = eval(config.get('data', 'shape_kernel'))

np.random.seed(SEED)

def load_sensor_matrix():
    with open('../floor.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        return [row for row in reader]


def strip_table(data, needed_cols):
    # Get the intersection of the column names in the DataFrame and the needed columns
    cols_to_keep = list(set(data.columns) & set(needed_cols))
    # Return a new DataFrame with only the selected columns
    return data[cols_to_keep]

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



def get_kernel_values(matrix, kernel):
    """
    Given a matrix of strings of size NxM and a kernel (x, y),
    returns a list containing arrays of strings that are the values
    of the matrix that are contained in respective position of
    the kernel.

    Parameters:
        - matrix (2D numpy array): the matrix of strings
        - kernel (tuple of integers): the size of the kernel (x, y)

    Returns:
        - a list of arrays of strings containing the values of the
        matrix that are contained in respective position of the kernel.
    """
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

"""
def extract_windows(data, filter_col, included_vals=[], sensor_matrix=[]):
    unique_vals = data[filter_col].unique()
    if len(included_vals) > 0:
        unique_vals = np.intersect1d(unique_vals, included_vals)

    sensor_matrix = np.array(sensor_matrix)
    for val in unique_vals:
        sample_data = data[data[filter_col] == val].values
        for kernel_values in get_kernel_values(sensor_matrix, KERNEL):
            reordered_data = sample_data[:, np.isin(data.columns, kernel_values)]
            for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
                window = reordered_data[window_offset : window_offset + WINDOW_SIZE, :]
                yield (window - SENSOR_LOWER_BOUND) / (SENSOR_UPPER_BOUND - SENSOR_LOWER_BOUND)


def extract_windows(data, filter_col, included_vals=[], sensor_matrix=[]):
    for val in data[filter_col].unique():
        if len(included_vals) > 0 and val not in included_vals:
            continue
        sample_data = data.loc[data[filter_col] == val]
        for mask in get_kernel_values(sensor_matrix, KERNEL):
            reordered_data = sample_data.iloc[:, sample_data.columns.isin(mask)]
            for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
                window = (reordered_data.iloc[window_offset: window_offset + WINDOW_SIZE, :]
                          - SENSOR_LOWER_BOUND) / (SENSOR_UPPER_BOUND - SENSOR_LOWER_BOUND)
                yield window.values
"""

def extract_windows(data, filter_col, included_vals=[], sensor_matrix=[]):
    data_array = data.to_numpy()  # Convert data to numpy array
    n_rows, n_cols = data.shape
    kernel_values = get_kernel_values(sensor_matrix, KERNEL)
    for val in data[filter_col].unique():
        if len(included_vals) > 0:
            if val not in included_vals:
                continue
        sample_data = data.loc[data[filter_col] == val]
        sample_data_array = sample_data.to_numpy()  # Convert sample_data to numpy array
        for mask in kernel_values:
            #print("...")
            #print(sample_data.columns)
            #print(mask)
            reordered_data = sample_data[sample_data.columns[sample_data.columns.isin(mask)]]
            #print(f"len {len(reordered_data.columns)} cols: {reordered_data.columns}")
            reordered_data_array = reordered_data.to_numpy()  # Convert reordered_data to numpy array
            for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
                window = reordered_data_array[ window_offset : window_offset + WINDOW_SIZE, : ]
                window = (window - SENSOR_LOWER_BOUND) / (SENSOR_UPPER_BOUND - SENSOR_LOWER_BOUND)
                #print(window.shape)
                yield window


def get_unique_strings(matrix):
    # Create a set to store unique strings
    unique_strings = set()
    # Iterate over each element in the matrix
    for row in matrix:
        for string in row:
            # Add each string to the set
            unique_strings.add(string)    
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
    #print(f"Label: {lab}")
    for j in tqdm(range(FOLDS)):
        windowed_data[i].append(list(extract_windows(data_set, GROUP_COL, folds[i][j], sensors)))
        #print(windowed_data[i])
        #print(f"extracted {len(windowed_data[i][j])} windows for label {lab} in fold {j} of shape np.array(windowed_data[i]).shape")

min_sample = len(windowed_data[0][0])
for i in range(len(labels)):
    for j in range(FOLDS):
        if len(windowed_data[i][j]) < min_sample:
            min_sample = len(windowed_data[i][j])
print(f"Min sample: {min_sample}")

print("Balancing sets...")
for i in range(len(labels)):
    for j in tqdm(range(FOLDS)):
        indices = np.random.choice(len(windowed_data[i][j]), size=(len(windowed_data[i][j]) - min_sample), replace=False)
        #windowed_data[i][j] = np.delete(windowed_data[i][j], indices, axis=0)
        #windowed_data[i][j] = [x for i, x in enumerate(windowed_data[i][j]) if i not in indices]
        # Convert the list of indices to a set for faster lookups
        index_set = set(indices)
        # Use a list comprehension to filter the elements not in index_set
        windowed_data[i][j] = [x for idx, x in enumerate(windowed_data[i][j]) if idx not in index_set]


"""
# Balancing sets
for i in range(len(labels)):
    # Concatenate windows for this set and flatten into a 2D array
    windows = np.concatenate(windowed_data[i], axis=0)

    # Generate random indices to select a subset of windows
    indices = np.random.choice(len(windows), size=min_sample, replace=False)

    # Create new array with only the selected windows
    windows = windows[indices]

    # Split the new array back into the original sets
    start = 0
    for j in range(FOLDS):
        end = start + len(windowed_data[i][j])
        windowed_data[i][j] = balanced_windows[start:end]
        start = end

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

print("Shuffling data...")
for i in tqdm(range(FOLDS)):
    X[i], Y[i] = shuffle(X[i], Y[i], random_state=SEED)

print("Saving data...")
for i in tqdm(range(FOLDS)):
    np.save(f"folds/X{i}.npy", X[i])
    np.save(f"folds/Y{i}.npy", Y[i])
    

