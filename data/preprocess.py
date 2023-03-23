import math
import numpy as np
import gc
import pandas as pd
import catboost as cb
from sklearn.utils import shuffle



WINDOW_SIZE = 300


def diff(list1, list2):
    li_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return li_dif   

def extract_windows(data, col, vals=[]):
  for val in data[col].unique():
    if len(vals) > 0:
        if val not in vals:
            continue
    sample_data = data.loc[data[col] == val]
    sample_data = sample_data.drop(columns=[col]).to_numpy()
    for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
      window = sample_data[ window_offset : window_offset + WINDOW_SIZE, : ]
      yield window / 65537



print("Loading raw data...")

#load both as pandas df
positive_set = pd.read_csv('positiveSet.csv')
negative_set = pd.read_csv('negativeSet.csv')

# Drop columns that are not needed
# only keep sample ID and sensor info
positive_set = positive_set.drop(columns=["person_ID", "fall_category", "tick"])
negative_set = negative_set.drop(columns=["neg_category", "person_ID", "tick"])

print("Selecting falls for train and test...")
np.random.seed(0)
train_size = math.floor(len(positive_set["fall_ID"].unique()) * 0.8)
train_falls_ids = np.random.choice(positive_set["fall_ID"].unique(), train_size, replace=False)
test_falls_ids = np.setdiff1d(positive_set["fall_ID"].unique(), train_falls_ids) 



print(f"ALL FALLS: {len(positive_set['fall_ID'].unique())}")
print(f"TRAIN FALLS: {len(train_falls_ids)}")
print(f"TEST FALLS: {len(test_falls_ids)}")

print("Extracting data from the sets...")
pdata_train = list(extract_windows(positive_set, "fall_ID", train_falls_ids))
pdata_test = list(extract_windows(positive_set, "fall_ID", test_falls_ids))

ndata = list(extract_windows(negative_set, "neg_ID"))



print("Balancing sets...")

# balance the poisitve and negative data sizes
# train
X_train_p = pdata_train 
X_train_n = ndata
X_test_p = pdata_test
X_test_n = ndata

len_diff = len(pdata_train) - len(ndata)
if len_diff > 0:
    indices = np.random.choice(len(pdata_train), size=(len_diff), replace=False)
    X_train_p = np.delete(pdata_train, indices, axis=0)
else:
    indices = np.random.choice(len(ndata), size=(-len_diff), replace=False)
    X_train_n = np.delete(ndata, indices, axis=0)

# test
len_diff = len(pdata_test) - len(ndata)
if len_diff > 0:
    indices = np.random.choice(len(pdata_test), size=(len_diff), replace=False)
    X_test_p = np.delete(pdata_test, indices, axis=0)
else:
    indices = np.random.choice(len(ndata), size=(-len_diff), replace=False)
    X_test_n = np.delete(ndata, indices, axis=0)





print("Merging sets...")

# concatinate positive and negative data into one set
X_train = np.concatenate((
    X_train_p,
    X_train_n
))

Y_train = np.concatenate((
    np.ones(len(X_train_p)),
    np.zeros(len(X_train_n))
))


X_test = np.concatenate((
    X_test_p,
    X_test_n
))

Y_test = np.concatenate((
    np.ones(len(X_test_p)),
    np.zeros(len(X_test_n))
))

print("Shuffling data...")

X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

print("Saving data...")

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)


