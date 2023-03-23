import numpy as np
import pandas as pd


positive_set = pd.read_csv('positiveSet.csv')
negative_set = pd.read_csv('negativeSet.csv')

window_size = 300

# Drop columns that are not needed
# only keep fall ID and sensor info
data_selected = positive_set.drop(columns=["person_ID", "fall_category", "tick"])
# all unique ids of falls 1,2,3...
fall_ids = data_selected['fall_ID'].unique()

# windowed data collection array
windowed_positive_data = []

for val in fall_ids:
  # group all time series data for one fall
  fall_data = data_selected.loc[data_selected['fall_ID']==val]
  # remove id column (only left with sensor data)
  fall_data1 = fall_data.drop(columns=["fall_ID"]).to_numpy()
  fall_data2 = fall_data.drop(columns=["fall_ID"]).reindex(columns=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14","s15"].reverse()).to_numpy()
  # create sliding windows for the fall
  for window_offset in range(0, len(fall_data) - window_size + 1):
    window = fall_data1[window_offset:window_offset + window_size,:]
    windowed_positive_data.append(window.flatten('C'))


# Drop columns that are not needed
# only keep negative ID and sensor info
data_selected = negative_set.drop(columns=["neg_category", "person_ID", "tick"])

# all unique ids of neg 1,2,3... 
neg_ids = data_selected['neg_ID'].unique()

# windowed data collection array
windowed_negative_data = []

for val in neg_ids:
  # group all time series data for one negative data event 
  neg_data = data_selected.loc[data_selected['neg_ID']==val]
  # remove id column (only left with sensor data)
  neg_data = neg_data.drop(columns=["neg_ID"]).to_numpy()
  # create sliding windows for the fall
  for window_offset in range(0, len(neg_data) - window_size + 1):
    window = neg_data[window_offset:window_offset + window_size,:]
    windowed_negative_data.append(window.flatten('C'))

idx = np.random.choice(len(windowed_positive_data), size=len(windowed_negative_data), replace=False)
selected_windows = np.empty((len(windowed_negative_data), 4800))
i = 0 
for index in idx:
    selected_windows[i] = (windowed_positive_data[index])
    i += 1
windowed_positive_data = selected_windows

X = np.empty((len(windowed_positive_data) + len(windowed_negative_data), 4800))


for i in range(len(windowed_positive_data)):
  X[i] = windowed_positive_data[i]
for i in range(len(windowed_negative_data)):
  X[i + len(windowed_positive_data)] = windowed_negative_data[i]


Y = np.concatenate((
    np.ones(len(windowed_positive_data)), 
    np.zeros(len(windowed_negative_data))
))

from sklearn.model_selection import train_test_split
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import HoeffdingTreeRegressor
from tqdm import tqdm

X_train, X_test, Y_train, Y_test = train_test_split(
  X, Y, test_size=0.2, random_state=4,
)


model = HoeffdingTreeRegressor(leaf_prediction='mean')
y_pred = np.empty(len(X_train))


model.fit(X_train, Y_train)

# Display results
print('{} samples analyzed.'.format(len(X_train)))
print('Hoeffding Tree regressor mean absolute error: {}'.
      format(np.mean(np.abs(Y_train - y_pred))))



