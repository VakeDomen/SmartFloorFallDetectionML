import numpy as np
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


WINDOW_SIZE = 300

#!gdown --id 1c69vPJQChbqEWIvQI3sJ6osO-tLVXXPj
#!gdown --id 1HEeF49zrKyxobAJl1JX7UonVs_F2v69S

#load both as pandas df
positive_set = pd.read_csv('positiveSet.csv')
negative_set = pd.read_csv('negativeSet.csv')

# Drop columns that are not needed
# only keep sample ID and sensor info
positive_set = positive_set.drop(columns=["person_ID", "fall_category", "tick"])
negative_set = negative_set.drop(columns=["neg_category", "person_ID", "tick"])

def extract_windows(data, col):
  for val in data[col].unique():
    sample_data = data.loc[data[col] == val]
    sample_data = sample_data.drop(columns=[col]).to_numpy()
    for window_offset in range(len(sample_data) - WINDOW_SIZE + 1):
      window = sample_data[ window_offset : window_offset + WINDOW_SIZE, : ]
      yield window / 65537


pdata = list(extract_windows(positive_set, "fall_ID"))
ndata = list(extract_windows(negative_set, "neg_ID"))
        
indices = np.random.choice(len(pdata), size=(len(pdata) - len(ndata)), replace=False)
pdata = np.delete(pdata, indices, axis=0)


X_train, X_test, Y_train, Y_test = train_test_split(
  np.concatenate((
    pdata,
    ndata
  )),
  np.concatenate((
    np.ones(len(pdata)),
    np.zeros(len(ndata))
  )),
  test_size=0.2,
  random_state=4,
)
del pdata, ndata
        
        
model=Sequential()
#adding convolution layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(300,16,1)))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#adding output layer
model.add(Dense(1,activation='sigmoid'))
#compiling the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


history = model.fit(X_train,Y_train,epochs=10, validation_data=(X_test, Y_test))
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
model.predict(np.array([ndata[20500]]))
print(test_loss, test_acc)






