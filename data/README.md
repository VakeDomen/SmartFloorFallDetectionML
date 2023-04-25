# Data

This project contains two main files, `download.sh` and `preprocess.py`, which are responsible for downloading and preprocessing sensor data, respectively.

## download.sh

`download.sh` is a shell script that downloads data from Google Drive using the `gdown` command line utility. The Google Drive ID of the data file is read from a `config.ini` file. The script checks if `gdown` is installed and downloads the data using the provided Google Drive ID.

## preprocess.py

`preprocess.py` is a Python script that loads, preprocesses, and saves sensor data into separate folds. The main steps in the preprocessing process are:

1. Load raw data and sensor matrix
2. Split data by prediction label
3. Create folds for each label
4. Extract sliding windows of data from each fold
5. Balance the number of samples in each set
6. Shuffle and save the preprocessed data

The script reads various configurations from a `config.ini` file and uses them to preprocess the data. The preprocessed data is saved in the `folds` directory, with each fold containing an `X.npy` and a `Y.npy` file, representing the feature data and labels, respectively.

To use these scripts, first, run `download.sh` to download the data, and then run `preprocess.py` to preprocess the data and generate the folds for further analysis or modeling.
