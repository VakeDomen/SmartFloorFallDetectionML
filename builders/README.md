# Builders

This folder contains scripts for training different machine learning models on preprocessed data using k-fold cross-validation. Each script reads the model's parameters from a `config.ini` file.

## Files

1. [CatBoost Classifier](./CatBoost/CatBoost.py): This script trains a CatBoost classifier on preprocessed data and saves the resulting model for each fold.

2. [Hoeffding Tree Classifier](./HoeffdingTree/HoeffdingTreeClassifier.py): This script trains a Hoeffding Tree classifier on preprocessed data and saves the resulting model for each fold.

3. [Transformer](./TransformerModel/transformer.py): This script trains a Transformer classifier on preprocessed data and saves the resulting model for each fold.

4. [CNN](./CNN/CNN.py): This script trains a CNN classifier on preprocessed data and saves the resulting model for each fold.

## Usage

To train a model using one of the scripts, run the corresponding script from the command line. Make sure that the preprocessed data is available in the appropriate folder and that the `config.ini` file contains the desired model parameters.

Example:

```
cd CatBoost
python3 CatBoost.py
```

This will train a CatBoost classifier on the preprocessed data using the parameters specified in the `config.ini` file.

## Dependencies

The following dependencies are required to run the scripts:

- NumPy
- TensorFlow
- Keras
- CatBoost
- skmultiflow
- tqdm
- configparser
- gc
