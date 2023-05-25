# Project Scripts

1. ## Run whole project

```
./run_all.sh
```
This is a Bash shell script file named `run_all.sh` that is designed to automate the process of running an entire project from start to finish. The script consists of a series of commands that execute other sub-scripts, which download, preprocess, and build models from data, and finally test the models and store the results in the results folder. The script executes the following sub-scripts in order: 
 - clear_data.sh
 - data_setup.sh
 - clear_models.sh
 - build_models.sh
 - clear_results.sh
 - test_models.sh


2. ## Clear data

```
./clear_data.sh
```
This is a Bash shell script file named `clear_data.sh` that is designed to clear any possible data from the data folder of the project. The script executes the following commands in order: change directory (cd) to the parent directory of the data folder, remove the `folds` directory and `dataset_smartfloor.csv` file from the data folder, and create a new `folds` directory in the data folder. This script is intended to be used as a sub-script within the project automation process to ensure a clean slate for model trainig.

3. ## Clear models

```
./clear_models.sh
```
This is a Bash shell script file named `clear_models.sh` that is designed to clear any previously built models from the models folder of the project. The script executes the following commands in order: change directory (cd) to the parent directory of the models folder, remove all files and directories within the `CatBoost`, `CNN`, `HoeffdingTree`, and `Transformer` subdirectories within the models folder. This script is intended to be used as a sub-script within the project automation process to ensure a clean slate for model building.

4. ## Clear results

```
./clear_results.sh
```
This is a Bash shell script file named `clear_results.sh` that is designed to clear any previously generated results from the results folder of the project. The script removes all files and directories within the `results` folder's parent directory and creates a new `plots` directory within the `results` directory. This script is intended to be used as a sub-script within the project automation process to ensure a clean slate for result storing.

5. ## Setup data

```
./data_setup.sh
```
The data_setup.sh file is a shell script that automates the process of downloading and preprocessing the dataset in the project. It is an essential part of the project, as it sets up the data for model training.The script navigates to the `../data` directory, which is assumed to contain the necessary data files and scripts and executes the `download.sh` script, which is responsible for downloading the dataset from Google Drive. Then it runs the `preprocess.py` Python script, which preprocesses the downloaded dataset, preparing it for model training.

6. ## Build models

```
./build_models.sh
```
This is a Bash shell script file named `build_models.sh` that is designed to build all the models included in the project using Python3. The script reads a configuration file called `config.ini` and executes the build_model function for each model specified in the configuration. The build_model function checks if a model is enabled in the config file and runs the corresponding builder script for the model if it is enabled. The builder scripts are located in the `../builders` directory and include relative paths to the `config.ini` file. This script is intended to be used as a sub-script within the project automation process to build all the models required in the project.

6. ## Test models

```
./test_models.sh
```
The test_models.sh file is a shell script that automates the process of testing the performance of the trained models in the project. It is an important part of the project, as it evaluates the quality of the models and helps you understand their effectiveness. The script navigates to the `../tests/` directory, which is assumed to contain the necessary test scripts for model evaluation. It executes the `test_auc_roc.py` Python script, which evaluates the models using the Area Under the Receiver Operating Characteristic (ROC AUC) metric. This metric helps assess the performance of classification models and is widely used for binary classification problems. Then it runs the `test_f1.py` Python script, which evaluates the models using the F1 score metric. The F1 score is another popular metric for classification problems, and it considers both precision and recall to provide a single value that balances the trade-off between false positives and false negatives.
