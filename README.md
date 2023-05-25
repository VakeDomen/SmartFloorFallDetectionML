# Time Series Classification: Smart Floor

The project is designed to quickly prototype multiple machine learning models to classify data for a given  grid-like sensor network. Developing an accurate and efficient machine learning (ML) model based on sensor data can be a time-consuming and iterative process. This project aims to reduce the time and effort needed to develop such models by providing a tool that can quickly prototype multiple machine-learning models for grid-like sensor networks. This tool can help to reduce the time and effort needed to develop models for grid-like sensor networks, as well as improve the accuracy and efficiency of the models. Additionally, the tool can help to reduce the need for extensive preprocessing of the data, as well as the need to experiment with multiple algorithms and fine-tune their hyperparameters. The models employed in this project include Convolutional Neural Networks (CNNs), Transformers, CatBoost, and Hoeffding Trees. The performance of these models is evaluated using ROC AUC scores, F1 scores.

## Table of contents
- [Structure](#structure)
- [Setup](#setup)
- [Usage](#usage)
- [Modification](#modification)
- [License](#license)
- [Author](#author)

## Structure

Folder structure of the project:
- `data`: contains data download and preprocess scripts
- `builders`: contains scripts to build ML models on the preprocessed data
- `models`: after building the models the built models are stored in this folder
- `tests`: stores the testing scripts for the models
- `results`: stores the outputs of the testing scripts, such as plots
- `scripts`: contains the scripts to run different parts of the project workflow


## Setup

1. Clone the repository
```
git clone https://github.com/VakeDomen/SmartFloorFallDetectionML.git
cd SmartFloorFallDetectionML
```

2. Install required packages
```
pip install -r requirements.txt
```

## Usage

The folder `scripts` contains multiple useful scripts that you might want to use in the project.

There are 3 important steps to complete prior to using the software. First you need to supply your data. See the [README](src/data/README.md) in the data folder.
Second you may want to modify the `config.ini` to suit your needs (see more under [Modification](#modification)) and third you should describe your topology in a `.csv` file and reference it 
in the `config.ini` file (reference file `floor.csv` is provided for you).

Once you have completed the above tasks you may want to navigate the `scripts` folder:
```
cd scripts
```
Run the entire pipeline, including data preprocessing, model training, and model evaluation.
```
./run_all.sh
```
There are more options for you, described in the [README](src/scripts/README.md) in the scripts folder.

## Modification

1. ### config.ini

The config.ini file contains various sections and key-value pairs to configure the data preprocessing, models, and their parameters.

- [general]: General settings for the project.

    - `random_seed`: Random seed for reproducibility (default: 42).

- [data]: Settings related to the raw data and its preprocessing.

    - `sensor_upper_bound`: Upper bound for sensor values - used for normalization (default: 65537).
    - `sensor_lower_bound`: Lower bound for sensor values - used for normalization (default: 0).
    - `data_gdrive_id`: Google Drive ID for the raw data file.
    - `data_file_name`: Name of the raw data file (default: dataset_smartfloor.csv).
    - `sensor_matrix`: Name of the sensor matrix file (default: floor.csv).
    - `group_col`: Name of the group column in the data - should respresent one time series sample (default: id).
    - `predict_col`: Name of the target prediction column in the data (default: class).
    - `shape_kernel`: Shape of the kernel for sensor data extraction window (default: (4, 4)).

- [data-preprocess]:Settings for data preprocessing.

    - `folds`: Number of folds for cross-validation (default: 5).
    - `window_size`: Window size for data preprocessing (default: 300).

- [models]: Boolean flags to indicate which models to build and test.

    - `cat_boost`: Flag for training and testing the CatBoost model (default: 1).
    - `CNN`: Flag for training and testing the CNN model (default: 1).
    - `transformer`: Flag for training and testing the Transformer model (default: 1).
    - `hoeffding_tree`: Flag for training and testing the Hoeffding Tree model (default: 1).

- [model-cat-boost]: Settings for the CatBoost model.

    - `iterations`: Number of iterations for training (default: 1000).
    - `learning_rate`: Learning rate for the model (default: 0.1).
    - `depth`: Maximum depth of the trees (default: 6).
    - `l2_leaf_reg`: L2 regularization coefficient (default: 3).
    - `loss_function`: Loss function to optimize (default: Logloss).
    - `random_seed`: Random seed for reproducibility (default: 42).
    - `task_type`: Type of hardware to use for training (default: GPU).
    - `devices`: GPU devices to use for training (default: 0:1).

- [model-cnn]: Settings for the CNN model.

    - `n_conv_filters`: Number of convolutional filters (default: 32).
    - `conv_activation`: Activation function for the convolutional layers (default: relu).
    - `shape_kernel`: Shape of the kernel for the CNN model (default: (3, 3)).
    - `shape_max_pool`: Shape of the max pooling layer (default: (2, 2)).
    - `dense_layers_units`: List of units for dense layers (default: [100]).
    - `dense_layers_activations`: List of activation functions for dense layers (default: ["sigmoid"]).
    - `loss_function`: Loss function to optimize (default: binary_crossentropy).
    - `optimizer`: Optimizer for training (default: adam).
    - `metrics`: List of evaluation metrics (default: ["accuracy"]).
    - `epochs`: Number of epochs for training (default: 10).

- [model-transformer]: Settings for the Transformer model.

    - `hidden_size`: Hidden size of the Transformer (default: 128).
    - `num_layers`: Number of layers in the Transformer (default: 2).
    - `num_heads`: Number of attention heads in the transformer (default: 4)
    - `encoder_layer_activation`: Activation function for the encoder layer (default: relu).
    - `dropout_rate`: Dropout rate for the Transformer (default: 0.2).
    - `batch_size`: Batch size for training (default: 32).
    - `learning_rate`: Learning rate for the model (default: 1e-4).
    - `epochs`: Number of epochs for training (default: 10).
    - `metrics`: List of evaluation metrics (default: ["accuracy"]).

- [model-hoeffding-tree]: Settings for the Hoeffding Tree model.

    - `max_byte_size`: Maximum memory consumed by the tree (default: 33554432).
    - `split_criterion`: Split criterion to use (default: info_gain).
    - `leaf_prediction`: Type of leaf prediction to use (default: mc).
    - `batch_size`: Batch size for training (default: 100).

Remember to adjust these parameters according to your dataset and requirements.

2. ### floor.csv
The `floor.csv` file contains the topological configuration of your sensors. Note that the name and the location of this file are selected arbitratily and can be changed in the `config.ini` under the `data`
section under the `sensor_matrix` attribute.

The function of this file is to tell the preprocessing script how the sensors are structured topologically,
such that the script can extract the data of the adjacent sensors and group the sensor data into windows.
The sensors selected are calulcated in from the `shape_kernel` attribute. 

Every value in the file should correspond to a column in the dataset, that represents the data collected by the specified sensor. In the test data that we provide, the senros were named `s0`, `s1`,`s2`...`s15` and were stup in a 4x4 grid. The matrix then looks like this:
```
s0,s1,s2,s3
s4,s5,s6,s7
s8,s9,s10,s11
s12,s13,s14,s15
```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Author

Domen Vake - [GitHub](https://github.com/VakeDomen)
