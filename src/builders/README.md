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

## Models

### CatBoost
We chose to utilize CatBoost, a gradient boosting algorithm, due to its robustness and efficiency in handling categorical features directly. Traditional machine learning algorithms often struggle with categorical data, necessitating extensive preprocessing and feature engineering, but CatBoost effectively manages such data. It also provides an excellent balance between computational efficiency and predictive performance, achieving competitive results with fewer iterations. The algorithm's capability to avoid overfitting, along with its interpretability features, such as SHAP values for feature importance, adds to its appeal. Consequently, we believe that CatBoost will significantly enhance the accuracy and efficiency of our prediction tasks.

### Hoeffding Trees
We've decided to employ Hoeffding Trees due to their unique characteristics that perfectly suit our needs when working with sensor networks. Hoeffding Trees operate efficiently in memory-constrained environments, as they are designed to handle streaming data and construct compact decision trees. This makes them particularly valuable when deploying models directly onto sensor networks where computational resources are limited. Furthermore, the process of building Hoeffding Trees is both procedural and incremental. As data streams in, the algorithm progressively update the tree structure. It only chooses to split a node when it has seen sufficient data to make a confident decision. This guarantees that our model continues to learn and adapt as new data arrives, without making premature or uninformed decisions based on limited data. Therefore, we anticipate that using Hoeffding Trees will enhance both the efficiency and adaptability of our predictive models.

### Transformer
In the wide array of machine learning methods, Transformers stand out as both robust and adaptable. They are especially beneficial for working with sensor data, which has a time-based element. Transformers come with a feature called a 'self-attention mechanism' that cleverly picks up patterns over time, even between widely separated data points. It pinpoints important moments in the data sequence and understands how they connect, thereby improving the accuracy of our predictions. As a result, we believe using these Transformer models will significantly improve how efficiently we can process data and also enhance the reliability of our prediction tasks. 

### CNN
We chose Convolutional Neural Networks (CNNs) for the classification of our planar time-series data due to several key advantages they offer. Firstly, CNNs are adept at automatically learning and extracting high-level features from the data, reducing the need for manual feature engineering. This is particularly beneficial in our case, as the complexity and temporal nature of time-series data make manual feature extraction challenging. Additionally, the architecture of CNNs, which employs multiple layers of convolutions and pooling, is particularly well-suited to handle spatial hierarchies and temporal dependencies inherent in time-series data. This allows our model to capture both local and global trends in the data, improving the accuracy of our classification tasks. Lastly, CNNs have demonstrated impressive performance on various types of time-series data in recent research, providing further confidence in our choice of this deep-learning architecture.
