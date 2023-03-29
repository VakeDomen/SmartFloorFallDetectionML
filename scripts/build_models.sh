#!/bin/bash

# Build all models
cd ../builders/CNN
python3 CNN.py
cd ../TransformerModel
python3 transformer.py
cd ../CatBoost
python3 CatBoost.py
cd ../HoeffdingTree
python3 HoeffdingTreeClassifier.py
