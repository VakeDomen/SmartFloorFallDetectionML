#!/bin/bash
#
# build_models.sh
# ----------
#
# This script runs the builders for all the models included in the project using
# python3.
#
# Author: Domen Vake
#
# MIT License
# Copyright (c) 2023 Domen Vake
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


# Read the config.ini file
config_file="../config.ini"

# Build all models
# It is important to move to the directory, as the builder scripts include 
# relative paths to the config.ini 

# Check if a model is enabled in the config file and run the corresponding script
build_model() {
  model_key=$1
  model_value=$(awk -F "=" "/^\[models\]/ {found_section=1} /^\s*${model_key}\s*=/{if(found_section) print \$2}" "${config_file}")

  if [[ $model_value -eq 1 ]]; then
    script_path=$2
    script_name=$(basename "${script_path}")
    echo "Building ${script_name%.py} model..."
    pushd "$(dirname "${script_path}")" >/dev/null
    python3 "${script_name}"
    popd >/dev/null
  else
    echo "Skipping ${model_key} model..."
  fi
}

# Build the models based on the configuration
build_model "cat_boost" "../builders/CatBoost/CatBoost.py"
build_model "CNN" "../builders/CNN/CNN.py"
build_model "transformer" "../builders/TransformerModel/transformer.py"
build_model "hoeffding_tree" "../builders/HoeffdingTree/HoeffdingTreeClassifier.py"
