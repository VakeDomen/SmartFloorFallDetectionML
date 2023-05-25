#!/bin/bash
#
# download.sh
# ----------
#
# This script downloads data from Google Drive using gdown. The Google Drive
# ID of the data file is read from a config.ini file.
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


# Read the data Google Drive ID from the config.ini file
config_file="./../config.ini"
set_key="data_gdrive_id"
data_id=$(awk -F "=" "/^\[data\]/ {found_section=1} /^\s*${set_key}\s*=/{if(found_section) print \$2}" "${config_file}")

# Check if gdown is installed and download the data
if command -v gdown >/dev/null 2>&1; then
    gdown --id $data_id
else
    echo "gdown is not installed. Please install it and run the script again."
fi

