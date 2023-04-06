config_file="./../config.ini"
#positive_key="positive_data_gdrive_id"
#negative_key="negative_data_gdrive_id"

set_key="data_gdrive_id"

data_id=$(awk -F "=" "/^\[data\]/ {found_section=1} /^\s*${set_key}\s*=/{if(found_section) print \$2}" "${config_file}")

#pos_id=$(awk -F "=" "/^\[data\]/ {found_section=1} /^\s*${positive_key}\s*=/{if(found_section) print \$2}" "${config_file}")
#neg_id=$(awk -F "=" "/^\[data\]/ {found_section=1} /^\s*${negative_key}\s*=/{if(found_section) print \$2}" "${config_file}")

if command -v gdown >/dev/null 2>&1; then
    gdown --id $data_id
    #gdown --id $neg_id 
else
    echo "gdown is not installed. Please install it and run the script again."
fi

