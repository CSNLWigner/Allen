#!/bin/bash

# Get git name
git_name="$1"

# Get the param key
param_key="$2"

# Check the initial value of the param
if ! grep -q "$param_key: 0" params.yaml; then
    echo "Error: The initial value of $param_key should be 0 in params.yaml"
    exit 1
fi

# Get the param value list (Internal Field Separator is comma, then read by read only into array named XY from the parameter
IFS=',' read -r -a param_value_list <<< "$3"

# Loop over the parameters
for param_value in "${param_value_list[@]}"
do
    # Print the informations
    echo "$git_name $param_key $param_value"

    # Add the param key
    new_param="$param_key: $param_value"
    old_param="$param_key: 0"

    # Modify the parameters in params.yaml
    sed -i "s/$old_param/$new_param/g" params.yaml

    # Run code
    dvc repro

    # Git
    #git add .
    #git commit -m "$git_name $param_key $param_value"

    # Copy results
    filename=$(grep -w "figures" dvc.yaml | awk -d' ' '{print $2}')
    dest="cache/${git_name}_${param_value}.png"
    echo "$dest"
    cp $filename $dest

    # Reset the parameters to avoid conflicts in the next iteration
    sed -i "s/$new_param/$old_param/g" params.yaml

    # Print current date and time
    date
done