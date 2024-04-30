#!/bin/bash




# Get the param value list (Internal Field Separator is comma, then read by read only into array named XY from the parameter
IFS=',' read -r -a sessions <<< "$1"

run_function () {

    # Print the informations
    echo "session $2"

    # Add the param key
    new_param="session: $2"
    old_param="session: 0"

    # Modify the parameters in params.yaml
    sed -i "s/$old_param/$new_param/g" params.yaml

    # Run code
    dvc repro > log_cache.txt

    # Git
    git add .
    git commit -m "$1 session $2"

    # Reset the parameters to avoid conflicts in the next iteration
    sed -i "s/$new_param/$old_param/g" params.yaml

    # Print current date and time
    date

}


# Loop over the parameters
for session in "${sessions[@]}"
do

    # copy param-search
    cp pipelines/param-search/params.yaml params.yaml
    cp pipelines/param-search/dvc.yaml dvc.yaml

    # TD search
    run_function "TD_param-search_GEN3" $session

    # continue if no maximum value
    if ! grep -q 'maximum value' log_cache.txt
	continue
    fi

    # get variables from log cache
    TD_cv=$(sed -n 's/cv=\(.*\)/\1/ < log_cache.txt) # add '
    TD_rank=$(sed -n 's/rank=\(.*\)/\1/ < log_cache.txt) # add '
    echo "cv is $TD_cv"
    echo "rank is $TD_rank"

    

done






# TD search
./runcode.sh TD_param-search_GEN3 session $1 >> GEN3.txt

# sort
grep -A 18 'maximum value' GEN3.txt > TD_params.txt
sessions=$(grep -o 'session [0-9]\+' TD_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot

# clear GEN3.txt
> GEN3.txt

# BU search
./runcode.sh BU_param-search_GEN3 session $sessions >> GEN3.txt

# sort
grep -A 18 'maximum value' GEN3.txt > BU_params.txt
sessions=$(grep -o 'session [0-9]\+' BU_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot

# copy params and pipeline
cp pipelines/crosstime/params.yaml params.yaml
cp pipelines/crosstime/dvc.yaml dvc.yaml

# TD crosstest
./runcode.sh TD_cross-time_GEN3 session $sessions

# cp outdir
cp cache/* cache/TD

# BU crosstest
./runcode.sh BU_cross-time_GEN3 session $sessions

# co outdir
cp cache/* cache/BU#!/bin/bash

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
    git add .
    git commit -m "$git_name $param_key $param_value"

    # Reset the parameters to avoid conflicts in the next iteration
    sed -i "s/$new_param/$old_param/g" params.yaml

    # Print current date and time
    date
done