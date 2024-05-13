#!/bin/bash

# Get the param value list (Internal Field Separator is comma, then read by read only into array named XY from the parameter
IFS=',' read -r -a sessions <<< "$1"

run_function () {

    # Print the informations
    echo -e "\nSession $2"

    # Add the param key
    new_param="session: $2"
    old_param="session: 0"

    # Modify the parameters in params.yaml
    sed -i "s/$old_param/$new_param/g" params.yaml

    # Run code
    dvc repro > log_cache.txt

    # Exit in case of dvc error
    if [[ $? -ne 0 ]]; then
	echo "ERROR: dvc was not successfull"
	exit 1
    fi

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

    # PARAM-SEARCH
    # copy param-search
    cp pipelines/param-search/params.yaml params.yaml
    cp pipelines/param-search/dvc.yaml dvc.yaml

    # TD search
    sed -i "s/predictor: '\(.*\)'/predictor: 'VISl'/" params.yaml
    sed -i "s/target: '\(.*\)'/target: 'VISp'/" params.yaml
    run_function "TD_param-search_GEN3" $session

    # continue if no maximum value
    if ! grep -q 'maximum value' log_cache.txt; then
	echo "WARNING: All top-down model failed"
	continue
    fi

    # get TD variables from log cache
    TD_cv=$(sed -n 's/cv=\(.*\)/\1/' < log_cache.txt) # add '
    TD_rank=$(sed -n 's/rank=\(.*\)/\1/' < log_cache.txt) # add '
    echo "cv is $TD_cv"
    echo "rank is $TD_rank"

    # BU search
    sed -i "s/predictor: '\(.*\)'/predictor: 'VISp'/" params.yaml
    sed -i "s/target: '\(.*\)'/target: 'VISl'/" params.yaml
    run_function "BU_param-search_GEN3" $session

    # continue if no maximum value
    if ! grep -q 'maximum value' log_cache.txt; then
	echo "WARNING: All bottom-up model failed"
	continue
    fi

    # get BU variables from log cache
    BU_cv=$(sed -n 's/cv=\([[:alnum:]]*\)/\1/' < log_cache.txt) # add '
    BU_rank=$(sed -n 's/rank=\([[:alnum:]]*\)/\1/' < log_cache.txt) # add '
    echo "cv is $BU_cv"
    echo "rank is $BU_rank"

    # print into crosstime params file
    cat >>pipelines/crosstime/params.yaml <<EOL
  ${session}:
    top-down:
      cv: ${TD_cv}
      rank: ${TD_rank}
    bottom-up:
      cv: ${BU_cv}
      rank: ${BU_rank}
EOL

    # CROSSTIME
    # copy crosstime
    cp pipelines/crosstime/params.yaml params.yaml
    cp pipelines/crosstime/dvc.yaml dvc.yaml

    # TD crosstime
    sed -i "s/predictor: '\(.*\)'/predictor: 'VISl'/" params.yaml
    sed -i "s/target: '\(.*\)'/target: 'VISp'/" params.yaml
    run_function "TD_cross-time_GEN3" $session

    # BU crosstime
    sed -i "s/predictor: '\(.*\)'/predictor: 'VISp'/" params.yaml
    sed -i "s/target: '\(.*\)'/target: 'VISl'/" params.yaml
    run_function "BU_cross-time_GEN3" $session

done

# OLD

# TD search
#./runcode.sh TD_param-search_GEN3 session $1 >> GEN3.txt

# sort
#grep -A 18 'maximum value' GEN3.txt > TD_params.txt
#sessions=$(grep -o 'session [0-9]\+' TD_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot

# clear GEN3.txt
#> GEN3.txt

# BU search
#./runcode.sh BU_param-search_GEN3 session $sessions >> GEN3.txt

# sort
#grep -A 18 'maximum value' GEN3.txt > BU_params.txt
#sessions=$(grep -o 'session [0-9]\+' BU_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot
