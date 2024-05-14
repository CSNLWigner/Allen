#!/bin/bash

# OLD

# Create GEN3.txt
#> GEN3.txt

# TD search
./runcode.sh TD_param-search-analysis_stimresidual session $1 >> GEN3.txt

# sort
grep -A 18 'maximum value' GEN3.txt > TD_params.txt
#sessions=$(grep -o 'session [0-9]\+' TD_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot

# clear GEN3.txt
#> GEN3.txt

# BU search
#./runcode.sh BU_param-search_stimresidual session $1 >> GEN3.txt

# sort
#grep -A 18 'maximum value' GEN3.txt > BU_params.txt
#sessions=$(grep -o 'session [0-9]\+' BU_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot
