#!/bin/bash

# OLD

# TD search
./runcode.sh TD_param-search_GEN3 session $1 >> BU_GEN3.txt

# sort
grep -A 18 'maximum value' BU_GEN3.txt > BU_params.txt
#sessions=$(grep -o 'session [0-9]\+' TD_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot

# clear GEN3.txt
#> GEN3.txt

# BU search
#./runcode.sh BU_param-search_GEN3 session $sessions >> GEN3.txt

# sort
#grep -A 18 'maximum value' GEN3.txt > BU_params.txt
#sessions=$(grep -o 'session [0-9]\+' BU_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//) # vegere rakj apostrofot
