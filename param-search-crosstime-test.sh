#!/bin/bash

# TD search
./runcode.sh TD_param-search_GEN3 session $1 >> GEN3.txt

# sort
grep -A 18 'maximum value' GEN3.txt > TD_params.txt
sessions=$(grep -o 'session [0-9]\+' TD_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//')

# clear GEN3.txt
> GEN3.txt

# BU search
./runcode.sh BU_param-search_GEN3 session $sessions >> GEN3.txt

# sort
grep -A 18 'maximum value' GEN3.txt > BU_params.txt
sessions=$(grep -o 'session [0-9]\+' BU_params.txt | sed 's/session \(.*\)/\1/' | tr '\n' ',' | sed 's/.$//')

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
cp cache/* cache/BU