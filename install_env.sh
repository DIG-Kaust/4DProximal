#!/bin/bash


echo 'Creating prox4d environment'

# create conda env
conda env create -f environment.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate prox4d
conda env list
echo 'Created and activated environment:' $(which python)

echo 'Done!'
