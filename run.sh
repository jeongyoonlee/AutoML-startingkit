#!/bin/bash

# Exit when any command fails.
set -e

# It requires two input arguments: the model and data folder names
if [ "$1" = "" ] || [ "$2" = "" ]; then
    echo "USAGE: ./run.sh [model folder] [data folder]"
    exit 1
fi

model_name="$(basename -- $1)"
model_dir=$1
data_dir="$2/"
output_dir="build/$model_name/"

echo "Model name: $model_name"
echo "Model directory: $1"
echo "Data directory: $data_dir"
echo "Output directory: $output_dir"

# If the data folder doesn't exist, exit.
if [ ! -d $data_dir ]; then
    echo "Data directory, $data_dir does not exist."
    exit 1
fi

# If the output folder doesn't exist, create it.
if [ ! -d $output_dir ]; then
    echo "Output directory, $output_dir does not exist. Creating $output_dir..."
    mkdir -p $output_dir;
fi

# Train the auto ML model.
echo "training..."
python3 pipeline/AutoML3_ingestion_program/ingestion.py $data_dir $output_dir $data_dir pipeline/AutoML3_ingestion_program $model_dir

# Predict for test data and score predictions.
echo "predicting..."
python3 pipeline/AutoML3_scoring_program/score.py "$data_dir/*/" $output_dir $output_dir
