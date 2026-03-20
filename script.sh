#!/bin/bash
EXPERIMENT_NAME=delfys75
FOLDER=./delfys75
export CUDA_VISIBLE_DEVICES=1

# Check if the folder exists; if not, create it
mkdir -p ./Results/"$EXPERIMENT_NAME"

# Remove files only if they exist
rm -f ./Results/"$EXPERIMENT_NAME"/*.png
rm -f ./Results/"$EXPERIMENT_NAME"/"$EXPERIMENT_NAME".log
rm -f ./Results/"$EXPERIMENT_NAME"/"$EXPERIMENT_NAME".csv

# Run with Python directly instead of apptainer
nohup python ./main.py --dt 3/60 --path "$FOLDER" --outfolder "$EXPERIMENT_NAME" >> ./Results/"$EXPERIMENT_NAME"/"$EXPERIMENT_NAME".log 2>&1 &