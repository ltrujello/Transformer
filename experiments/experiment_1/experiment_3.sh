#!/bin/bash

# NUM_EPOCHS = 20
# NUM_BATCHES = 600
# BATCH_SIZE = 50
# PRINT_EVERY = 30
# LEARNING_RATE = 1
# MODEL_DIR = "experiment_1"


NUM_EPOCHS=20
NUM_BATCHES=600
BATCH_SIZE=100
LEARNING_RATE=1
MODEL_DIR="models/experiment_3"
NUM_LAYERS=6
NUM_HEADS=8

for ((head = 0; head < $NUM_HEADS; head++)); do
    python src/transformer/train.py \
        --num-epochs $NUM_EPOCHS \
        --num-batches $NUM_BATCHES \
        --batch-size $BATCH_SIZE \
        --print-every 30 \
        --log-level INFO \
        --save-model-run true \
        --learning-rate $LEARNING_RATE \
        --model-root $MODEL_DIR \
        --exp-layer 2 \
        --exp-head $head \
        --run-experiment true \
        --message "This is a run with the full dataset and $NUM_EPOCHS epochs. In this test run, I am guiding the attention head $head in layer 2." | colorlog
done
