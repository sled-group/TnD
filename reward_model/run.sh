#!/bin/bash

# Define global variables for the additional arguments
export REGRESSION_DATA_PATH = '/path/to/regression/data'
export OUTPUT_DIR = '/path/to/reward/model'


accelerate launch --config_file config.yaml FSDP_reward_model_llama2.py \
                    --regression_data=$REGRESSION_DATA_PATH \
                    --output_dir=$OUTPUT_DIR \
                    --batch_size=16 \
                    --num_epochs=3 \
                    --learning_rate=3e-5 \