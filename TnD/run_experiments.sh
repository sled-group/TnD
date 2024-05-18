#!/bin/bash

# Define global variables for the additional arguments
export REWARD_MODEL_PATH='/path/to/reward/model'
export TRAIN_SET_PATH='/path/to/train/set'
export EVAL_SET_PATH='/path/to/eval/set'
export WORD_SET_PATH='/path/to/word/set'
export CHILD_PATH='/path/to/child/model'
export TEACHER_PATH='/path/to/teacher/model'
export SAVE_PATH='/path/to/save/results'

# Run the Python script with the existing and new arguments
# NOTE that the training requires 2 GPUs
python run_TnD.py --ppo_per_step=1 \
                  --clm_per_step=3 \
                  --ppo_lr=2e-5 \
                  --reward_type='sub' \
                  --use_ground_truth='True' \
                  --mask_type='none' \
                  --n_head=12 \
                  --n_embd=768 \
                  --teacher_demo_only=False \
                  --double_clm=False \
                  --reward_model_path="$REWARD_MODEL_PATH" \
                  --train_set_path="$TRAIN_SET_PATH" \
                  --eval_set_path="$EVAL_SET_PATH" \
                  --word_set_path="$WORD_SET_PATH" \
                  --gpt2_path="$CHILD_PATH" \
                  --teacher_path="$TEACHER_PATH" \
                  --save_path="$SAVE_PATH"


