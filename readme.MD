# Babysit A Language Model From Scratch: Interactive Language Learning by Trials and Demonstrations 

## File Structures
- `reward_model\`
    - `train_reward.py`: Python script to train the reward model. The reward model is a [Llama2-7B with regression head](https://huggingface.co/docs/transformers/en/model_doc/llama2#transformers.LlamaForSequenceClassification). The training is distributed via Accelerate FSDP. 
    - `run.sh`: Shell script to run the training of the reward model.
- `TnD\`
    - `EPPOTrainer.py`: Custom PPOTrainer class. Adapted from `trl` PPOTrainer class.
    - `trainer_util.py`: Training utility functions.
    - `run_TnD.py`: Main script to run training and evaluation for TnD models
    - `run_experiments.sh`: Shell script to replicate the experiments in the paper.

## Running Experiments
Before running any experiments, have the paths for **reward model**, **training set**, **evaluation set**, **word set**, **teacher model**, and **output directory** ready.

**NOTE: the training requires 2 GPUs**

### Main experiments - TnD on BabyLM and BookCorpus
- To run the main experiments on BabyLM and BookCorpus, change the `TRAIN_SET_PATH`, `EVAL_SET_PATH`, and `WORD_SET_PATH` in `run_experiments.sh` to the paths of the training set, evaluation set, and word set respectively.
- Then, change the path for the corresponding teacher and reward model in `run_experiments.sh`.

### Main experiments - CLM baseline on BabyLM and BookCorpus
- To run the CLM baseline, simply set the `clm_per_step` to any number greater than 10001 in `run_experiments.sh`.

### Main experiments - Ablation study
- To run "teacher's demostration only" training, set the `teacher_demo_only` flag to `True` in `run_experiments.sh`.
- To run "student's trial only" training, set both the `teacher_demo_only` and `use_ground_truth` flags to `False` in `run_experiments.sh`.

### Model distillation experiments
- Use the same `run_experiments.sh` for the main experiments - TnD on BabyLM and BookCorpus and CLM baseline on BabyLM and BookCorpus.
- Set `n_head` and `n_embd` to the number of heads and embedding size of the teacher model in `run_experiments.sh`. Numbers used in the paper are `n_head=12, n_embd=588`, `n_head=10, n_embd=360`, and `n_head=10, n_embd=250`.

### Masked Teacher experiments
- To run the masked teacher experiments, refraining the teacher model from generating certain tokens, set the `mask_type` flag to `mask` in `run_experiments.sh`.

### Double CLM experiments
- To run the double CLM experiments, set the `double_clm` flag to `True` in `run_experiments.sh`.
