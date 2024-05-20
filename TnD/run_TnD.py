import wandb
import pickle
import argparse
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import os
import random
import math
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from EPPOTrainer import EPPOTrainer
from trainer_util import (
    VHTrainer, 
    train_model, 
    InvailActionMaskedGenerationModel, 
    save_condition, 
    NLPEvaluator, 
    update_scheduler, 
    eval_perplexity,
    get_masked_words
)


def set_all_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed_value)  # Sets seed for transformers

sweep_config = wandb.config

#==================================================================================================#
# Training Arguments
#==================================================================================================#
# Enable args
parser = argparse.ArgumentParser(description="accepting arguments")

# Set paths
parser.add_argument("--reward_model_path", type=str, default='')
parser.add_argument("--train_set_path", type=str, default='')
parser.add_argument("--eval_set_path", type=str, default='')
parser.add_argument("--word_set_path", type=str, default='')
parser.add_argument("--gpt2_path", type=str, default='')
parser.add_argument("--teacher_path", type=str, default='')
parser.add_argument("--save_path", type=str, default='')

# set hyperparameters
parser.add_argument("--clm_per_step", type=int, default=3)
parser.add_argument("--ppo_per_step", type=int, default=1)
parser.add_argument("--overlay", type=str, default='CLM')
parser.add_argument("--ppo_lr", type=float, default=2e-5)
parser.add_argument("--use_ground_truth", type=str, default='True')
parser.add_argument("--use_binary_reward", action="store_true", default=False)
parser.add_argument("--ref_reward", action="store_true", default=False)
parser.add_argument("--ref_gen", action="store_true", default=True)
parser.add_argument("--reward_type", type=str, default='sub')
parser.add_argument("--mask_type", type=str, default='none')
parser.add_argument("--vh", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--word_set", type=str, default='BKPS_CDI') # BKPS_TOP, BABYLM_CDI, BABYLM_TOP
parser.add_argument("--harsh_punishment", type=str, default='False')

# set flag for the ablation study
parser.add_argument("--teacher_demo_only", type=str, default='False')

# set flag for the double CLM experiment
parser.add_argument("--double_clm", type=str, default='False')

# set hyperparameters for smaller student models
parser.add_argument("--n_head", type=int, default=12)
parser.add_argument("--n_embd", type=int, default=768)


#=================================#

PPO_EPOCH = 1#sweep_config.ppo_epoch
PPO_LR = 2e-5#sweep_config.ppo_lr
CKPT_START_NUMBER = 2#sweep_config.ckpt_start_number
SCALING = 1.0#sweep_config.scaling
PPL_COEF = 0.0
STEP_COEF = 1.0
ENTROPY_COEFF = 0.0#sweep_config.entropy_coeff
VF_COEF = 0.1#sweep_config.vf_coef
CLIPRANGE = 0.2#sweep_config.cliprange
CLIPRANGE_VALUE = 0.2#sweep_config.cliprange_value
PPO_LOSS_SCALE = 1.0#float(sweep_config.ppo_loss_scale)
#==================================#
USE_GROUND_TRUTH = True
#==================================#
#==================================#
USE_BINARY_REWARD = False
#==================================#
REF_REWARD = True
#==================================#
REF_GEN = True
REF_GEN_CKPT = 100000
GT_CKPTS_AHEAD = 0 # if this enabled, REF_GEN_CKPT will be ignored
#==================================#
TRAIN_VHEAD_IN_CLM = False
#==================================#
#PPL_REWARD = True
#==================================#

#==================================#
# if args, overwrite the default
if True:
    args = parser.parse_args()

    RM_path = args.reward_model_path
    gpt2_path = args.gpt2_path
    teacher_path = args.teacher_path

    word_set_path = args.word_set_path
    train_set_path = args.train_set_path
    eval_set_path = args.train_set_path
    save_path = args.save_path
    
    CLM_PER_STEP = args.clm_per_step
    PPO_PER_STEP = args.ppo_per_step
    OVERLAY = args.overlay
    CLM_FIRST = True
    PPO_LR = args.ppo_lr
    USE_GROUND_TRUTH = args.use_ground_truth
    if USE_GROUND_TRUTH == 'False':
        USE_GROUND_TRUTH = False
    else:
        USE_GROUND_TRUTH = True
    USE_BINARY_REWARD = args.use_binary_reward
    REF_REWARD = args.ref_reward
    REF_GEN = args.ref_gen
    REWARD_TYPE = args.reward_type
    if REWARD_TYPE == 'bin':
        USE_BINARY_REWARD = True
    VF_COEF = args.vh
    MASK_TYPE = args.mask_type
    SEED = args.seed
    WORD_SET = args.word_set
    HARSH_PUNISHMENT = args.harsh_punishment
    if HARSH_PUNISHMENT == 'False':
        HARSH_PUNISHMENT = False
    else:
        HARSH_PUNISHMENT = True
    TEACHER_DEMO_ONLY = args.teacher_demo_only
    DOUBLE_CLM = args.double_clm

    n_head = args.n_head
    n_embd = args.n_embd
    LR = 0
    
#==================================#
name = f'[{WORD_SET}_words]-MASK-{MASK_TYPE}-ppo-steps-{PPO_PER_STEP},-clm-steps-{CLM_PER_STEP},-ppo-lr-{PPO_LR}-overlay-{OVERLAY}-clm-first-{CLM_FIRST}-{REWARD_TYPE}-use_GT-{USE_GROUND_TRUTH}-default-PPO-no-threshold-{SEED}-HARSH_PUNISHMENT-{HARSH_PUNISHMENT}'
run = wandb.init(project='TnD', name=name)

masked_words = get_masked_words(MASK_TYPE, WORD_SET)

seed_value = SEED

set_all_seeds(seed_value)

ckpt_numbers = [i for i in range(0, 500001) if save_condition(i)]

def find_CDI_word_surprisal(model, tokenizer, sentence_dict):
    model.to('cuda:1')
    word_surprisal = {}
    for word in tqdm(sentence_dict.keys()):
        if isinstance(sentence_dict[word][0], str):
            sentences = sentence_dict[word]
        else:
            sentences = [pair[0] for pair in sentence_dict[word]]
        surprisals = []
        target_id = tokenizer(' '+word, return_tensors="pt")["input_ids"].to('cuda:1')[0, 0]
        batch_size = 32
        for i in range(0, len(sentences), batch_size):
            sentences_batch = sentences[i:i + batch_size]
            examples_tokenized = tokenizer(sentences_batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda:1')
            input_ids = examples_tokenized['input_ids'][0].tolist()
            labels = examples_tokenized.input_ids.clone().to('cuda:1')
            target_index = (labels==target_id)[:, 1:]
            mask = (examples_tokenized.input_ids != tokenizer.pad_token_id)[:, :-1]
            with torch.no_grad():
                output = model(**examples_tokenized, labels=labels)
                logits = output[0][:, :-1]
                #print(logits)
                mask_logits = logits[target_index, :]
                #print(mask_logits)
                probs = F.softmax(mask_logits, dim=-1)
                token_prob = probs[:, target_id]
                token_prob += 0.000001 # Smooth with (1e-6).
                surprisal = -1.0*torch.log2(token_prob)
                surprisals.extend(surprisal.tolist())
        word_surprisal[word] = np.mean(surprisals)
    model.to('cuda:0')
    return word_surprisal
        
def has_accumulated_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"No gradient for {name}")
                return False
            elif torch.all(param.grad == 0):
                print(f"Zero gradient for {name}")
                return False
    return True

### USE LLAMA-2-7B
reward_model = AutoModelForSequenceClassification.from_pretrained(
    RM_path, device_map='auto'
)
rm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
rm_tokenizer.pad_token = rm_tokenizer.eos_token

# model = GPT2LMHeadModel.from_pretrained(f'{gpt2_path}/checkpoint-{CKPT_START_NUMBER}').to('cuda')
gpt2_configurations = GPT2Config(n_embd=n_embd, n_head=n_head)
model = GPT2LMHeadModel(config=gpt2_configurations).to('cuda')
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

if REF_GEN:
    if GT_CKPTS_AHEAD > 0:
        ref_checkpoint_number = ckpt_numbers[ckpt_numbers.index(CKPT_START_NUMBER)+GT_CKPTS_AHEAD]
        ref_gen_model = InvailActionMaskedGenerationModel.from_pretrained(f'{teacher_path}/checkpoint-{ref_checkpoint_number}').to('cuda')
    else:
        ref_gen_model = InvailActionMaskedGenerationModel.from_pretrained(f'{teacher_path}/checkpoint-{REF_GEN_CKPT}').to('cuda')
    ref_gen_model.update_input_vocab(masked_words)
if REF_REWARD:
    # create a static copy of the initial model used for CLM update only
    clm_ref_model = GPT2LMHeadModel.from_pretrained(f'{teacher_path}/checkpoint-{CKPT_START_NUMBER}').to('cuda')

student_tokenizer = AutoTokenizer.from_pretrained('gpt2')
student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

### Load CDI words to find the perplexity of each word later
# load the sentences_dict to a file
with open(word_set_path, 'rb') as f:
    sentences_dict = pickle.load(f)
## sentences_dict is a dictionary with key as the word and value as the list of sentences containing the word
## we should first filter out the items whoes keys that have more than one token after tokenization
sentences_dict = {k:v for k, v in sentences_dict.items() if len(student_tokenizer.tokenize(' '+k)) == 1}

# load the already tokeinzed dataset
training_dataset = load_from_disk(train_set_path, keep_in_memory=True)
eval_ppl_dataset = load_from_disk(eval_set_path, keep_in_memory=True)

# filter out sentences that have less than 25 words for both datasets
print("train size: ", len(training_dataset))
print("eval size: ", len(eval_ppl_dataset))

# randomly select 2000 for eval
eval_ppl_dataset = eval_ppl_dataset.shuffle(seed=seed_value).select(range(2000))

def collate_fn(batch):
    # Sort the batch in the descending order
    batch.sort(key=lambda x: len(x['combined']), reverse=True)

    # Get sequences
    combined, prompts, continuations = [], [], []
    for item in batch:
        combined.append(torch.tensor(item['combined']))
        prompts.append(torch.tensor(item['prompt']))
        continuations.append(torch.tensor(item['continuation']))

    # Pad sequences
    combined = pad_sequence(combined, batch_first=True, padding_value=student_tokenizer.pad_token_id)
    prompts = pad_sequence(prompts, batch_first=True, padding_value=student_tokenizer.pad_token_id)
    continuations = pad_sequence(continuations, batch_first=True, padding_value=student_tokenizer.pad_token_id)

    # Create attention masks for combined
    attention_mask = (combined != student_tokenizer.pad_token_id).long()

    return combined, prompts, continuations, attention_mask

## Create dataloader
train_batch_size = 128
eval_batch_size = 128
eval_dataloader = torch.utils.data.DataLoader(eval_ppl_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

data_collator = DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False)

training_dataset = training_dataset.remove_columns(["prompt", "continuation"])
training_dataset = training_dataset.rename_column("combined", "input_ids")
training_args = TrainingArguments(         
        output_dir='',          # output directory
        per_device_train_batch_size=64,   # batch size per device during training
        save_strategy='no',
        logging_strategy='steps',
        logging_dir='logs',            # directory for storing logs
        warmup_steps=500,
        report_to='wandb',
        max_steps=50000,
        weight_decay=1e-6,                     # strength of weight decay 
        learning_rate=1e-4                    # starting learning rate
    )
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=training_dataset,                    # TODO: Define training dataset. Using the conversation history?
    data_collator=data_collator
)
dl = trainer.get_train_dataloader()

ppo_config = {'batch_size': 256 if USE_GROUND_TRUTH else 128,#BATCH_SIZE,
              'mini_batch_size': 16,#int(BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS),
              'gradient_accumulation_steps': 16 if USE_GROUND_TRUTH else 8,#GRADIENT_ACCUMULATION_STEPS,
              'cliprange': CLIPRANGE,
              'cliprange_value': CLIPRANGE_VALUE,
              'vf_coef': VF_COEF,
              'learning_rate': PPO_LR,
              'ppo_epochs': PPO_EPOCH,
              'ratio_threshold': 10000,#math.inf,
              'max_grad_norm': 1.0,
              'use_score_norm': True,
              }

ppo_config = PPOConfig(**ppo_config)

ppo_trainer = EPPOTrainer(ppo_config, 
                            model=model, 
                            ref_model=model, 
                            tokenizer=student_tokenizer, 
                            entropy_coeff=ENTROPY_COEFF, 
                            loss_scaling=PPO_LOSS_SCALE, 
                            logger=wandb)

# create a optimizer
CLM_LR = 1e-4
optimizer = AdamW(model.parameters(), lr=CLM_LR, eps=1e-8)
## old linear scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=8000,  # The number of warmup steps
    num_training_steps=500000  # The total number of training steps
)
scheduler.last_epoch = CKPT_START_NUMBER - 1
scheduler.step()

if REF_REWARD:
    # make a deep copy of both scheduler and optimizer
    clm_ref_optimizer = AdamW(clm_ref_model.parameters(), lr=CLM_LR, eps=1e-8)#, weight_decay=1e-6)
    clm_ref_scheduler = get_linear_schedule_with_warmup(
        clm_ref_optimizer, 
        num_warmup_steps=8000,  # The number of warmup steps
        num_training_steps=500000  # The total number of training steps
    )
    clm_ref_scheduler.last_epoch = CKPT_START_NUMBER - 1
    clm_ref_scheduler.step()

MAX_STEPS = 3
step = CKPT_START_NUMBER
        
# Maintain a vocabulary of the tokens from the training dataset
VOCAB = set()
clm_dataset_training_freq = {}
ref_gen_freq = {}
model_gen_freq = {}
ref_reward_freq = {}
for i in range(MAX_STEPS):
    for batch in tqdm(dl):
        # = anneal_scheduler.get_update_policy()

        policy = update_scheduler(step-CKPT_START_NUMBER, PPO_PER_STEP, CLM_PER_STEP, CLM_FIRST, OVERLAY)

        # update the VOCAB by filling new unique tokens from the training dataset
        VOCAB.update(set(batch['input_ids'].flatten().tolist()))
        ppo_trainer.update_vocab(VOCAB)

        prompt = batch['input_ids'][:, :5]

        continuation = batch['input_ids'][:, 5:]
        step_stats = {}
        step_stats['step'] = step
        if policy == 'PPO':
            step_stats['current_policy'] = 1
        elif policy == 'both':
            step_stats['current_policy'] = 0
        elif policy == 'CLM':
            step_stats['current_policy'] = -1

        if save_condition(step):
            # evaluate the CDI perplexity
            perplexities = find_CDI_word_surprisal(model, student_tokenizer, sentences_dict)
            step_stats.update(perplexities)

            # save model
            model.save_pretrained(f'{save_path}/{name}/checkpoint-{step}')
            # save frequency files to csv
            with open(f'{save_path}/{name}/checkpoint-{step}/clm_dataset_training_freq_{step}.csv', 'w') as f:
                for key in clm_dataset_training_freq.keys():
                    f.write("%s,%s\n"%(key,clm_dataset_training_freq[key]))
            with open(f'{save_path}/{name}/checkpoint-{step}/ref_gen_freq_{step}.csv', 'w') as f:
                for key in ref_gen_freq.keys():
                    f.write("%s,%s\n"%(key,ref_gen_freq[key]))
            with open(f'{save_path}/{name}/checkpoint-{step}/model_gen_freq_{step}.csv', 'w') as f:
                for key in model_gen_freq.keys():
                    f.write("%s,%s\n"%(key,model_gen_freq[key]))
            with open(f'{save_path}/{name}/checkpoint-{step}/ref_reward_freq_{step}.csv', 'w') as f:
                for key in ref_reward_freq.keys():
                    f.write("%s,%s\n"%(key,ref_reward_freq[key]))
            # model saved
            print(f"Model saved at step {step}")

            ppls = []
            ppls_old = []
            eval_ppls = []
            with torch.no_grad():
                for eval_combined, eval_prompt, eval_continuation, attenstion_mask in tqdm(eval_dataloader):
                    eval_ppl = eval_perplexity(model, student_tokenizer, eval_combined, attenstion_mask)
                    eval_ppls.append(eval_ppl)
            step_stats['ppl_over_evalset'] = np.mean(eval_ppls)
        if policy == 'PPO' or policy == 'both':
            # First step: prompt the model with the prompt and collect the ouputs
            generated_sentences_raw = model.generate(
                    input_ids=prompt.to('cuda'),
                    do_sample=True,
                    max_length=128-prompt.size(1)-1,
                    top_k=20,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=student_tokenizer.pad_token_id
                )
            for sentence in generated_sentences_raw:
                for token_id in sentence:
                    if token_id == student_tokenizer.pad_token_id:
                        break
                    token_id = token_id.item()
                    if token_id in model_gen_freq:
                        model_gen_freq[token_id] += 1
                    else:
                        model_gen_freq[token_id] = 1
            
            if REF_GEN:
                ref_generation = ref_gen_model.generate(
                    input_ids=prompt.to('cuda'),
                    do_sample=True,
                    max_length=128-prompt.size(1)-1,
                    top_k=20,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=student_tokenizer.pad_token_id
                )
                for sentence in ref_generation:
                    for token_id in sentence:
                        if token_id == student_tokenizer.pad_token_id:
                            break
                        token_id = token_id.item()
                        if token_id in ref_gen_freq:
                            ref_gen_freq[token_id] += 1
                        else:
                            ref_gen_freq[token_id] = 1
                ref_generation_text = student_tokenizer.batch_decode(ref_generation, skip_special_tokens=True)
            if REF_REWARD:
                # generate reference sentences
                generated_sentences_raw_ref = clm_ref_model.generate(
                        input_ids=prompt.to('cuda'),
                        do_sample=True,
                        max_length=128-prompt.size(1)-1,
                        top_k=20,
                        top_p=0.95,
                        num_return_sequences=1,
                        pad_token_id=student_tokenizer.pad_token_id
                    )
                for sentence in generated_sentences_raw_ref:
                    for token_id in sentence:
                        if token_id == student_tokenizer.pad_token_id:
                            break
                        token_id = token_id.item()
                        if token_id in ref_reward_freq:
                            ref_reward_freq[token_id] += 1
                        else:
                            ref_reward_freq[token_id] = 1
                generated_sentences_ref = student_tokenizer.batch_decode(generated_sentences_raw_ref, skip_special_tokens=True)
            generated_sentences = student_tokenizer.batch_decode(generated_sentences_raw, skip_special_tokens=True)
            
        if (policy == 'PPO' or policy == 'both') and DOUBLE_CLM == 'False':
            REWARD_INPUT_BATCH = 16
            reward = None
            for ib_idx in range(0, len(generated_sentences), REWARD_INPUT_BATCH):
                generated_sentences_batch = generated_sentences[ib_idx: ib_idx+REWARD_INPUT_BATCH]
                if TEACHER_DEMO_ONLY:
                    generated_sentences_batch = ref_generation_text[ib_idx: ib_idx+REWARD_INPUT_BATCH]
                reward_tokenized = rm_tokenizer(generated_sentences_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)

                reward_model.eval()
                with torch.no_grad():
                    output = reward_model(input_ids=reward_tokenized.input_ids.to('cuda'), attention_mask=reward_tokenized.attention_mask.to('cuda'))
                    if reward == None:
                        reward = output.logits.detach().cpu()
                    else:
                        reward = torch.cat((reward, output.logits.detach().cpu()), dim=0)
            if REF_REWARD:
                # get the reference reward
                reward_ref = None
                for ib_idx in range(0, len(generated_sentences_ref), REWARD_INPUT_BATCH):
                    generated_sentences_batch = generated_sentences_ref[ib_idx: ib_idx+REWARD_INPUT_BATCH]
                    reward_tokenized = rm_tokenizer(generated_sentences_batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    reward_model.eval()
                    with torch.no_grad():
                        output = reward_model(input_ids=reward_tokenized.input_ids.to('cuda'), attention_mask=reward_tokenized.attention_mask.to('cuda'))
                        if reward_ref == None:
                            reward_ref = output.logits.detach().cpu()
                        else:
                            reward_ref = torch.cat((reward_ref, output.logits.detach().cpu()), dim=0)
                reward = reward.squeeze(-1)
                log_step = reward_ref.squeeze(-1)
            else:
                log_step = math.log2(step)
                
            ground_truth_reward = torch.tensor([math.log2(REF_GEN_CKPT) for i in range(train_batch_size)])#.bfloat16()
            if REWARD_TYPE == 'sub':
                ground_truth_score = (ground_truth_reward - log_step) * SCALING
            elif REWARD_TYPE == 'div':
                ground_truth_score = (ground_truth_reward / log_step) * SCALING
            else:
                ground_truth_score = ground_truth_reward * SCALING
            # set all negative scores to -1 and all positive scores to 1
            if USE_BINARY_REWARD:
                ground_truth_score = (ground_truth_reward - log_step) * SCALING
                ground_truth_score = torch.where(ground_truth_score < 0, torch.tensor(-1.0), ground_truth_reward)
            ground_truth_score_list = [i for i in ground_truth_score.to('cuda')]

            step_stats['avergae reward'] = reward.mean().item()
            if REF_REWARD:
                step_stats['average ref reward'] = reward_ref.mean().item()
            
            if REWARD_TYPE == 'sub':
                score = (reward - log_step) * SCALING
            elif REWARD_TYPE == 'div':
                score = (reward / log_step) * SCALING
            else:
                score = reward * SCALING
            if USE_BINARY_REWARD:
                score = (reward - log_step) * SCALING
                # set all negative scores to -1 and all positive scores to 1
                score = torch.where(score < 0, torch.tensor(-1.0), reward)
            
            step_stats['average score'] = score.mean().item()
            if step == 10000:
                wandb.log({"final_score": score.mean().item() / SCALING})
            
            score_tensor_list = [i for i in score.to('cuda')]
            prompt_tensor_list = [i for i in prompt.to('cuda')]
            continuation_tensor_list = [i for i in generated_sentences_raw.to('cuda')]

            if USE_GROUND_TRUTH:
                score_tensor_list.extend(ground_truth_score_list)
                prompt_tensor_list.extend([i for i in prompt.to('cuda')])
                if REF_GEN:
                    continuation_tensor_list.extend([i for i in ref_generation.to('cuda')])
                else:
                    ground_truth_tensor_list = [i for i in continuation.to('cuda')]
                    continuation_tensor_list.extend(ground_truth_tensor_list)
            
            if TEACHER_DEMO_ONLY:
                continuation_tensor_list = [i for i in ref_generation.to('cuda')]

            # Fourth step: run PPO to update the policy
            if policy == 'PPO' or policy == 'both':
                ppo_trainer.set_vloss_only(False)
                stats, ppo_loss = ppo_trainer.step(prompt_tensor_list, continuation_tensor_list, score_tensor_list) # TODO: check if this is correct

                step_stats['KL'] = stats['objective/kl']
                step_stats['KL penalty'] = stats['ppo/mean_non_score_reward']
                step_stats['mean_scores'] = stats['ppo/mean_scores']
                #step_stats['ppo_loss'] = ppo_loss
                step_stats['ppo_loss_total'] = stats['ppo/loss/total']
                step_stats['ppo_loss_policy'] = stats['ppo/loss/policy']
                step_stats['ppo_loss_value'] = stats['ppo/loss/value']
                step_stats['ppo_lr'] = stats['ppo/learning_rate']
                #step_stats['update_policy'] = 1
                step_stats['pg_clipfrac'] = stats['ppo/policy/clipfrac']
                step_stats['vf_clipfrac'] = stats['ppo/val/clipfrac']
                #step_stats['advantage'] = stats['ppo/policy/advantages']
                #step_stats['advantage_mean'] = stats['ppo/policy/advantages_mean']
                step_stats['vpred'] = stats['ppo/val/vpred']
                #step_stats['ratio'] = stats['ppo/policy/ratio']
            
        # Fifth step: update with CLM
        #else:
        if LR > -1.0:
            if policy == 'CLM' or policy == 'both':
                train_set = Dataset.from_dict(batch)
            if policy == 'PPO' or policy == 'both':
                # use generated sentences and ref generation for clm
                # first, pad the sentence generation and ref generation to the same length
                ref_gen_length = ref_generation.shape[1]
                generated_sentences_raw_length = generated_sentences_raw.shape[1]
                if ref_gen_length > generated_sentences_raw_length:
                    generated_sentences_raw = torch.cat((generated_sentences_raw, torch.ones((generated_sentences_raw.shape[0], ref_gen_length - generated_sentences_raw_length), dtype=torch.long) * student_tokenizer.pad_token_id), dim=-1)
                elif ref_gen_length < generated_sentences_raw_length:
                    ref_generation = torch.cat((ref_generation, torch.ones((ref_generation.shape[0], generated_sentences_raw_length - ref_gen_length), dtype=torch.long) * student_tokenizer.pad_token_id), dim=-1)

                all_input_ids = torch.cat((generated_sentences_raw, ref_generation), dim=0)
                train_set = Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": (all_input_ids != student_tokenizer.pad_token_id)})
            # use ref_gen for clm
            train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=data_collator)

            if policy == 'CLM' or policy == 'both':
                for sentence in batch['input_ids']:
                    for token_id in sentence:
                        if token_id == student_tokenizer.pad_token_id:
                            break
                        token_id = token_id.item()
                        if token_id in clm_dataset_training_freq:
                            clm_dataset_training_freq[token_id] += 1
                        else:
                            clm_dataset_training_freq[token_id] = 1
                train_loss, mean_sim = train_model(None, model, train_dl, None, optimizer, scheduler, logger=wandb, ppo_grads=ppo_trainer.get_grads(), disable_vhead=True)
                step_stats['train_loss'] = train_loss
                step_stats['mean_sim'] = mean_sim

            if policy == 'PPO' or policy == 'both':
                for sentence in all_input_ids:
                    for token_id in sentence:
                        if token_id == student_tokenizer.pad_token_id:
                            break
                        token_id = token_id.item()
                        if token_id in clm_dataset_training_freq:
                            clm_dataset_training_freq[token_id] += 1
                        else:
                            clm_dataset_training_freq[token_id] = 1
                #if (step - CKPT_START_NUMBER) % CLM_PER_STEP == 0:
                train_loss, mean_sim = train_model(None, model, train_dl, None, optimizer, scheduler, logger=wandb, ppo_grads=ppo_trainer.get_grads(), disable_vhead=True, grad_accum_steps=8)
                step_stats['train_loss'] = train_loss
                step_stats['mean_sim'] = mean_sim

            if REF_REWARD:
                ref_train_loss, _ = train_model(None, clm_ref_model, train_dl, None, clm_ref_optimizer, clm_ref_scheduler, logger=wandb)
                step_stats['ref_train_loss'] = ref_train_loss
            step_stats['CLM_learning_rate'] = optimizer.param_groups[0]['lr']
            step_stats['update_policy'] = -1
        # check if there are nan or inf in step_stats, if any, change them to 0
        for key in step_stats.keys():
            try:
                if math.isnan(step_stats[key]) or math.isinf(step_stats[key]):
                    step_stats[key] = 0.0
            except:
                pass
        wandb.log(step_stats)
        scheduler.step()
        clm_ref_scheduler.step()
        step += 1