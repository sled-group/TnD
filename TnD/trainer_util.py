from typing import Any, Dict, Union
from torch import nn
from transformers import Trainer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_pt_utils import LabelSmoother
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import math
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers.training_args import TrainingArguments
import numpy as np
import random
from collections import defaultdict
import  pdb
import datasets
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.optim.lr_scheduler as lr_scheduler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationMixin
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed as dist
from transformers.generation.utils import *
from transformers import set_seed
import os
from datasets import Dataset, load_dataset, load_from_disk
from torch.nn import functional as F

def set_all_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed_value)  # Sets seed for transformers

# Set the seed value
seed_value = 42

# Call the function to freeze the seeds
set_all_seeds(seed_value)

def update_scheduler(step, PPO_steps, CLM_steps, CLM_first, overlay):
    total_steps = PPO_steps + CLM_steps
    mod_step = step % total_steps
    
    # Handle CLM_first
    if CLM_first:
        if mod_step < CLM_steps:
            return 'CLM' if overlay != 'PPO' else 'both'
        return 'PPO' if overlay != 'CLM' else 'both'
    else:
        if mod_step < PPO_steps:
            return 'PPO' if overlay != 'CLM' else 'both'
        return 'CLM' if overlay != 'PPO' else 'both'

class NLPEvaluator():
    def __init__(self, task_name):
        self.task_name = task_name
        # read and load all json files in both blimp_filtered and supplement_filtered folders
        self.blimp_filtered = self.read_json_files('blimp_filtered')
        self.supplement_filtered = self.read_json_files('supplement_filtered')
        self.all_datasets = {**self.blimp_filtered, **self.supplement_filtered}

    def read_json_files(self, folder):
        # read to huggingface dataset
        datasets = {}
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                datasets[filename[:-5]] = load_dataset('json', data_files=f"{folder}/{filename}")['train']
        return datasets

    def evaluate(self, model, tokenizer, device):
        # evaluate the model on all datasets
        results = {}
        total_acc = 0
        for dataset_name, dataset in self.all_datasets.items():
            #print(dataset)
            if dataset_name == 'turn_taking':
                dataset = dataset.remove_columns("other")
            print(f"Evaluating on {dataset_name}")
            # evaluate the model on the dataset
            result = self.evaluate_dataset(model, tokenizer, dataset, device)
            total_acc += result
            print(result)
            results[dataset_name] = result
        average_accuracy = total_acc / len(results)
        results['average_accuracy'] = average_accuracy
        return results
    
    def evaluate_dataset(self, model, tokenizer, dataset, device):
        # evaluate the model on the dataset
        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        for batch in tqdm(dataloader):
            sent_good = batch['sentence_good']
            sent_bad = batch['sentence_bad']
            # tokenize the sentences
            sent_good_tokenized = tokenizer(sent_good, return_tensors="pt").to(device)
            sent_bad_tokenized = tokenizer(sent_bad, return_tensors="pt").to(device)
            #print(sent_good_tokenized)
            #print(sent_bad_tokenized)

            # append eos token at the front to serve an empty context
            sent_good_tokenized['input_ids'] = torch.cat([torch.tensor([tokenizer.eos_token_id]).to(device), sent_good_tokenized['input_ids'][0]]).unsqueeze(0)
            sent_good_tokenized['attention_mask'] = torch.cat([torch.tensor([1]).to(device), sent_good_tokenized['attention_mask'][0]]).unsqueeze(0)
            sent_bad_tokenized['input_ids'] = torch.cat([torch.tensor([tokenizer.eos_token_id]).to(device), sent_bad_tokenized['input_ids'][0]]).unsqueeze(0)
            sent_bad_tokenized['attention_mask'] = torch.cat([torch.tensor([1]).to(device), sent_bad_tokenized['attention_mask'][0]]).unsqueeze(0)

            good_logits = F.log_softmax(model(**sent_good_tokenized)[0][:, :-1, :], dim=-1).cpu().squeeze(0)
            bad_logits = F.log_softmax(model(**sent_bad_tokenized)[0][:, :-1, :], dim=-1).cpu().squeeze(0)

            logits_at_good_token = torch.gather(good_logits, 1, sent_good_tokenized['input_ids'].cpu().squeeze(0)[1:].unsqueeze(-1)).squeeze(-1)
            logits_at_bad_token = torch.gather(bad_logits, 1, sent_bad_tokenized['input_ids'].cpu().squeeze(0)[1:].unsqueeze(-1)).squeeze(-1)
            logprob_good = torch.sum(logits_at_good_token).item()
            logprob_bad = torch.sum(logits_at_bad_token).item()

            if logprob_good > logprob_bad:
                correct += 1
            total += 1
        return correct / total


class InvailActionMaskedGenerationModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab = []

    # def forward(self, input_ids, **kwargs):
    #     #print(f"Shape of input_ids: {input_ids.shape}")
    #     return super(InvailActionMaskedGenerationModel, self).forward(input_ids, **kwargs)
        

    def update_input_vocab(self, vocab):
        self.vocab = vocab
        #print(self.vocab)
        #print(self)

    def update_model(self, model):
        # make a deep copy of the model's state_dict
        state_dict = copy.deepcopy(model.state_dict())
        self.load_state_dict(state_dict, strict=False)        

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        #print(self.lm_head.weight)
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            #============================================#
            # sample
            # Avoid generating unwanted tokens by setting prob to -1e11 for those tokens
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            #print(probs)
            for token_id in self.vocab:
                #print(token_id)
                probs[:, token_id] = 0
                probs /= probs.sum(dim=-1, keepdim=True)
            try:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                #print(next_tokens)
            except Exception as e:
                print(e, 'Set to eos token instead.')
                next_tokens = torch.tensor([50256]).to(input_ids.device) # eos token
                
            #============================================#
            


            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


# create a cosine learning rate scheduler
class CosineLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, steps_per_cycle, from_zero=False, last_epoch=-1):
        self.max_lr = max_lr
        self.steps_per_cycle = steps_per_cycle
        self.from_zero = from_zero
        super(CosineLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        A = self.max_lr / 2.0
        B = (2 * math.pi) / self.steps_per_cycle
        C = math.pi if self.from_zero else 0
        
        return [A * (math.cos(B * self.last_epoch - C) + 1) for _ in self.base_lrs]

class WarmupLinearSchedule(lr_scheduler._LRScheduler):
    """Linear warmup and then linear decay of learning rate."""
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch
        if step_num < self.warmup_steps:
            return [base_lr * (step_num / self.warmup_steps) for base_lr in self.base_lrs]
        return [base_lr * max(0, 1.0 - (step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                for base_lr in self.base_lrs]
    
class PPOLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, L, P=0.0, Q=0.0, D=0.0, R=0.0, last_epoch=-1):
        self.L = L
        self.P = 0.32
        self.Q = 0.0
        self.D = 1.25e-7
        self.R = 0.0
        self.current_A = None
        super(PPOLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.current_A is None:
            return [self.L for _ in self.optimizer.param_groups]
        
        new_lr = self.L
        if self.current_A > self.P * self.L:
            new_lr += self.D
        #elif self.current_A < self.Q * self.L:
        #    new_lr -= self.R

        self.L = new_lr
        return [new_lr for _ in self.optimizer.param_groups]

    def set_current_clm_lr(self, A):
        self.current_A = A

class AnnealScheduler():
    def __init__(self, T, D, N_CLM, N_PPO):
        self.T = T
        self.S = T
        self.D = D
        self.N_CLM = N_CLM
        self.N_PPO = N_PPO
        self.epoch = 0
        self.current_step = 0
    
    def get_update_policy(self):
        # Check if we're in the CLM-only epochs
        if self.epoch < self.N_CLM:
            policy = 'CLM'
        else:
            # We are in a PPO epoch
            # Check the current step to decide between 'CLM' and 'PPO'
            if self.current_step < self.S:
                policy = 'CLM'
            else:
                policy = 'PPO'
        
        # Update step and check if we need to move to the next epoch
        self.current_step += 1
        if self.current_step >= self.T:
            self.current_step = 0
            self.epoch += 1
            
            # Check if we need to decrease S for the next epoch
            if self.epoch >= self.N_CLM and (self.epoch - self.N_CLM) % self.N_PPO == 0:
                self.S = max(self.S - self.D, 0)  # Ensure S does not go below 0
        
        return policy

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

## Trainer like trainer
def train_model(accelerator, model, dataloader, evaldl, optimizer, scheduler, 
                num_epochs=1, grad_accum_steps=4, logger=None, scaling=1.0, ppo_grads=None, disable_vhead=False, ppo_loss=None):
    if accelerator==None:
        device = 'cuda'
    else:
        device = accelerator.device
    
    if disable_vhead:
        model.v_head.summary.weight.requires_grad = False
        model.v_head.summary.bias.requires_grad = False
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    epoch_loss = 0.0
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # Get loss and perform a backward pass
            loss = outputs.loss if type(outputs) is not tuple else outputs[1]
            loss = loss * scaling
            epoch_loss += loss.detach()
            loss.backward()

            # Perform gradient accumulation
            if (step + 1) % grad_accum_steps == 0 or step == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                gradient_info = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        gradient_info[name] = param.grad.clone().detach().flatten()

                mean_sim = None
                if ppo_grads is not None:
                    similarity = []
                    for name in gradient_info.keys():
                        sim = torch.nn.functional.cosine_similarity(gradient_info[name], ppo_grads[name], dim=-1).item()
                        similarity.append(sim)
                    mean_sim = np.mean(similarity)

                print(step)
                optimizer.step()
                optimizer.zero_grad()

    # re-enable vhead
    if disable_vhead:
        model.v_head.summary.weight.requires_grad = True
        model.v_head.summary.bias.requires_grad = True
    return epoch_loss.item() / grad_accum_steps, mean_sim

label_smoother = LabelSmoother()
class VHTrainer(Trainer):
    def __init__(self, *args, accelerator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_batch = None
        self.accelerator = accelerator

    # Enable this if using PPO trainer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = label_smoother(outputs, inputs["labels"], shift_labels=True)
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        self.current_batch = inputs
        return super().training_step(model, inputs)
    
    def create_optimizer_and_scheduler(self, num_training_steps):
        super().create_optimizer_and_scheduler(num_training_steps)

        checkpoint_step = 2 # or wherever you want to start

        if checkpoint_step < self.args.warmup_steps:
            print(self.args.warmup_steps)
            # Warmup phase
            lr_at_checkpoint = self.args.learning_rate * checkpoint_step / self.args.warmup_steps
            print(lr_at_checkpoint)
        else:
            # Linear decay phase
            lr_at_checkpoint = self.args.learning_rate * (1 - (checkpoint_step - self.args.warmup_steps) / (num_training_steps - self.args.warmup_steps))

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=checkpoint_step-1,
        )
        print(self.lr_scheduler.get_last_lr()[0])
    
class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, trainer, tokenizer, eval_dl, is_interactive=False, save=False, conversation=0, dialogue_history=None):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.is_interactive = is_interactive
        self.conversation = conversation
        self.dialogue_history = dialogue_history
        self.save = save
        self.sample_frequency = defaultdict(int)
        self.eval_dl = eval_dl

    def count_sample_frequency(self, inputs):
        tokenizer = self.tokenizer
        for batch in inputs:
            # convert to tokens
            if len(batch.shape) == 2:
                batch = batch[0]
            batch = tokenizer.decode(batch, skip_special_tokens=True)
            for token in batch.split():
                if token[-1] in ".,!?;])>":
                    self.sample_frequency[token[-1]] += 1
                    self.sample_frequency[token[:-1]] += 1
                else:
                    self.sample_frequency[token] += 1

    def on_step_end(self, args, state, control, **kwargs):
        if self.is_interactive:
            if self.save and state.global_step == 1:
                #print("HEREEEEEEEE")
                checkpoint_path = f"{self.output_dir}/checkpoint--conversation-{self.conversation}"
                # enable this to save the model
                kwargs['model'].save_pretrained(checkpoint_path)
                print(f"Saved checkpoint at step {state.global_step} in {checkpoint_path}")
                # TODO: also save the word frequences and dialogue frequencies
                #pdb.set_trace()
                input_ids = self.trainer.current_batch['input_ids']
                self.count_sample_frequency(input_ids)
                # save the sample frequency to the checkpoint path
                with open(f"{checkpoint_path}/sample_frequency.txt", "w") as f:
                    for token, freq in self.sample_frequency.items():
                        f.write(f"{token}\t{freq}\n")
                # print the dialogue history
                with open(f"{checkpoint_path}/dialogue_history.txt", "a") as f:
                    for dialogue in self.dialogue_history:
                        f.write(dialogue + "\n")       
        else:
            if save_condition(state.global_step):
                checkpoint_path = f"{self.output_dir}/checkpoint-{state.global_step}"
                kwargs['model'].save_pretrained(checkpoint_path)
                print(f"Saved checkpoint at step {state.global_step} in {checkpoint_path}")

def save_condition(this_iter_num):
    # uncomment this
    do_save = False
    if this_iter_num < 20:
        if this_iter_num % 2 == 0:
            do_save = True
    elif this_iter_num < 100:
        if this_iter_num % 5 == 0:
            do_save = True
    elif this_iter_num < 500:
        if this_iter_num % 10 == 0:
            do_save = True
    elif this_iter_num < 1000:
        if this_iter_num % 50 == 0:
            do_save = True
    elif this_iter_num < 10000:
        if this_iter_num % 500 == 0:
            do_save = True
    else:
        if this_iter_num % 10000 == 0:
            do_save = True
    return do_save

def eval_perplexity(model, tokenizer, input_ids, attention_mask):

    # Labels are the same as combined_input_ids but have -100 where the input id is a pad token
    labels = input_ids.clone().to('cuda')
    labels[input_ids == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids.to('cuda'), attention_mask=attention_mask.to('cuda'), labels=labels)
        loss = outputs[1]
    # Compute perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

def get_masked_words(mask_type, word_set):
    if mask_type == 'none':
        masked_words = []

    else:
    ## Words with last ref gen freqs BKPS words
        if word_set == 'BABYLM_TOP':
            masked_words = [1487,1642,1657,1621,2192,2835,3336,983,2589,4483,2084,2647,4398,2636,
                        2925,1318,1100,1210,1767,2540,2058,1690,2222,1466,3730,2626,1917,5156,
                        2479,3511,1597,1762,1494,1336,1551,3589,2832,3863,1285,15967] # BABYLM TOP
        elif word_set == 'BKPS_TOP':
            masked_words = [743,4838,1057,2652,3436,1745,3516,3375,8788,2431,966,2356,3612,1545,2354,1109,
                        2300,1103,1719,5770,6364,2642,5938,2626,3505,1573,7926,4978,1854,4692,2187,
                        1661,2035,3492,2506,1862,2077,17841,5055,779] # BKPS TOP

    ## Words with middle range ref gen freqs BKPS words
        elif word_set == 'BABYLM_CDI':
            masked_words = [ 3952, 14720,  4771,  3491,  8212, 33847,  6507,  7872,  3881, 22045,
            1702,  9686,  7779,  2366,  3211,  1382,  9280,  4929,  6576,  2029,
            5897,  7545,  3714,  1323,  3424,  3195, 10481,  9245,  5044,  2648,
        30860,  3654,  6729, 12607,  7586,  5445,  5465,  7815,  7765,  9875] # BABYLM CDI
        elif word_set == 'BKPS_CDI':
            masked_words = [618, 1560, 534, 1182, 1965, 640, 2993, 835, 460, 1165, 1807, 866, 922, 826, 1239, 606, 691, 1521, 787, 1986, 2227, 994, 1309, 656, 772, 281, 625, 477, 783, 910, 3809, 757, 467, 2900, 991, 422, 3420, 517, 2936, 584] #BKPS
    return masked_words

    
def find_subsequence(tensor_batch, subsequence):
    sub_len = len(subsequence)
    batch_size, length = tensor_batch.shape
    indices = []

    for batch_idx in range(batch_size):
        found = -1
        for i in range(length - sub_len + 1):
            if torch.all(tensor_batch[batch_idx, i:i+sub_len] == subsequence.to(tensor_batch.dtype)):
                found = i
                break
        indices.append(found)

    return indices
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])
    
def calculate_char_level_perplexity(tokenizer, model, context, word):
    '''
    Find the word in each batch of the context
    Then extract the logits of the characters in the word at each batch
    Return the average perplexity of the word across all batches
    '''

    context_sentence = [cc for cc, idx in context]
    context_indices = [idx for cc, idx in context]
    context_tokenized = tokenizer(context_sentence, padding=True, truncation=True, return_tensors="pt").to('cuda')
    word_tokenized = torch.tensor(tokenizer.encode(word)[:-1])

    context_tensor = context_tokenized['input_ids']

    char_surprisal = []
    char_probs = []
    char_ppl = []

    PER_DEVICE_BATCH_SIZE = 24  # Set this to the batch size you want per GPU
    num_gpus = torch.cuda.device_count()  # Get the number of available GPUs
    BATCH_SIZE = num_gpus * PER_DEVICE_BATCH_SIZE  # Total batch size

    dataset = CustomDataset(context_tokenized)
    # DataLoader will handle the batching
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Pass the context tensor through the model
    model.eval()
    all_logits = []  # List to store logits from each batch
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to('cuda') for k, v in batch.items()}  # Move to GPU
            outputs = model(**batch)
            logits = outputs[0][:, :-1]
            all_logits.append(logits.cpu())  # Move to CPU after processing
    all_logits = torch.cat(all_logits, dim=0)
    context_tensor = context_tensor.detach()

    # Find the starting index of the word in the context
    word_start_indices = context_indices#find_subsequence(context_tensor, word_tokenized)
    for batch_idx, word_start_idx in enumerate(word_start_indices):
        # Check if the word is found in the context
        if word_start_idx == -1:
            continue
        if word_start_idx > 0:
            word_start_idx -= 1
        word_logits = all_logits[batch_idx, word_start_idx:word_start_idx+len(word_tokenized), :]
        probs = torch.nn.Softmax(dim=-1)(word_logits)

        word_char_probs = torch.gather(probs, -1, word_tokenized.unsqueeze(1)).squeeze()

        char_probs.append(torch.prod(word_char_probs).item())
        char_surprisal.append(torch.sum(-1.0*torch.log2(word_char_probs)).item() / len(word_tokenized))
        char_ppl.append(2**(torch.sum(-1.0*torch.log2(word_char_probs)).item() / len(word_tokenized)))

    average_perplexity = np.mean(char_ppl)
    return average_perplexity

def construct_word_sentences_pair(corpus, words):
    '''
    corpus: List of sentences
    words: List of words
    
    '''
    # TODO: Implement this function
    bookcorpus = datasets.load_dataset('bookcorpus')
    wikitext = datasets.load_dataset('wikitext', 'wikitext-103-raw-v1')

    bookcorpus_text = bookcorpus['train']['text'] 
    wikitext_text = wikitext['test']['text']
    corpus = bookcorpus_text + wikitext_text

    holdout = random.sample(corpus, round(len(corpus)*0.001)) # TODO: find a better way to sample

    df = pd.read_csv('CDI_data/wordbank_item_data.csv')
    df = df['item_definition']
    CDI_words = df.tolist()
    # remove any words in brackets
    cleaned_CDI_words = []
    for word in CDI_words:
        tokens = word.split('(')
        token = tokens[0]
        tokens = word.split('/')
        if len(tokens) > 1:
            for token in tokens:
                cleaned_CDI_words.append(token)
        else:
            cleaned_CDI_words.append(token.strip('*').strip('!').strip('?').strip(' ').strip('\t'))
    cleaned_CDI_words = list(set(cleaned_CDI_words)) 

    def find_matching_sentences(sentences, target):
        for sentence in sentences:
            if ' '+target+' ' in sentence:
                yield sentence
    selected_sentences = {}
    for target in cleaned_CDI_words:
        matching = list(find_matching_sentences(holdout, target))
        selected = random.sample(matching, min(512, len(matching)))
        selected_sentences[target] = selected
    
    return selected_sentences

def evaluate_on_CDI_words(tokenizer, model, word_sentences_pair, path):
    '''
    word_sentences_pair: Dict of words and their sentences
    '''
    word_average_perplexities = {}
    for word in word_sentences_pair.keys():
        if len(word_sentences_pair[word]) > 0:
            word_average_perplexities[word] = calculate_char_level_perplexity(tokenizer, model, word_sentences_pair[word], word)
            print(f'Word: {word}, Perplexity: {word_average_perplexities[word]}')
        else:
            word_average_perplexities[word] = -1
            print(f'Word: {word}, Perplexity: {word_average_perplexities[word]}')
    # save the results to a file
    with open(path+'/ppl.txt', 'w') as f:
        for word, ppl in word_average_perplexities.items():
            f.write(f'{word}: {ppl}\n')


    