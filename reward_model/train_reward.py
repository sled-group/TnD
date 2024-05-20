# parse the generated text from the format prompts: <text> Generated test: <text>
from datasets import Dataset, load_dataset, load_from_disk
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
# import roberta
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from accelerate import Accelerator
import wandb
import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--regression_data', type=str, help='path to regression data')
parser.add_argument('--output_dir', type=str, help='path to save model')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
args = parser.parse_args()

data_path = args.regression_data
output_dir = args.output_dir
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr


accelerator = Accelerator(gradient_accumulation_steps=1)

if accelerator.is_local_main_process:
    run = wandb.init(project='FSDP_reward_model_llama2', name='BABYLM-100M')


def parse_generated_text(text, folder):
    try:
        text = text.split('Generated Text:')[1].strip()
    except:
        print(text, folder)
    return text
## Load modal and do inference
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




dataset = load_from_disk(data_path ,keep_in_memory=True)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id=tokenizer.eos_token_id
def collate_fn(batch):
    text, labels = [], []
    for item in batch:
        text.append(item['text'])
        labels.append(item['labels'])

    # Tokenize and pad input sequences
    encoding = tokenizer(
        text, 
        truncation=True,
        padding=True,
        max_length=130,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Return the processed data
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels, dtype=torch.float)  # Ensure labels are floats
    }

# split dataset into train and test
num_gpus = torch.cuda.device_count()  # Get the number of available GPUs
batch_size = batch_size # per device
train_val_dataset = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1, shuffle=True)
print('data prepared')



## load model
print("loading model")
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf", num_labels=1
)
model.config.pad_token_id = model.config.eos_token_id
model.gradient_checkpointing_enable()  # reduce number of stored activations
model = accelerator.prepare(model)


## Prepare optimizer and schedule (linear warmup and decay)
# Set the maximum gradient norm (for gradient clipping)
max_grad_norm = 1.0
# Initialize step count
step = 0
# Set gradient accumulation steps
grad_acc_steps = 1

optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6, weight_decay=0.0)
# Initialize the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=700,
    num_training_steps=len(train_dataloader) * 3
)

# prepare using accelerator
optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
    optimizer, train_dataloader, val_dataloader, scheduler
)

scalar = GradScaler()

stats = {}
model.train()
for i in range(num_epochs):
    for batch in tqdm(train_dataloader):
        with accelerator.accumulate(model):
            batch['labels'] = batch['labels'].log2()
            #print(batch['labels'].dtype)
            # auto cast
            output = model(**batch)
            loss = output.loss
            #print(loss.dtype)
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Print loss every 1000 steps
        # write to file every 10 steps
        if step % 1 == 0:
            stats['loss'] = loss.item()
            stats['lr'] = scheduler.get_last_lr()[0]
            stats['step'] = step
        
        #if accelerator.is_local_main_process:
        if (step + 1) % 1500 == 0:
            model.eval()
            with torch.no_grad():
                loss = []
                for val_batch in tqdm(val_dataloader):
                    val_batch['labels'] = val_batch['labels'].log2()
                    val_out = model(**val_batch)
                    loss.append(val_out.loss.sum().item())
            stats['val_loss'] = np.mean(loss)
            model.train()
        # wait for all processes to synchronize
        #accelerator.wait_for_everyone()

        #if local_rank == 0:  # Only save and log on the main process
        if accelerator.is_local_main_process:
            # Log metrics   
            wandb.log(stats)    
        # Increase step count
        step += 1
        
        #if (step + 1) % 1000 == 0:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"{output_dir}/epoch{i}",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=False,
    )
