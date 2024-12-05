import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

#################################
### Setup Quantization Config ###
#################################
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

#######################
### Load Base Model ###
#######################
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=quant_config,
    device_map={"":0},
)

####################
### Load Dataset ###
####################
train_dataset_name = "gretelai/synthetic_text_to_sql"
train_dataset = load_dataset(train_dataset_name, split="train")

#########################################
### Load LoRA Configurations for PEFT ###
#########################################
peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout= 0.1 ,
    r= 64 ,
    bias="none",
    task_type="CAUSAL_LM",
)

##############################
### Set Training Arguments ###
##############################
training_arguments = TrainingArguments(
    output_dir="./tuning_results",
    num_train_epochs= 1 ,
    per_device_train_batch_size= 4 ,
    gradient_accumulation_steps= 1 ,
    optim="paged_adamw_32bit",
    save_steps= 25 ,
    logging_steps= 25 ,
    learning_rate= 2e-4 ,
    weight_decay= 0.001 ,
    fp16=False, 
    bf16 =False,
    max_grad_norm= 0.3 ,
    max_steps=- 1 ,
    warmup_ratio= 0.03 ,
    group_by_length=True,
    lr_scheduler_type="constant"
)


##########################
### Set SFT Parameters ###
##########################
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    # dataset_text_field="text",

    # tokenizer=tokenizer,
    # max_seq_length=None,
    # packing=False,
)

#######################
### Fine-Tune Model ###
#######################
trainer.train()

# Save and push to hub
trainer.save_model(training_arguments.output_dir)
