# Databricks notebook source
# MAGIC %md
# MAGIC #### Loading and Updating a GPT-2 Large Model from the Custom Checkpoint

# COMMAND ----------

# MAGIC %pip install trl accelerate peft --upgrade mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#The model https://huggingface.co/keithxanderson/GPT2_124M_PARAM/tree/main
#import the necessary libraries 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from huggingface_hub import login, hf_hub_download
from trl import SFTTrainer
import mlflow


# COMMAND ----------

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# COMMAND ----------

repo_id = "AviSoori1x/gpt2_755_amd"
filename = "ckpt.pt"

# Download the file
file_path = hf_hub_download(repo_id, filename)

# COMMAND ----------

def load_checkpoint_to_hf_model(ckpt_path, model_name="gpt2-large"):
    # Load the state dictionary from the checkpoint file
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # Load the Hugging Face GPT-2 model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Map the state dictionary keys if necessary
    # Here, you need to ensure that the keys in the state_dict align with the Hugging Face model's keys
    
    model.load_state_dict(state_dict, strict=False)
    
    return model

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# COMMAND ----------

model = load_checkpoint_to_hf_model(file_path, model_name="gpt2-large")


# COMMAND ----------

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# COMMAND ----------

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.1, top_k=10):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text using the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example prompt
prompt = "here's a fun story:"

# Generate text
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform supervised finetuning with LoRA
# MAGIC

# COMMAND ----------

from datasets import load_dataset
from trl import SFTTrainer


# COMMAND ----------

dataset = load_dataset("imdb", split="train")

# COMMAND ----------

# Load the model we saved before
# Load the model from the checkpoint
model

# COMMAND ----------

#The tokenizer is the same 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# COMMAND ----------

#https://huggingface.co/docs/trl/en/sft_trainer#training-adapters
#train the adapter instead

# COMMAND ----------

from peft import LoraConfig

# COMMAND ----------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# COMMAND ----------

#enable logging system metrics
mlflow.enable_system_metrics_logging()

# COMMAND ----------

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    peft_config=peft_config

)
with mlflow.start_run():
    trainer.train()

# COMMAND ----------

trainer.save_model("/Volumes/DAIS_2024_Avi/default/gpt2_large_adapter")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading and merging the adapter to the pretrained model

# COMMAND ----------

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# COMMAND ----------

#Load the checkpoint as before
repo_id = "AviSoori1x/gpt2_755_amd"
filename = "ckpt.pt"

# Download the file
file_path = hf_hub_download(repo_id, filename)

# COMMAND ----------

def load_checkpoint_to_hf_model(ckpt_path, model_name="gpt2-large"):
    # Load the state dictionary from the checkpoint file
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # Load the Hugging Face GPT-2 model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Map the state dictionary keys if necessary
    # Here, you need to ensure that the keys in the state_dict align with the Hugging Face model's keys
    
    model.load_state_dict(state_dict, strict=False)
    
    return model

# COMMAND ----------

model = load_checkpoint_to_hf_model(file_path, model_name="gpt2-large")


# COMMAND ----------

adapter_path = "/Volumes/DAIS_2024_Avi/default/gpt2_large_adapter"
from peft import PeftModel, PeftConfig


# COMMAND ----------

config = PeftConfig.from_pretrained(adapter_path)
config.base_model_name_or_path

# COMMAND ----------

peft_model = PeftModel.from_pretrained(model, adapter_path)
print(peft_model)

# COMMAND ----------

peft_model.to("cuda")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merging the layers and running tests
# MAGIC

# COMMAND ----------

merged_model = peft_model.merge_and_unload()


# COMMAND ----------

prompt = """
Movie Review: "Eternal Echoes": "Eternal Echoes" is a visual masterpiece that
"""

# COMMAND ----------

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

generation_output = merged_model.generate(
    input_ids=input_ids, max_new_tokens=30
)
print(tokenizer.decode(generation_output[0]))

# COMMAND ----------

type(merged_model)

# COMMAND ----------

merged_model_path = "/Volumes/dais_2024_avi/default/gpt2_large_finetuned"

# COMMAND ----------

merged_model.save_pretrained(merged_model_path)

# COMMAND ----------

#Push model to the hub/ read directly from volume
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(merged_model_path)
tokenizer = GPT2Tokenizer.from_pretrained(merged_model_path)

from huggingface_hub import HfApi

api = HfApi()
repo_name = "gpt2-large-finetuned"
api.create_repo(repo_name, exist_ok=True)


# Push to Hugging Face Hub
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log model directly from Hugging Face Hub

# COMMAND ----------

# Databricks notebook source
%pip install --upgrade transformers
%pip install --upgrade accelerate
%pip install tf-keras
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
import numpy as np
import pandas as pd
from transformers import pipeline, set_seed

# COMMAND ----------

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
save_directory = "/Volumes/dais_2024_avi/default/gpt2_large_finetuned"
tokenizer.save_pretrained(save_directory)

# COMMAND ----------

task = 'text-generation'

text_generation_pipeline = pipeline(task, model='AviSoori1x/gpt2-large-finetuned', device= 0)

# COMMAND ----------

# inference configuration for the model
inference_config = {
    "max_length": 100,
    "temperature": 0.4
}


# COMMAND ----------

# schema for the model
input_example = "Movie Review: 'Eternal Echoes': 'Eternal Echoes' is a visual masterpiece that"
output = generate_signature_output(text_generation_pipeline, input_example)
signature = infer_signature(input_example, output)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=text_generation_pipeline,
        artifact_path="my_review_generator",
        inference_config=inference_config,
        registered_model_name='gpt2-large-finetuned',
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)
my_sentence_generator.predict(pd.DataFrame(["Movie Review: 'Terminator': 'Terminator' is an action packed "]))
