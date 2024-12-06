from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "tuned_model300", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

from datasets import load_dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql", split = "test")

# alpaca_prompt = You MUST copy from above!
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Initialize the streamer
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

# Function to process each dataset entry and run inference
def generate_sql_from_entry(entry):
    # Format the input using the dataset's structure
    instruction = entry["sql_prompt"]  # Replace with your dataset's column name for instruction
    input_text = entry["sql_context"]         # Replace with your dataset's column name for input
    
    # Prepare the prompt
    prompt = alpaca_prompt.format(
        instruction,  # Instruction
        input_text,   # Input
        ""            # Leave output blank for generation
    )
    
    # Tokenize and prepare the input
    inputs = tokenizer(
        [prompt], return_tensors="pt", truncation=True, padding=True
    ).to("cuda")
    
    # Generate output
    generated = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
    
    # Return the generated SQL
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Loop over the dataset and run inference
results = []
count = 0
for entry in dataset:
    if entry['sql'][:6] != 'SELECT':
        continue
    if count > 500:
        break
    count += 1
    print(f"Processing entry {count}/{len(dataset)}...")
    generated_sql = generate_sql_from_entry(entry)
    result = [entry['sql_prompt'], entry['sql_context'], generated_sql, entry["sql"]] # predict, target
    results.append(result)
    print(entry["sql"])

# Optionally, save the results
import json
with open("outputs/generated_sql_results300.json", "w") as f:
    json.dump(results, f)


# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "What is the average explainability score of creative AI applications in 'Europe' and 'North America' in the 'creative_ai' table?", # instruction
#         "CREATE TABLE creative_ai (application_id INT, name TEXT, region TEXT, explainability_score FLOAT); INSERT INTO creative_ai (application_id, name, region, explainability_score) VALUES (1, 'ApplicationX', 'Europe', 0.87), (2, 'ApplicationY', 'North America', 0.91), (3, 'ApplicationZ', 'Europe', 0.84), (4, 'ApplicationAA', 'North America', 0.93), (5, 'ApplicationAB', 'Europe', 0.89);", # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")
# 
# text_streamer = TextStreamer(tokenizer)
# generated = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
# print(generated)
