# pre-trained.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)