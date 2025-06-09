# fine_tuning.py
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
import pandas as pd

def fine_tune_model(model, tokenizer):
    # Check if conversations.csv exists and is not empty
    if not os.path.exists("conversations.csv") or os.path.getsize("conversations.csv") == 0:
        raise ValueError("conversations.csv is empty or missing. Please have some conversations with the bot before fine-tuning.")
    
    # Verify CSV format and size
    try:
        df = pd.read_csv("conversations.csv")
        if set(df.columns) != {"user_input", "bot_response"}:
            raise ValueError("conversations.csv must have exactly two columns: 'user_input' and 'bot_response'.")
        if len(df) < 10:
            raise ValueError(f"conversations.csv has only {len(df)} entries. At least 10 conversations are required for fine-tuning.")
    except Exception as e:
        raise ValueError(f"Failed to read conversations.csv: {str(e)}")
    
    # Load dataset from CSV
    dataset = load_dataset('csv', data_files='conversations.csv')
    
    # Tokenize the dataset
    def tokenize_function(examples):
        text = [f"{inp} {tokenizer.eos_token} {resp}" for inp, resp in zip(examples['user_input'], examples['bot_response'])]
        tokenized = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_attention_mask=True
        )
        return tokenized
    
    tokenized_dataset = dataset['train'].map(tokenize_function, batched=True, remove_columns=['user_input', 'bot_response'])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    return model