# fine_tuning.py
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
import os
import pandas as pd

def fine_tune_model(model, tokenizer, use_dailydialog=False):
    # Load user conversations
    user_dataset = None
    if os.path.exists("conversations.csv") and os.path.getsize("conversations.csv") > 0:
        try:
            df = pd.read_csv("conversations.csv")
            if set(df.columns) != {"user_input", "bot_response"}:
                raise ValueError("conversations.csv must have exactly two columns: 'user_input' and 'bot_response'.")
            if len(df) < 10 and not use_dailydialog:
                raise ValueError(f"conversations.csv has only {len(df)} entries. At least 10 conversations are required for fine-tuning.")
        except Exception as e:
            raise ValueError(f"Failed to read conversations.csv: {str(e)}")
        user_dataset = load_dataset('csv', data_files='conversations.csv')['train']

    # Optionally load DailyDialog and convert to the same format
    if use_dailydialog:
        dd = load_dataset("daily_dialog")
        def dd_to_pairs(example):
            # Convert DailyDialog format to user_input/bot_response pairs
            pairs = []
            utterances = example['dialog']
            for i in range(len(utterances) - 1):
                pairs.append({'user_input': utterances[i], 'bot_response': utterances[i + 1]})
            return pairs
        dd_pairs = dd['train'].map(dd_to_pairs, batched=True, remove_columns=dd['train'].column_names)
        # Flatten the list of pairs
        dd_pairs = dd_pairs.flatten_indices()
        if user_dataset:
            dataset = concatenate_datasets([user_dataset, dd_pairs])
        else:
            dataset = dd_pairs
    else:
        if not user_dataset:
            raise ValueError("conversations.csv is empty or missing. Please have some conversations with the bot before fine-tuning.")
        dataset = user_dataset

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

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['user_input', 'bot_response'])

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