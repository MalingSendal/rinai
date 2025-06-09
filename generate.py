# generate.py
def generate_response(user_input, model, tokenizer):
    user_input = str(user_input).strip()
    inputs = tokenizer(
        user_input + tokenizer.eos_token,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
        return_attention_mask=True
    )
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,  # Use pad_token_id
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return response.strip()