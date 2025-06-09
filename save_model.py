# save-model.py
def save_model(model, tokenizer):
    model.save_pretrained("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")

def load_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained("./chatbot_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./chatbot_model")
    return model, tokenizer