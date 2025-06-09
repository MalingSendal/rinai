# main.py
from pre_trained import tokenizer, model  # Import model and tokenizer
from store_conv import save_conversation
from generate import generate_response
from fine_tuning import fine_tune_model
from loop_chat import chat
from save_model import save_model, load_model

def main():
    # Ensure model and tokenizer are available globally or passed appropriately
    global model, tokenizer
    
    # Optionally, load a previously fine-tuned model
    try:
        model, tokenizer = load_model()
        print("Loaded fine-tuned model.")
    except:
        print("Using pre-trained model.")

    # Start the chat loop, passing model and tokenizer to functions that need them
    chat(model, tokenizer, save_conversation, generate_response, fine_tune_model)

if __name__ == "__main__":
    main()

