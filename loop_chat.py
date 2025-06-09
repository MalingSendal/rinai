# loop-chat.py
def chat(model, tokenizer, save_conversation, generate_response, fine_tune_model):
    print("Start chatting with the bot (type 'exit' to stop, 'fine-tune' to train):")
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "fine-tune":
            print("Fine-tuning model...")
            model = fine_tune_model(model, tokenizer)
            print("Fine-tuning complete!")
            continue
        
        response = generate_response(user_input, model, tokenizer)
        print(f"Bot: {response}")
        save_conversation(user_input, response)