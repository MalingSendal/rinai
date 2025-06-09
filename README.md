# 🎀 rinai: Conversational AI Chatbot 🎀

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Overview

**rinai** is a simple, interactive chatbot powered by a fine-tunable GPT-2 model. You can chat, collect conversations, and fine-tune the model on your own data — all from the command line!

---

## 🚀 Features

- 💬 Interactive chat loop
- 📝 Conversation logging and filtering
- 🔄 On-the-fly model fine-tuning with your own conversations
- 💾 Model saving and loading
- 🦄 Built with [HuggingFace Transformers](https://huggingface.co/transformers/)

---

## 🛠️ Project Structure

rinai/ ├── main.py # Entry point for the chatbot ├── pre_trained.py # Loads the pre-trained GPT-2 model and tokenizer ├── generate.py # Handles response generation ├── store_conv.py # Saves and filters conversations to CSV ├── fine_tuning.py # Fine-tunes the model on your conversations ├── save_model.py # Save/load model and tokenizer ├── loop_chat.py # Chat loop logic ├── conversations.csv # Your conversation history (used for fine-tuning) ├── chatbot_model/ # Saved/fine-tuned model checkpoints └── README.md # This file!


---

## ⚡ Quickstart

1. **Install dependencies:**
    ```sh
    pip install transformers datasets pandas
    ```

2. **Run the chatbot:**
    ```sh
    python main.py
    ```

3. **Chat with rinai!**
    - Type your message and press Enter.
    - Type `fine-tune` to train the model on your collected conversations.
    - Type `exit` to quit.

---

## 🏆 Fine-tuning

- Your chats are saved in [`conversations.csv`](conversations.csv).
- After collecting enough conversations, type `fine-tune` in the chat to improve the model with your data.
- The fine-tuned model is saved in [`chatbot_model/`](chatbot_model/).

---

## 📁 Data Format

The conversation log is stored as a CSV file with two columns:

| user_input | bot_response |
|------------|--------------|
| "Hello"    | "Hi! How can I assist you today?" |

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 💡 Credits

- Built with [Transformers](https://huggingface.co/transformers/) by Hugging Face
- Inspired by open-source conversational AI projects

---

> 🎀 Happy chatting with **rinai**! 🎀