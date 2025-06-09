# store-conv.py
import pandas as pd
import csv
import re

def filter_text(text):
    """
    Filter text to allow only alphabetic characters, spaces, and basic punctuation (. , ? !).
    Remove symbols, newlines, and other special characters.
    """
    # Convert to string and remove newlines
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    # Keep only a-z, A-Z, spaces, and basic punctuation; replace others with space
    text = re.sub(r'[^a-zA-Z\s.,?!]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    return text.strip()

def save_conversation(user_input, bot_response):
    # Filter user_input and bot_response
    user_input = filter_text(user_input)
    bot_response = filter_text(bot_response)
    
    # Ensure inputs are not empty after filtering
    if not user_input or not bot_response:
        return  # Skip saving if either field is empty
    
    # Escape double quotes for CSV
    user_input = user_input.replace('"', '""')
    bot_response = bot_response.replace('"', '""')
    
    # Create a DataFrame with quoted strings
    data = {"user_input": f'"{user_input}"', "bot_response": f'"{bot_response}"'}
    df = pd.DataFrame([data])
    
    # Write to CSV, ensuring proper quoting
    df.to_csv("conversations.csv", mode='a', index=False, header=not pd.io.common.file_exists("conversations.csv"), quoting=csv.QUOTE_NONE, escapechar='\\')