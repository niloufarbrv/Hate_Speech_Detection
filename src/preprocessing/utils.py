# Authors: Niloufar Beyranvand
import sys 
sys.path.append('/home/nbeyran/Hate_Speech_Detection')
from typing import List, Tuple
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import string
import transformers
from src.preprocessing.constants import HASHTAG_PATTERN, URL_PATTERN, USER_PATTERN, NUMBER_PATTERN, EMOTICAN_PATTERN, EMOJI_PATTERN


def drop_unwanted_columns(data, columns_to_drop=None):
    """
    Filters out specified columns

    Parameters:
    data (DataFrame): The pandas DataFrame to process.
    columns_to_drop (list of str, optional): Columns to drop from the DataFrame.

    Returns:
    DataFrame: The processed DataFrame.
    """
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return data


def replace_emoticons(text: str):
    """Replace emoticons with a placeholder."""
    for emoticon, placeholder in EMOTICAN_PATTERN.items():
        text = text.replace(emoticon, placeholder)
    return text

def replace_emojis(text: str):
    """Replace emojis with a placeholder."""
    return ''.join(EMOJI_PATTERN.get(char, char) for char in text)

def replace_urls(text: str):
    """Replace URLs with a placeholder."""
    return URL_PATTERN.sub("<url >", text)

def replace_users(text: str):
    """Replace user mentions with a placeholder."""
    return USER_PATTERN.sub("<user>", text)

def replace_numbers(text: str):
    """Replace numbers with a placeholder."""
    return NUMBER_PATTERN.sub("<number >", text)

def remove_punctuation(text: str):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def replace_hashtags(text: str):
    """Replace hashtags with a placeholder."""
    return HASHTAG_PATTERN.sub("<hashtag >", text)


def clean_text(text: str):
    """Apply all cleaning and tokenization steps to the text."""
    text = text.lower()
    text = replace_urls(text)
    text = replace_users(text)
    text = replace_hashtags(text)
    text = replace_emoticons(text)
    text = replace_emojis(text)
    text = replace_numbers(text)
    text = remove_punctuation(text)
    return text

def tokenize_and_prepare_dataset(sentences: List[str], labels, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenizes and prepares dataset for BERT model training.

    Args:
        sentences (list): List of sentences to tokenize.
        labels (list): List of labels corresponding to each sentence.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for encoding sentences.

    Returns:
        tuple: Tuple containing tokenized input ids, attention masks, and labels as numpy arrays.
    """
    # Prepare lists to collect the encoded results
    encoded_batch = [tokenizer(sentence, max_length=36, padding='max_length', truncation=True) 
                     for sentence in sentences]
    
    # Convert lists of encoded results to numpy arrays
    input_ids = np.array([eb['input_ids'] for eb in encoded_batch])
    attention_masks = np.array([eb['attention_mask'] for eb in encoded_batch])
    label_array = np.array(labels)
    
    return input_ids, attention_masks, label_array



def load_and_process(path: Path,
                    columns_to_drop : Tuple[str] =('count', 'hate_speech', 'offensive_language', 'neither')):
    
    """
    Load a dataset from a CSV file, preprocess the text data, and prepare it for input into a BERT model.

    The function performs the following steps:
    1. Reads the original dataset from a CSV file at the specified path.
    2. Drops specified columns that are not required for model training or analysis.
    3. Extracts tweets and their corresponding classification labels.
    4. Cleans the tweet text by removing noise and standardizing the format.
    
    Parameters:
    - path (Path): A Path object pointing to the CSV file to be processed.
    - tokenizer (transformers.PreTrainedTokenizer): A tokenizer instance compatible with the BERT model.
    - columns_to_drop (Tuple[str]): A tuple of column names to be removed from the dataset.
    
    Returns:
    - Tuple of NumPy arrays: input IDs, attention masks, and labels, all prepared for BERT model input.
    """
    original_data = pd.read_csv(path)
    # Filter out the unwanted columns from the DataFrame
    filtered_data = drop_unwanted_columns(data=original_data, columns_to_drop=columns_to_drop)
    # Separate the tweet texts and their associated labels
    data , labels = filtered_data['tweet'].tolist(), filtered_data['class']
    # Clean the tweet texts by removing unwanted characters, correcting case, etc.
    cleaned_tweets = [clean_text(tweet) for tweet in data]
    return cleaned_tweets, labels


def get_args():
    """Get command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/labeled_data.csv')
    parser.add_argument('--language_model_name_or_path', type=str, default='roberta-base', help='The name or path of the pretrained language model to us')
    parser.add_argument('--model_checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--do_train', action="store_false", help="Whether to run training.")
    parser.add_argument('--do_test', action="store_true", help="Whether to run testing.")
    parser.add_argument("--class_weights", action="store_false", help="Whether to use class weights for training.")
    parser.add_argument('--random_state', type=int, default=2018)
    parser.add_argument('--gpu', type=str, default='0', help='Specify which GPU(s) to use')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--validation_batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=36)
    parser.add_argument('--freeze_lm', type=bool, default=True)
    parser.add_argument('--number_of_classes', type=int, default=3)
    args = parser.parse_args()
    return args