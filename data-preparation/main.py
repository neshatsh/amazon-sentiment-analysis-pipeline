"""
Data Preparation
-----------------------------------------
This script performs tokenization, cleaning, and data splitting on the Amazon reviews corpus.
"""

import argparse
import os
import random
import re
import string


def load_data(data_dir):
    """
    Load positive and negative reviews from the given directory.
    
    Args:
        data_dir (str): Path to directory containing pos.txt and neg.txt
        
    Returns:
        list: List of tuples (review_text, label)
    """

    data = []
    pos_path = os.path.join(data_dir, 'pos.txt')
    neg_path = os.path.join(data_dir, 'neg.txt')

    with open(pos_path, 'r', encoding='utf-8') as pos_file:
        for line in pos_file:
            line = line.strip()
            if line:
                data.append((line, 'positive'))


    with open(neg_path, 'r', encoding='utf-8') as neg_file:
        for line in neg_file:
            line = line.strip()
            if line:
                data.append((line, 'negative'))

    return data



def shuffle_data(labeled_data, seed=42):
    """
    Shuffle the labeled data.
    
    Args:
        labeled_data (list): List of (review_text, label) tuples
        seed (int): Random seed for reproducibility
        
    Returns:
        list: Shuffled list of (review_text, label) tuples
    """

    random.seed(seed)
    random.shuffle(labeled_data)
    return labeled_data


def tokenize(text):
    """
    Tokenize a text string.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """

    pattern = r"[!#$%&()*+/:,;.<=>@\\[\\]^`{|}~\t\n]"
    cleaned_text = re.sub(pattern, '', text)
    tokens = cleaned_text.strip().split()
    return tokens


def load_stopwords(stopwords_path):
    """
    Load stopwords from the provided file.
    
    Args:
        stopwords_path (str): Path to the stopwords file
        
    Returns:
        set: Set of stopwords
    """

    try:
        with open(stopwords_path, 'r', encoding='utf-8') as stopwords_file:
            return set(line.strip() for line in stopwords_file if line.strip())

    except FileNotFoundError:
        print(f'Stopwords file not found.')
        return set()


def remove_stopwords(tokens, stopwords):
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens (list): List of tokens
        stopwords (set): Set of stopwords
        
    Returns:
        list: List of tokens with stopwords removed
    """

    return [token for token in tokens if token not in stopwords]


def split_data(tokenized_texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        tokenized_texts (list): List of tokenized texts
        labels (list): List of corresponding labels
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio != 1.0:
        raise ValueError(f"Train, val, and test ratios must sum to 1.0 (got {total_ratio})")

    combined = list(zip(tokenized_texts, labels))
    random.seed(seed)
    random.shuffle(combined)

    tokenized_texts, labels = zip(*combined)
    total = len(tokenized_texts)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (
        list(tokenized_texts[:train_end]),
        list(tokenized_texts[train_end:val_end]),
        list(tokenized_texts[val_end:]),
        list(labels[:train_end]),
        list(labels[train_end:val_end]),
        list(labels[val_end:])
    )


def write_to_csv(tokenized_texts, output_file):
    """
    Write tokenized texts to a CSV file.
    
    Args:
        tokenized_texts (list): List of token lists
        output_file (str): Path to output file
    """

    with open(output_file, 'w', encoding='utf-8') as output_file:
        for tokens in tokenized_texts:
            output_file.write(','.join(tokens) + '\n')


def write_labels_to_csv(labels, output_file):
    """
    Write labels to a CSV file.
    
    Args:
        labels (list): List of labels
        output_file (str): Path to output file
    """

    with open(output_file, 'w', encoding='utf-8') as output_file:
        for label in labels:
            output_file.write(label + '\n')



def main():
    parser = argparse.ArgumentParser(description='MSCI 641 Assignment 1: Data Preparation')
    parser.add_argument('data_dir', type=str, help='Path to directory containing pos.txt and neg.txt')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    pos_path = os.path.join(data_dir, 'pos.txt')
    neg_path = os.path.join(data_dir, 'neg.txt')
    stopwords_path = os.path.join(data_dir, 'stopwords.txt')
    
    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        print(f"Error: pos.txt or neg.txt not found in {data_dir}")
        return
    
    if not os.path.exists(stopwords_path):
        print(f"Error: stopwords.txt not found in {data_dir}")
        return

    # 1. Load data (combine positive and negative reviews with their labels)
    labeled_data = load_data(data_dir)
    
    # 2. Shuffle the data
    shuffled_data = shuffle_data(labeled_data)
    
    # 3. Separate texts and labels
    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]
    
    # 4. Tokenize the texts
    tokenized_texts = [tokenize(text) for text in texts]
    
    # 5. Load stopwords
    stopwords = load_stopwords(stopwords_path)
    
    # 6. Create version without stopwords
    tokenized_texts_ns = [remove_stopwords(tokens, stopwords) for tokens in tokenized_texts]
    
    # 7. Split the data into train/val/test sets
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(tokenized_texts, labels)
    train_texts_ns, val_texts_ns, test_texts_ns, _, _, _ = split_data(tokenized_texts_ns, labels)
    
    # 8. Write tokenized texts and labels to CSV files
    write_to_csv(tokenized_texts, os.path.join(data_dir, 'out.csv'))
    write_to_csv(train_texts, os.path.join(data_dir, 'train.csv'))
    write_to_csv(val_texts, os.path.join(data_dir, 'val.csv'))
    write_to_csv(test_texts, os.path.join(data_dir, 'test.csv'))
    
    write_to_csv(tokenized_texts_ns, os.path.join(data_dir, 'out_ns.csv'))
    write_to_csv(train_texts_ns, os.path.join(data_dir, 'train_ns.csv'))
    write_to_csv(val_texts_ns, os.path.join(data_dir, 'val_ns.csv'))
    write_to_csv(test_texts_ns, os.path.join(data_dir, 'test_ns.csv'))
    
    write_labels_to_csv(labels, os.path.join(data_dir, 'out_labels.csv'))
    write_labels_to_csv(train_labels, os.path.join(data_dir, 'train_labels.csv'))
    write_labels_to_csv(val_labels, os.path.join(data_dir, 'val_labels.csv'))
    write_labels_to_csv(test_labels, os.path.join(data_dir, 'test_labels.csv'))
    
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()