#!/usr/bin/env python3
"""
Text Classification with Multinomial Naive Bayes

"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load tokenized data from a CSV file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [line.strip().split(',') for line in f if line.strip()]
    return data

def load_labels(file_path):
    """Load labels from a CSV file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def tokens_to_strings(token_lists):
    """Convert list of token lists into space-separated strings"""
    return [' '.join(tokens) for tokens in token_lists]


def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    """Train a Multinomial Naive Bayes classifier
    
    Args:
        train_data: List of texts to train on
        train_labels: List of corresponding labels
        use_bigrams: Whether to include bigram features
        use_unigrams: Whether to include unigram features
        
    Returns:
        A trained model that can be used for prediction
    
    Implementation steps:
    1. Configure the n-gram range based on the feature requirements
       - If using only unigrams, set range to (1,1)
       - If using only bigrams, set range to (2,2)
       - If using both, set range to (1,2)
       
    2. Create a pipeline with two steps:
       - A text vectorizer that converts text to token counts with the configured n-gram range
       - A Multinomial Naive Bayes classifier
       
    3. Train the pipeline on the training data and labels
    
    4. Return the trained pipeline
    """
    
    if use_unigrams and use_bigrams:
        ngram_range = (1, 2)
    elif use_bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=ngram_range)),
        ('classifier', MultinomialNB())
    ])

    pipeline.fit(train_data, train_labels)
    return pipeline

def evaluate_model(model, test_data, test_labels):
    """Evaluate a model on test data
    
    Args:
        model: A trained classifier model
        test_data: List of texts to evaluate on
        test_labels: List of true labels for the test data
        
    Returns:
        float: The accuracy of the model on the test data
        
    Implementation steps:
    1. Use the model to generate predictions for all samples in the test data
    
    2. Calculate the accuracy by:
       - Comparing each predicted label with the corresponding true label
       - Counting how many predictions match their true labels
       - Dividing this count by the total number of test samples
       
    3. Return the calculated accuracy (a value between 0 and 1)
    """
    
    predictions = model.predict(test_data)
    correct = sum(p == t for p, t in zip(predictions, test_labels))
    return correct / len(test_labels)

def main():
    
    configs = [
        ("no", "unigrams", "train.csv", "test.csv"),
        ("no", "bigrams", "train.csv", "test.csv"),
        ("no", "unigrams+bigrams", "train.csv", "test.csv"),
        ("yes", "unigrams", "train_ns.csv", "test_ns.csv"),
        ("yes", "bigrams", "train_ns.csv", "test_ns.csv"),
        ("yes", "unigrams+bigrams", "train_ns.csv", "test_ns.csv"),
    ]

    os.makedirs("models", exist_ok=True)
    results = []

    for stopwords_flag, feature_type, train_file, test_file in configs:
        use_unigrams = "unigrams" in feature_type
        use_bigrams = "bigrams" in feature_type

        train_tokens = load_data(os.path.join("data", train_file))
        test_tokens = load_data(os.path.join("data", test_file))
        train_labels = load_labels(os.path.join("data", "train_labels.csv"))
        test_labels = load_labels(os.path.join("data", "test_labels.csv"))

        train_texts = tokens_to_strings(train_tokens)
        test_texts = tokens_to_strings(test_tokens)

        model = train_model(train_texts, train_labels, use_bigrams, use_unigrams)
        accuracy = evaluate_model(model, test_texts, test_labels)

        print(f"Trained model: {feature_type} (stopwords removed: {stopwords_flag}) — Accuracy: {accuracy:.3f}")

        if stopwords_flag == "no":
            stopword_tag = "with_stopwords"
        else:
            stopword_tag = "without_stopwords"

        model_name = f"{feature_type.replace('+', '_')}_{stopword_tag}.pkl"

        
        model_path = os.path.join("models", model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        results.append([stopwords_flag, feature_type, round(accuracy, 3)])

    
    df = pd.DataFrame(results, columns=["Stopwords removed", "text features", "Accuracy (test set)"])
    df.to_csv("results.csv", index=False)
    print("Finished training all models. Results saved to results.csv.")

if __name__ == "__main__":
    main()