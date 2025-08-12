#!/usr/bin/env python3
"""
Word2Vec Training with Gensim

"""

import os
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

def load_data(file_path):
    """Load tokenized data from a CSV file
    
    Args:
        file_path: Path to the CSV file containing tokenized sentences
        
    Returns:
        List of lists, where each inner list is a tokenized sentence
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Each line contains comma-separated tokens
        data = [line.strip().split(',') for line in f if line.strip()]
    return data

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=5, 
                        workers=4, epochs=10, sg=1):
    """Train a Word2Vec model using Gensim
    
    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Ignores words with total frequency lower than this
        workers: Number of worker threads
        epochs: Number of training epochs
        sg: Training algorithm (1 for skip-gram, 0 for CBOW)
        
    Returns:
        Trained Word2Vec model
        
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )
    return model

def find_similar_words(model, word, topn=20):
    """Find the most similar words to a given word
    
    Args:
        model: Trained Word2Vec model
        word: Target word to find similarities for
        topn: Number of most similar words to return
        
    Returns:
        List of tuples (word, similarity_score) sorted by similarity

    """
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        print(f"'{word}' not in vocabulary.")
        return []

def save_model(model, filepath):
    """Save the trained Word2Vec model to disk
    
    Args:
        model: Trained Word2Vec model
        filepath: Path where to save the model
        
    """
    model.save(filepath)

def load_model(filepath):
    """Load a Word2Vec model from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded Word2Vec model
        
    """
    
    return Word2Vec.load(filepath)

def analyze_word_similarities(model):
    """Analyze similarities for 'good' and 'bad' words
    
    Args:
        model: Trained Word2Vec model
        
    Returns:
        Dictionary containing analysis results

    """
    results = {
        'good_similar': [],
        'bad_similar': [],
        'analysis': {
            'good_words': [],
            'bad_words': []
        }
    }
    
    results['good_similar'] = find_similar_words(model, "good", topn=20)
    results['bad_similar'] = find_similar_words(model, "bad", topn=20)
    results['analysis']['good_words'] = [word for word, _ in results['good_similar']]
    results['analysis']['bad_words'] = [word for word, _ in results['bad_similar']]
    
    
    return results

def save_results(results, filepath='word_similarities.json'):
    """Save analysis results to a JSON file
    
    Args:
        results: Dictionary containing analysis results
        filepath: Path to save the results
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    """Main function to train Word2Vec model and analyze word similarities
    """
    
    print("Starting Word2Vec training...")
    

    data_path = os.path.join("data", "train_subset.csv")  # or train_ns_subset.csv
    sentences = load_data(data_path)

    model = train_word2vec_model(sentences)
    save_model(model, "word2vec_model.model")

    print("Analyzing word similarities...")
    results = analyze_word_similarities(model)
    save_results(results, 'word_similarities.json')

    print("Top similar words to 'good':")
    for word, score in results['good_similar'][:10]:
        print(f"  {word}: {score:.3f}")

    print("Top similar words to 'bad':")
    for word, score in results['bad_similar'][:10]:
        print(f"  {word}: {score:.3f}")

    print("All done! Model and results saved.")


if __name__ == "__main__":
    main()
