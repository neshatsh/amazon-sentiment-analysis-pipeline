# Overview

A comprehensive sentiment classification system using Multinomial Naive Bayes (MNB) classifiers on Amazon product reviews. This implementation explores the impact of different text preprocessing strategies (stopword handling) and feature extraction techniques (unigrams, bigrams, and their combinations) on classification performance.

# Features

- Multiple Feature Configurations: Unigrams, bigrams, and combined unigram+bigram features
- Preprocessing Comparison: Evaluates performance with and without stopword removal
- Automated Pipeline: Complete training, evaluation, and results generation workflow
- Performance Analysis: Systematic comparison of 6 different model configurations
- Scalable Implementation: Handles large datasets efficiently using scikit-learn
- Reproducible Results: Consistent model training and evaluation metrics

# Quick Start

## 1. Setup environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

## 2. Prepare Data

This project requires preprocessed data from the data preparation pipeline. Ensure you have:
```bash
data/
├── train.csv         # Training data with stopwords
├── train_ns.csv      # Training data without stopwords
├── test.csv          # Test data with stopwords
├── test_ns.csv       # Test data without stopwords
├── train_labels.csv  # Training labels
└── test_labels.csv   # Test labels

```

## 3. Run Classification

```bash
python main.py
```

# Model Configurations

The system trains and evaluates 6 different classifier configurations:

## Without Stopword Removal

1. Unigrams Only - Single word features
2. Bigrams Only - Two consecutive word features
3. Unigrams + Bigrams - Combined single and two-word features

## With Stopword Removal

4. Unigrams Only - Single word features, stopwords filtered
5. Bigrams Only - Two consecutive word features, stopwords filtered
6. Unigrams + Bigrams - Combined features, stopwords filtered


# Implementation

## Core Pipeline:

```bash
# For each configuration:
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=ngram_range)),
    ('classifier', MultinomialNB())
])
```

## N-gram Configuration:

- Unigrams only: ngram_range=(1,1)
- Bigrams only: ngram_range=(2,2)
- Combined: ngram_range=(1,2)

# Output

## Results: results.csv with accuracy metrics for each configuration
