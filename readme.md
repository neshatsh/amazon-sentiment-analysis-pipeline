# Text Analytics Portfolio

## Overview

A complete text analytics pipeline for sentiment analysis on Amazon product reviews. Shows the full journey from raw text to deep learning classification - preprocessing, traditional ML, word embeddings, and neural networks.


## The Pipeline

1. data-preparation

Clean and tokenize 1.28M Amazon reviews. Creates train/test splits and handles stopwords.

2. model-training
Multinomial Naive Bayes classification with different n-gram features. Traditional ML baseline.

3. word2vec-embeddings
Train word embeddings and analyze semantic similarities between sentiment words.

4. fully-connected-neural-network-classifier
PyTorch neural networks using Word2Vec features. Experiments with different activation functions and regularization.


## How It Connects
```bash
Raw Reviews → Preprocessing → Naive Bayes Models
                    ↓
            Word2Vec Training → Neural Network Classification

```

## Getting Started

1. Start with data-preparation to create clean datasets
2. Run model-training for baseline models
3. Train embeddings with word2vec-embeddings
4. Build neural networks in fully-connected-neural-network-classifier

Each folder has its own README with detailed instructions.
