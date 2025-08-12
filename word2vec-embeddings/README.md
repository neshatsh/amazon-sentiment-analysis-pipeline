#Word2Vec Word Embeddings for Sentiment Analysis

# Overview

Word2Vec implementation using Gensim to train word embeddings on Amazon product reviews. This project analyzes semantic relationships between words by training skip-gram models and examining similarity patterns for sentiment-bearing words like "good" and "bad".

## Features

- Skip-gram Word2Vec Training: Uses Gensim to train word embeddings on review text
- Similarity Analysis: Finds most similar words to target sentiment words
- Configurable Parameters: Adjustable vector dimensions, window size, and training epochs
- Comprehensive Output: Generates both model files and detailed similarity analysis
- Memory Efficient: Handles large datasets with subset training options

## Quick Start

### 1: Setup Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2: Prepare Data

Requires preprocessed tokenized data from the data preparation pipeline:
```bash
data/
├── train_subset.csv    # Subset of training data (recommended)
└── train_ns_subset.csv # Alternative: without stopwords
```

### 3. Train Model and Analyze
```bash
python main.py
```


## Model Configuration
### Default Parameters:

- Algorithm: Skip-gram (sg=1)
- Vector Size: 100 dimensions
- Context Window: 5 words
- Min Count: 5 (filters rare words)
- Training Epochs: 10
- Workers: 4 (parallel processing)

## Training Process

1. Data Loading: Reads comma-separated tokens from CSV files
2. Model Initialization: Creates Word2Vec model with specified parameters
3. Vocabulary Building: Automatically builds vocabulary from training corpus
4. Training: Uses skip-gram algorithm to learn word representations
5. Similarity Analysis: Computes cosine similarities for target words
6. Results Export: Saves model and analysis to files

## Output Files
### Model File

- **`word2vec_model.model`** - Trained Word2Vec model (can be loaded for further analysis)

### Analysis Results

- **`word_similarities.json`** containing:
   ```json
   {
     "good_similar": [
       ["excellent", 0.734],
       ["great", 0.712],
       ...
     ],
     "bad_similar": [
       ["terrible", 0.689],
       ["awful", 0.654],
       ...
     ],
     "analysis": {
       "good_words": ["list", "of", "similar", "words"],
       "bad_words": ["list", "of", "similar", "words"]
     }
   }
   ```

### Written Analysis

`analysis.txt` - Interpretation of similarity results and Word2Vec behavior

## Key Insights
### Word Similarity Analysis
The implementation analyzes how Word2Vec captures semantic relationships:

- Sentiment Clustering: Similar sentiment words cluster together in vector space
- Contextual Learning: Word2Vec learns from surrounding word context
- Semantic vs. Sentiment: Distinguishes between semantic similarity and sentiment polarity