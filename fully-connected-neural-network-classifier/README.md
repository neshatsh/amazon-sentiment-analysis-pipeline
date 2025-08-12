## Overview

PyTorch implementation of a fully-connected neural network for sentiment classification using Word2Vec embeddings. Systematically experiments with different activation functions, regularization techniques, and hyperparameters to optimize performance on Amazon product reviews.
Quick Start

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run experiments (requires preprocessed data and Word2Vec model from the data-preparation and word2vec-embeddings pipelines)
python main.py
```
## Architecture
### Network Design:

```bash
Input Layer (Word2Vec embeddings) 
    ↓
Hidden Layer (128 neurons, configurable activation)
    ↓
Dropout (regularization)
    ↓
Output Layer (2 classes: positive/negative)
```

### Document Embedding Strategy:

- Averages Word2Vec embeddings for all words in each document
- Handles out-of-vocabulary words by skipping them
- Creates fixed-size representations for variable-length documents

### Experiments

Systematic Hyperparameter Search:

- Activation Functions: ReLU, Sigmoid, Tanh
- L2 Regularization: 0.001, 0.01
- Dropout Rates: 0.3, 0.5
- Early Stopping: Monitors validation performance

### Model Configurations Tested:

```bash
experiments = [
    ('relu', 0.001, 0.3),
    ('relu', 0.01, 0.3),
    ('relu', 0.001, 0.5),
    ('sigmoid', 0.001, 0.3),
    ('sigmoid', 0.01, 0.3),
    ('tanh', 0.001, 0.3),
    ('tanh', 0.01, 0.3),
]
```

## Output

### Results Files:

- hyperparameter_tuning.csv - Every experiment I ran and its results
- results.csv - Best model for each activation function
- report.md - My analysis of what worked and why

### Trained Models:

 When you run the code, models are saved locally in a models/ folder (excluded from repo due to file size). The code automatically saves the best performing model for each configuration.