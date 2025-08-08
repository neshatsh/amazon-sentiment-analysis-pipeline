## Amazon Reviews Text Preprocessing Pipeline

## Overview
A comprehensive text preprocessing system for sentiment analysis on Amazon product reviews. This pipeline transforms raw review text into clean, tokenized datasets suitable for machine learning applications, generating both stopword-preserved and stopword-filtered versions with proper train/validation/test splits.


### Environment Setup
Set up a Python virtual environment and install the required dependencies:

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


### Download Dataset
We worked with the Amazon reviews corpus which contains two classes of consumer product reviews: positive and negative. The dataset is available at:
https://github.com/fuzhenxin/textstyletransferdata/tree/master/sentiment

Download the dataset by running:
```bash
# In the data directory
curl -o data/pos.txt https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/pos.txt
curl -o data/neg.txt https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/neg.txt
```


### Output Files

The pipeline generates 12 CSV files in the `data/` directory:

#### Text Files (with comma-separated tokens):
- `out.csv`: All tokenized sentences with stopwords
- `train.csv`: Training set with stopwords (80%)
- `val.csv`: Validation set with stopwords (10%)
- `test.csv`: Test set with stopwords (10%)
- `out_ns.csv`: All tokenized sentences without stopwords
- `train_ns.csv`: Training set without stopwords (80%)
- `val_ns.csv`: Validation set without stopwords (10%)
- `test_ns.csv`: Test set without stopwords (10%)

#### Label Files (one label per line):
- `out_labels.csv`: Labels for all sentences ("positive" or "negative")
- `train_labels.csv`: Labels for training set
- `val_labels.csv`: Labels for validation set
- `test_labels.csv`: Labels for test set


### Sample Output Format

#### Example lines for text files (with stopwords):
```
this,product,is,amazing,i,really,love,it,so,much
the,battery,life,on,this,device,is,quite,impressive,it,lasts,all,day
not,worth,the,money,very,disappointed,with,the,quality
```

#### Corresponding lines for text files (without stopwords):
```
product,amazing,really,love,much
battery,life,device,quite,impressive,lasts,day
worth,money,disappointed,quality
```

#### Corresponding lines for label files:
```
positive
positive
negative
```

### Processing Pipeline

### 1. Data Loading & Labeling

- Loads positive reviews from pos.txt → labels as "positive"
- Loads negative reviews from neg.txt → labels as "negative"
- Combines into unified dataset with preserved label associations

### 2. Data Shuffling

- Randomizes review order using fixed seed (42) for reproducibility
- Maintains text-label pair integrity throughout process

### 3. Tokenization & Cleaning

- Splits text into individual word tokens using whitespace
- Removes special characters: !#$%&()*+/:,;.<=>@[\]^{|}~\t\n`
- Preserves original word order and capitalization

### 4. Stopword Processing

- Loads custom stopwords list from file
- Creates parallel versions:

- Standard: All tokens preserved
- Filtered: Common stopwords removed


### 5. Data Splitting

- Applies consistent 80%/10%/10% train/validation/test split
- Uses same split indices for both text versions and labels
- Ensures reproducible partitions across runs

### 6. File Export

- Exports tokenized text as comma-separated values
- Generates corresponding label files with perfect line alignment
- All files ready for immediate use in ML pipelines

