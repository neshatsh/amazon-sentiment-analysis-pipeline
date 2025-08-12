#!/usr/bin/env python3
"""
Neural Network Classifier with PyTorch

This script implements a fully-connected feed-forward neural network classifier
for sentiment analysis of Amazon reviews using Word2Vec embeddings.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_embeddings(embeddings_path):
    """
    Load Word2Vec embeddings from Assignment 3
    
    Args:
        embeddings_path: Path to the Word2Vec model file
        
    Returns:
        word2vec_model: Loaded Word2Vec model
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embedding file not found at {embeddings_path}")

    model = Word2Vec.load(embeddings_path)
    
    return model

def load_data_and_labels(data_file, labels_file):
    """
    Load tokenized data and labels from CSV files
    
    Args:
        data_file: Path to tokenized data file
        labels_file: Path to labels file
        
    Returns:
        tuple: (processed_data, processed_labels)
    """
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [line.strip().split(',') for line in f if line.strip()]
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return data, labels

def create_document_embeddings(tokenized_documents, word2vec_model):
    """
    Convert tokenized documents to document embeddings using Word2Vec
    
    Args:
        tokenized_documents: List of tokenized documents (list of lists)
        word2vec_model: Trained Word2Vec model
        
    Returns:
        numpy.ndarray: Document embeddings matrix
    """
    embeddings = []
    for tokens in tokenized_documents:
        vecs = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        if vecs:
            doc_embedding = np.mean(vecs, axis=0)
        else:
            doc_embedding = np.zeros(word2vec_model.vector_size)
        embeddings.append(doc_embedding)
    return np.array(embeddings, dtype=np.float32)

class NeuralNetworkClassifier(nn.Module):
    """
    Fully-connected neural network classifier for sentiment analysis
    """
    
    def __init__(self, input_size, hidden_size, num_classes, activation='relu', dropout_rate=0.5):
        """
        Initialize the neural network
        
        Args:
            input_size: Size of input features (Word2Vec embedding dimension)
            hidden_size: Size of hidden layer
            num_classes: Number of output classes
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            dropout_rate: Dropout rate for regularization
        """
        super(NeuralNetworkClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, l2_reg=0.001):
    """
    Train the neural network model
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        l2_reg: L2 regularization strength
        
    Returns:
        dict: Training history with losses and accuracies, best validation accuracy
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, best_val_acc


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        
    Returns:
        float: Test accuracy
    """
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def make_sure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    """
    Main execution function
    """
    
    print("MSCI 641 Assignment 4: Neural Network Classifier")
    print("=" * 50)
    
    # Configuration
    EMBEDDING_PATH = "embeddings/word2vec_model.model" 
    DATA_DIR = "data"
    MODELS_DIR = "models"
    
    # Hyperparameters
    HIDDEN_SIZE = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Experimental configurations
    experiments = [
        # (activation, l2_reg, dropout_rate)
        ('relu', 0.001, 0.3),
        ('relu', 0.01, 0.3),
        ('relu', 0.001, 0.5),
        ('sigmoid', 0.001, 0.3),
        ('sigmoid', 0.01, 0.3),
        ('tanh', 0.001, 0.3),
        ('tanh', 0.01, 0.3),
    ]
    
    # Create models directory
    make_sure_dir_exists(MODELS_DIR)
  
    word2vec_model = load_embeddings(EMBEDDING_PATH)

    # Load and preprocess data
    train_x, train_y = load_data_and_labels(os.path.join(DATA_DIR, "train_subset.csv"), os.path.join(DATA_DIR, "train_labels_subset.csv"))
    val_x, val_y = load_data_and_labels(os.path.join(DATA_DIR, "val_subset.csv"), os.path.join(DATA_DIR, "val_labels_subset.csv"))
    test_x, test_y = load_data_and_labels(os.path.join(DATA_DIR, "test.csv"), os.path.join(DATA_DIR, "test_labels.csv"))

    # Create document embeddings
    X_train = create_document_embeddings(train_x, word2vec_model)
    X_val = create_document_embeddings(val_x, word2vec_model)
    X_test = create_document_embeddings(test_x, word2vec_model)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_y)
    y_val = label_encoder.transform(val_y)
    y_test = label_encoder.transform(test_y)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE)

    results = []
    best_models = {}
    best_model_state = None
    best_val_acc = 0.0
    best_model_info = None

    for activation, l2, dropout in experiments:
        model = NeuralNetworkClassifier(word2vec_model.vector_size, HIDDEN_SIZE, 2, activation, dropout)
        history, val_acc = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, l2)
        test_acc = evaluate_model(model, test_loader)

        model_file = f"{activation}_l2_{l2}_dropout_{dropout}.pth"
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, model_file))

        results.append([activation, l2, dropout, val_acc, test_acc])

        if activation not in best_models or val_acc > best_models[activation][3]:
            best_models[activation] = [activation, l2, dropout, val_acc, test_acc]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_model_info = [activation, l2, dropout, val_acc, test_acc]

    # Save best overall model
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(MODELS_DIR, "best_model.pth"))

    # Save results to CSV files
    pd.DataFrame(results, columns=["Activation Function", "L2 Regularization", "Dropout Rate", "Validation Accuracy", "Test Accuracy"]).to_csv("hyperparameter_tuning.csv", index=False)
    pd.DataFrame(best_models.values(), columns=["Activation Function", "L2 Regularization", "Dropout Rate", "Validation Accuracy", "Test Accuracy"]).to_csv("results.csv", index=False)

    # Print best configuration
    print(f"\nBest model configuration (based on validation accuracy):")
    print(f"Activation: {best_model_info[0]}, L2: {best_model_info[1]}, Dropout: {best_model_info[2]}, "
          f"Val Acc: {best_model_info[3]:.4f}, Test Acc: {best_model_info[4]:.4f}")


if __name__ == "__main__":
    main()