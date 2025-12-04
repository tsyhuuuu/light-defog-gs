#!/usr/bin/env python3
"""
Deep Learning Classifiers for Fog Gaussian Classification
Includes: MLP, Transformer
Based on fog_removal.ipynb notebook
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path="config/dl_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(csv_path, test_size=0.2, batch_size=1024, num_workers=4):
    """Load and prepare data for deep learning."""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Feature columns
    feature_columns = [col for col in df.columns 
                      if col not in ['beta', 'alpha', 'is_fog', 'pos_x', 'pos_y', 'pos_z', 
                                   'rot_0', 'rot_1', 'rot_2', 'rot_3', 'id']]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Using {len(feature_columns)} features")
    print(f"Class distribution: {df['is_fog'].value_counts().to_dict()}")
    
    X = df[feature_columns].values
    y = df['is_fog'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, scaler, feature_columns, len(feature_columns)

# Model Definitions
class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, input_size, hidden_layers, dropout=0.3, activation='relu'):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        activation_fn = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh()
        }[activation]
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn,
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        # layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CNN1DClassifier(nn.Module):
    """1D Convolutional Network."""
    def __init__(self, input_size, channels, kernel_sizes, dropout=0.2):
        super(CNN1DClassifier, self).__init__()
        
        # Reshape input for 1D CNN
        self.input_size = input_size
        
        layers = []
        in_channels = 1
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape for 1D CNN: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, channels[-1])
        return self.classifier(x)

class LSTMClassifier(nn.Module):
    """LSTM Network for classification."""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,  # Process each feature as a sequence
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape for LSTM: (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)  # (batch_size, input_size, 1)
        
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the last output
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.classifier(hidden)

class TransformerClassifier(nn.Module):
    """Transformer Encoder for classification."""
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = self.input_projection(x).unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.pos_encoder
        
        x = self.transformer(x)  # (batch_size, 1, d_model)
        x = x.squeeze(1)  # (batch_size, d_model)
        
        return self.classifier(x)

def create_model(model_name, config, input_size):
    """Create model based on configuration."""
    if model_name == 'mlp':
        return MLPClassifier(
            input_size,
            config['hidden_layers'],
            config['dropout'],
            config['activation']
        )
    
    elif model_name == 'cnn1d':
        return CNN1DClassifier(
            input_size,
            config['channels'],
            config['kernel_sizes'],
            config['dropout']
        )
    
    elif model_name == 'lstm':
        return LSTMClassifier(
            input_size,
            config['hidden_size'],
            config['num_layers'],
            config['dropout'],
            config['bidirectional']
        )

    elif model_name == 'transformer':
        return TransformerClassifier(
            input_size,
            config['d_model'],
            config['nhead'],
            config['num_layers'],
            config['dim_feedforward'],
            config['dropout']
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, test_loader, config):
    """Train a deep learning model."""
    device = torch.device(config['hardware']['device'] if config['hardware']['device'] != 'auto' 
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['models']['deep_learning']['training']['learning_rate']),
        weight_decay=float(config['models']['deep_learning']['training']['weight_decay'])
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"Training on device: {device}")
    
    for epoch in tqdm(range(config['models']['deep_learning']['training']['epochs']), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        i = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

            i += 1
        
        print('='*10, i)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - float(config['models']['deep_learning']['training']['min_delta']):
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['models']['deep_learning']['training']['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    best_model_state = model.state_dict().copy()
    model.load_state_dict(best_model_state)
    return model, history

def evaluate_model(model, test_loader, device):
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            predictions = (outputs.squeeze() > 0.5).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions)
    
    return accuracy, cm, report, all_predictions, all_targets

def plot_training_history(history, model_name, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(y_test, y_pred, accuracy, cm, model_name, output_dir):
    """Create visualization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{model_name} Confusion Matrix\n(Accuracy: {accuracy:.4f})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Class distribution
    class_dist = pd.Series(y_test).value_counts()
    axes[1].bar(class_dist.index, class_dist.values, color=['blue', 'red'], alpha=0.7)
    axes[1].set_title('Test Set Class Distribution')
    axes[1].set_xlabel('Class (0=No Fog, 1=Fog)')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load configuration
    config = load_config()
    
    # Create output directories
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    # Prepare data
    train_loader, test_loader, scaler, features, input_size = prepare_data(
        config['data']['csv_path'],
        config['data']['test_size'],
        config['data']['batch_size'],
        config['data']['num_workers']
    )
    
    device = torch.device(config['hardware']['device'] if config['hardware']['device'] != 'auto' 
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Train all models
    for model_name, model_config in config['models'].items():
        print(f"\n{'='*50}")
        print(f"Training {model_config['name']} ({model_name.upper()})")
        print(f"{'='*50}")
        
        # Create model
        model = create_model(model_name, model_config, input_size)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        model, history = train_model(model, train_loader, test_loader, config)
        
        # Evaluate model
        accuracy, cm, report, y_pred, y_test = evaluate_model(model, test_loader, device)
        
        print(f"\n{model_config['name']} Results:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        # Save results
        results[model_name] = {
            'accuracy': accuracy,
            'model_name': model_config['name']
        }
        
        # Create visualizations
        if config['output']['save_plots']:
            plot_training_history(history, model_config['name'], config['output']['results_dir'])
            create_visualizations(y_test, y_pred, accuracy, cm, 
                               model_config['name'], config['output']['results_dir'])
        
        # Save model
        if config['output']['save_best_only']:
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'features': features,
                'config': model_config,
                'accuracy': accuracy
            }, f"{config['output']['model_dir']}/{model_name}_model.pth")
    
    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_results):
        print(f"{i+1}. {result['model_name']:25s} - {result['accuracy']:.4f}")
    
    # Create comparison plot
    if config['output']['save_plots']:
        plt.figure(figsize=(12, 6))
        model_names = [results[name]['model_name'] for name, _ in sorted_results]
        accuracies = [result['accuracy'] for _, result in sorted_results]
        
        bars = plt.bar(model_names, accuracies, color='skyblue', alpha=0.8)
        plt.title('Deep Learning Models Comparison - Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{config['output']['results_dir']}/model_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll models completed successfully!")
    print(f"Best model: {sorted_results[0][1]['model_name']} ({sorted_results[0][1]['accuracy']:.4f})")

if __name__ == "__main__":
    main()