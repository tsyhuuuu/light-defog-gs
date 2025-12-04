#!/usr/bin/env python3
"""
Support Vector Classifier for Fog Gaussian Classification
Based on fog_removal.ipynb notebook
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(csv_path, test_size=0.2):
    """Load and prepare data for classification."""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Feature columns (exclude position coordinates, target, and metadata)
    feature_columns = [col for col in df.columns 
                      if col not in ['beta', 'alpha', 'is_fog', 'pos_x', 'pos_y', 'pos_z', 
                                   'rot_0', 'rot_1', 'rot_2', 'rot_3', 'id']]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Using {len(feature_columns)} features")
    print(f"Class distribution: {df['is_fog'].value_counts().to_dict()}")
    
    X = df[feature_columns]
    y = df['is_fog']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    """Train SVM classifier."""
    print(f"\nTraining SVM with kernel={kernel}, C={C}, gamma={gamma}")
    
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm_model.fit(X_train, y_train)
    
    return svm_model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance."""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    return y_test_pred, test_acc, cm

def create_visualizations(y_test, y_pred, accuracy, cm, output_dir='results'):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'SVM Confusion Matrix\n(Accuracy: {accuracy:.4f})')
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
    plt.savefig(f'{output_dir}/svm_results.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {output_dir}/svm_results.png")
    plt.close()

def save_model(model, scaler, feature_columns, output_dir='models'):
    """Save trained model and preprocessing objects."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f'{output_dir}/svm_model.pkl'
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = f'{output_dir}/svm_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    # Save feature columns
    features_path = f'{output_dir}/svm_features.pkl'
    joblib.dump(feature_columns, features_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Features saved to {features_path}")

def load_model(model_dir='models'):
    """Load trained model and preprocessing objects."""
    model = joblib.load(f'{model_dir}/svm_model.pkl')
    scaler = joblib.load(f'{model_dir}/svm_scaler.pkl')
    features = joblib.load(f'{model_dir}/svm_features.pkl')
    
    return model, scaler, features

def predict_fog(model, scaler, features, new_data):
    """Predict fog classification for new data."""
    # Prepare features
    X_new = new_data[features]
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    predictions = model.predict(X_new_scaled)
    
    return predictions

def main():
    # Configuration
    CSV_PATH = "data/dataset_truck_beta6_alpha250.csv"
    OUTPUT_DIR = "results/svm"
    MODEL_DIR = "models/svm"
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(CSV_PATH)
    
    # Train model
    model = train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale')
    
    # Evaluate model
    y_pred, accuracy, cm = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Create visualizations
    create_visualizations(y_test, y_pred, accuracy, cm, OUTPUT_DIR)
    
    # Save model
    save_model(model, scaler, features, MODEL_DIR)
    
    print(f"\nSVM Classification completed successfully!")
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()