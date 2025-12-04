#!/usr/bin/env python3
"""
LightGBM Classifier for Fog Gaussian Classification
Based on fog_removal.ipynb notebook
"""

import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
    
    return X_train, X_test, y_train, y_test, feature_columns

def train_lightgbm(X_train, y_train, 
                   n_estimators=200, 
                   learning_rate=0.05, 
                   max_depth=-1, 
                   num_leaves=31,
                   subsample=0.8,
                   colsample_bytree=0.8,
                   verbose=-1):
    """Train LightGBM classifier."""
    print(f"\nTraining LightGBM...")
    print(f"Parameters: n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}")
    
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1,
        verbose=verbose
    )
    
    lgbm_model.fit(X_train, y_train)
    
    return lgbm_model

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

def plot_feature_importance(model, feature_names, top_k=20, output_dir='results'):
    """Plot feature importance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top K features
    plt.figure(figsize=(10, 8))
    top_features = feature_imp.head(top_k)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_k} Most Important Features - LightGBM')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {output_dir}/lightgbm_feature_importance.png")
    plt.close()
    
    return feature_imp

def create_visualizations(y_test, y_pred, accuracy, cm, output_dir='results'):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
    axes[0].set_title(f'LightGBM Confusion Matrix\n(Accuracy: {accuracy:.4f})')
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
    plt.savefig(f'{output_dir}/lightgbm_results.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {output_dir}/lightgbm_results.png")
    plt.close()

def save_model(model, feature_columns, output_dir='models'):
    """Save trained model and feature information."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f'{output_dir}/lightgbm_model.pkl'
    joblib.dump(model, model_path)
    
    # Save feature columns
    features_path = f'{output_dir}/lightgbm_features.pkl'
    joblib.dump(feature_columns, features_path)
    
    print(f"Model saved to {model_path}")
    print(f"Features saved to {features_path}")

def load_model(model_dir='models'):
    """Load trained model and feature information."""
    model = joblib.load(f'{model_dir}/lightgbm_model.pkl')
    features = joblib.load(f'{model_dir}/lightgbm_features.pkl')
    
    return model, features

def predict_fog(model, features, new_data):
    """Predict fog classification for new data."""
    # Prepare features
    X_new = new_data[features]
    
    # Predict
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    return predictions, probabilities

def hyperparameter_search(X_train, X_test, y_train, y_test):
    """Simple grid search for hyperparameter tuning."""
    print("\nPerforming hyperparameter search...")
    
    param_grid = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31},
        {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31},
        {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 63},
        {'n_estimators': 200, 'learning_rate': 0.1, 'num_leaves': 63},
    ]
    
    best_score = 0
    best_params = None
    best_model = None
    
    for params in param_grid:
        model = lgb.LGBMClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        print(f"Params: {params} | Test Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    
    return best_model, best_params

def main():
    # Configuration
    CSV_PATH = "data/dataset_truck_beta6_alpha250.csv"
    OUTPUT_DIR = "results/lightgbm"
    MODEL_DIR = "models/lightgbm"
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, features = prepare_data(CSV_PATH)
    
    # Option 1: Train with default parameters
    model = train_lightgbm(X_train, y_train)
    
    # Option 2: Hyperparameter search (uncomment to use)
    # model, best_params = hyperparameter_search(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    y_pred, accuracy, cm = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Plot feature importance
    feature_imp = plot_feature_importance(model, features, top_k=20, output_dir=OUTPUT_DIR)
    
    # Create visualizations
    create_visualizations(y_test, y_pred, accuracy, cm, OUTPUT_DIR)
    
    # Save model
    save_model(model, features, MODEL_DIR)
    
    print(f"\nLightGBM Classification completed successfully!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Print top 10 most important features
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_imp.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:30s} ({row['importance']:.4f})")

if __name__ == "__main__":
    main()