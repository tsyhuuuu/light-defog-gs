#!/usr/bin/env python3
"""
Feature Selector for Fog Gaussian Classification
Handles various feature selection strategies based on YAML configuration
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, f_classif, 
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import logging

class FeatureSelector:
    """Handles feature selection based on configuration."""
    
    def __init__(self, config: Dict, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.logger.propagate = False 
        self.selected_features = None
        self.feature_scores = None
        
    def select_features(self, df: pd.DataFrame, target_column: str = 'is_fog') -> List[str]:
        """Select features based on configuration."""
        feature_config = self.config.get('dataset', {}).get('features', {})
        selection_mode = feature_config.get('selection_mode', 'auto')
        
        self.logger.info(f"Feature selection mode: {selection_mode}")
        
        # Get all available features (excluding target and metadata)
        all_features = self._get_all_available_features(df)
        
        if selection_mode == 'auto':
            selected_features = self._auto_feature_selection(df, all_features)
        elif selection_mode == 'manual':
            selected_features = self._manual_feature_selection(df, feature_config)
        elif selection_mode == 'exclude':
            selected_features = self._exclude_feature_selection(df, feature_config)
        elif selection_mode == 'statistical':
            selected_features = self._statistical_feature_selection(df, feature_config, target_column)
        elif selection_mode == 'categories':
            selected_features = self._category_feature_selection(df, feature_config)
        else:
            raise ValueError(f"Unknown feature selection mode: {selection_mode}")
        
        # Validate selected features
        if feature_config.get('feature_selection_validation', True):
            selected_features = self._validate_features(df, selected_features)
        
        self.selected_features = selected_features
        self.logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        
        return selected_features
    
    def _get_all_available_features(self, df: pd.DataFrame) -> List[str]:
        """Get all available feature columns from dataframe."""
        # Always exclude these columns
        always_exclude = ['beta', 'alpha', 'is_fog', 'id', 'source_dataset']
        
        available_features = [col for col in df.columns if col not in always_exclude]
        return available_features
    
    def _auto_feature_selection(self, df: pd.DataFrame, all_features: List[str]) -> List[str]:
        """Automatic feature selection (excludes positions and rotations by default)."""
        default_excludes = ['pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        selected = [feat for feat in all_features if feat not in default_excludes]
        
        self.logger.info("Using automatic feature selection (excluding position and rotation)")
        return selected
    
    def _manual_feature_selection(self, df: pd.DataFrame, feature_config: Dict) -> List[str]:
        """Manual feature selection from specified list."""
        manual_features = feature_config.get('manual_features', [])
        
        if not manual_features:
            self.logger.warning("Manual feature selection specified but no features listed. Using auto mode.")
            return self._auto_feature_selection(df, self._get_all_available_features(df))
        
        self.logger.info(f"Using manual feature selection: {len(manual_features)} features specified")
        return manual_features
    
    def _exclude_feature_selection(self, df: pd.DataFrame, feature_config: Dict) -> List[str]:
        """Feature selection by exclusion."""
        all_features = self._get_all_available_features(df)
        exclude_features = feature_config.get('exclude_features', [])
        
        selected = [feat for feat in all_features if feat not in exclude_features]
        
        self.logger.info(f"Using exclusion-based selection: excluded {len(exclude_features)} features")
        return selected
    
    def _statistical_feature_selection(self, df: pd.DataFrame, feature_config: Dict, 
                                     target_column: str) -> List[str]:
        """Statistical feature selection using various methods."""
        stat_config = feature_config.get('statistical_selection', {})
        method = stat_config.get('method', 'mutual_info')
        k_best = stat_config.get('k_best', 20)
        threshold = stat_config.get('threshold', 0.01)
        
        all_features = self._get_all_available_features(df)
        
        # Prepare data
        X = df[all_features].fillna(0)  # Handle any missing values
        y = df[target_column]
        
        self.logger.info(f"Using statistical feature selection: {method}")
        
        try:
            if method == 'mutual_info':
                # Mutual information
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, len(all_features)))
                X_selected = selector.fit_transform(X, y)
                selected_features = [all_features[i] for i in selector.get_support(indices=True)]
                self.feature_scores = dict(zip(all_features, selector.scores_))
                
            elif method == 'chi2':
                # Chi-squared test (requires non-negative features)
                X_positive = X - X.min() + 1e-8  # Make features positive
                selector = SelectKBest(score_func=chi2, k=min(k_best, len(all_features)))
                X_selected = selector.fit_transform(X_positive, y)
                selected_features = [all_features[i] for i in selector.get_support(indices=True)]
                self.feature_scores = dict(zip(all_features, selector.scores_))
                
            elif method == 'f_classif':
                # F-statistic
                selector = SelectKBest(score_func=f_classif, k=min(k_best, len(all_features)))
                X_selected = selector.fit_transform(X, y)
                selected_features = [all_features[i] for i in selector.get_support(indices=True)]
                self.feature_scores = dict(zip(all_features, selector.scores_))
                
            elif method == 'variance':
                # Variance threshold
                selector = VarianceThreshold(threshold=threshold)
                X_selected = selector.fit_transform(X)
                selected_features = [all_features[i] for i in selector.get_support(indices=True)]
                # Take top k_best features by variance
                if len(selected_features) > k_best:
                    variances = X[selected_features].var()
                    top_features = variances.nlargest(k_best).index.tolist()
                    selected_features = top_features
                self.feature_scores = dict(zip(selected_features, X[selected_features].var()))
                
            elif method == 'random_forest':
                # Random Forest feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                
                # Get feature importances
                importances = rf.feature_importances_
                feature_importance = list(zip(all_features, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Select top k features
                selected_features = [feat for feat, _ in feature_importance[:k_best]]
                self.feature_scores = dict(feature_importance)
                
            else:
                raise ValueError(f"Unknown statistical method: {method}")
                
        except Exception as e:
            self.logger.error(f"Statistical feature selection failed: {e}")
            self.logger.info("Falling back to auto feature selection")
            return self._auto_feature_selection(df, all_features)
        
        # Log feature scores if available
        if self.feature_scores and self.logger.level <= logging.INFO:
            self.logger.info("Top 10 feature scores:")
            sorted_scores = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
            for feat, score in sorted_scores[:10]:
                self.logger.info(f"  {feat}: {score:.4f}")
        
        return selected_features
    
    def _category_feature_selection(self, df: pd.DataFrame, feature_config: Dict) -> List[str]:
        """Feature selection based on predefined categories."""
        categories = feature_config.get('use_categories', [])
        feature_categories = feature_config.get('feature_categories', {})
        
        if not categories:
            self.logger.warning("Category-based selection specified but no categories listed. Using auto mode.")
            return self._auto_feature_selection(df, self._get_all_available_features(df))
        
        selected_features = []
        for category in categories:
            if category in feature_categories:
                selected_features.extend(feature_categories[category])
                self.logger.info(f"Added {len(feature_categories[category])} features from category '{category}'")
            else:
                self.logger.warning(f"Unknown feature category: {category}")
        
        # Remove duplicates while preserving order
        selected_features = list(dict.fromkeys(selected_features))
        
        self.logger.info(f"Using category-based selection: {len(selected_features)} features from {len(categories)} categories")
        return selected_features
    
    def _validate_features(self, df: pd.DataFrame, selected_features: List[str]) -> List[str]:
        """Validate that selected features exist in the dataframe."""
        available_features = list(df.columns)
        valid_features = []
        missing_features = []
        
        for feature in selected_features:
            if feature in available_features:
                valid_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            self.logger.warning(f"Missing features (will be skipped): {missing_features}")
        
        if not valid_features:
            self.logger.error("No valid features found! Using auto selection as fallback.")
            return self._auto_feature_selection(df, self._get_all_available_features(df))
        
        self.logger.info(f"Feature validation: {len(valid_features)} valid, {len(missing_features)} missing")
        return valid_features
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get a summary of feature importance if available."""
        if not self.feature_scores:
            return pd.DataFrame()
        
        df_importance = pd.DataFrame([
            {'feature': feat, 'score': score} 
            for feat, score in self.feature_scores.items()
        ]).sort_values('score', ascending=False)
        
        return df_importance
    
    def save_feature_info(self, output_path: str):
        """Save feature selection information to file."""
        import json
        
        info = {
            'selected_features': self.selected_features,
            'feature_count': len(self.selected_features) if self.selected_features else 0,
            'feature_scores': self.feature_scores,
            'selection_config': self.config.get('dataset', {}).get('features', {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        self.logger.info(f"Feature selection info saved to {output_path}")

def main():
    """Example usage of FeatureSelector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Selector Test')
    parser.add_argument('--data-file', required=True, help='CSV file to test feature selection on')
    parser.add_argument('--config-file', default='config/experiment_config.yaml', 
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_csv(args.data_file)
    
    # Test feature selection
    selector = FeatureSelector(config)
    selected_features = selector.select_features(df)
    
    print(f"\nSelected Features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i:2d}. {feat}")
    
    # Show feature importance if available
    importance_df = selector.get_feature_importance_summary()
    if not importance_df.empty:
        print(f"\nTop 10 Feature Scores:")
        print(importance_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()