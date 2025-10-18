"""
Gait Analysis Model Training Pipeline

This module provides functionality to train and evaluate machine learning models
for gait and balance analysis using motion capture data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    FunctionTransformer
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    make_scorer
)
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb

# Custom transformer for feature engineering
class GaitFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from raw gait data."""
    
    def __init__(self):
        self.feature_names = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Extract features from raw gait data.
        
        Args:
            X: DataFrame containing raw gait data
            
        Returns:
            DataFrame with extracted features
        """
        # Create a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Example feature engineering steps
        # 1. Temporal features
        if 'timestamp' in X_transformed.columns:
            X_transformed['timestamp'] = pd.to_datetime(X_transformed['timestamp'])
            X_transformed['hour'] = X_transformed['timestamp'].dt.hour
            X_transformed['day_of_week'] = X_transformed['timestamp'].dt.dayofweek
            
        # 2. Kinematic features (example - adjust based on your data structure)
        if all(col in X_transformed.columns for col in ['left_ankle_x', 'right_ankle_x']):
            X_transformed['step_length'] = abs(X_transformed['left_ankle_x'] - X_transformed['right_ankle_x'])
            
        # 3. Statistical features
        if 'stride_time' in X_transformed.columns:
            X_transformed['stride_time_var'] = X_transformed['stride_time'].rolling(window=5).var()
            X_transformed['stride_time_mean'] = X_transformed['stride_time'].rolling(window=5).mean()
        
        # 4. Frequency domain features (placeholder - implement based on your needs)
        
        # Store feature names for reference
        self.feature_names = X_transformed.columns.tolist()
        
        return X_transformed

class GaitModelTrainer:
    """Train and evaluate gait analysis models."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the gait model trainer.
        
        Args:
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'stacking')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = GaitFeatureExtractor()
        self.training_metrics = {}
        
    def load_data(self, data_path: str, target_col: str = 'gait_type') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess gait data.
        
        Args:
            data_path: Path to the data file (CSV, JSON, or directory of files)
            target_col: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        # Handle different input formats
        data_path = Path(data_path)
        
        if data_path.is_file():
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix == '.json':
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        elif data_path.is_dir():
            # Load and combine multiple files
            data_files = list(data_path.glob('*.csv')) + list(data_path.glob('*.json'))
            dfs = []
            for file in data_files:
                if file.suffix == '.csv':
                    dfs.append(pd.read_csv(file))
                else:
                    dfs.append(pd.read_json(file))
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Basic data validation
        required_columns = [target_col, 'timestamp']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Drop rows with missing target values
        df = df.dropna(subset=[target_col])
        
        # Encode target variable
        y = self.label_encoder.fit_transform(df[target_col])
        X = df.drop(columns=[target_col])
        
        return X, y
    
    def preprocess_data(self, X: pd.DataFrame, y: np.ndarray = None, fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the input features and target.
        
        Args:
            X: Input features
            y: Target variable (optional)
            fit: Whether to fit the preprocessing steps
            
        Returns:
            Tuple of (preprocessed_X, preprocessed_y)
        """
        # Extract features
        X_features = self.feature_extractor.fit_transform(X)
        
        # Handle missing values
        X_features = X_features.fillna(X_features.median())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_features)
        else:
            X_scaled = self.scaler.transform(X_features)
            
        return X_scaled, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Optional[dict] = None) -> dict:
        """
        Train the selected model.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparams: Dictionary of hyperparameters
            
        Returns:
            Dictionary with training results and metrics
        """
        if hyperparams is None:
            hyperparams = {}
        
        # Define the model based on model_type
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', -1),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'stacking':
            # Define base models
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=self.random_state))
            ]
            
            # Define meta-learner
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        self.model = model
        
        # Evaluate on training set
        y_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred, average='weighted')
        
        # Store training metrics
        self.training_metrics = {
            'model_type': self.model_type,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'feature_importance': self._get_feature_importance(model, X_train.shape[1]) if hasattr(model, 'feature_importances_') else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.training_metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.evaluation_metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """
        Perform cross-validation on the training data.
        
        Args:
            X: Input features
            y: Target variable
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call train() first.")
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring=make_scorer(accuracy_score),
            n_jobs=-1
        )
        
        # Store CV results
        self.cv_metrics = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'cv_scores': cv_scores.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.cv_metrics
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model and related artifacts.
        
        Args:
            output_dir: Directory to save the model and artifacts
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = output_path / 'gait_model.joblib'
        joblib.dump(self.model, model_path)
        
        # Save the scaler
        scaler_path = output_path / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save the label encoder
        encoder_path = output_path / 'label_encoder.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature names
        feature_names_path = output_path / 'feature_names.json'
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_extractor.feature_names, f)
        
        # Save training metrics
        metrics_path = output_path / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Save evaluation metrics if available
        if hasattr(self, 'evaluation_metrics'):
            eval_metrics_path = output_path / 'evaluation_metrics.json'
            with open(eval_metrics_path, 'w') as f:
                json.dump(self.evaluation_metrics, f, indent=2)
    
    def load_model(self, model_dir: str) -> None:
        """
        Load a trained model and related artifacts.
        
        Args:
            model_dir: Directory containing the saved model and artifacts
        """
        model_dir = Path(model_dir)
        
        # Load the model
        model_path = model_dir / 'gait_model.joblib'
        self.model = joblib.load(model_path)
        
        # Load the scaler
        scaler_path = model_dir / 'scaler.joblib'
        self.scaler = joblib.load(scaler_path)
        
        # Load the label encoder
        encoder_path = model_dir / 'label_encoder.joblib'
        self.label_encoder = joblib.load(encoder_path)
        
        # Load feature names
        feature_names_path = model_dir / 'feature_names.json'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_extractor.feature_names = json.load(f)
    
    def _get_feature_importance(self, model, n_features: int) -> dict:
        """Extract and format feature importances."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Get feature names if available
            feature_names = getattr(self.feature_extractor, 'feature_names', 
                                  [f'feature_{i}' for i in range(n_features)])
            
            # Return top 20 most important features
            return {
                'importances': importances[indices].tolist(),
                'indices': indices.tolist(),
                'feature_names': [feature_names[i] for i in indices[:20]] if feature_names else []
            }
        return {}


def train_gait_model(data_path: str, output_dir: str = 'models', 
                    model_type: str = 'random_forest',
                    test_size: float = 0.2,
                    random_state: int = 42) -> dict:
    """
    Train a gait analysis model with the given data.
    
    Args:
        data_path: Path to the training data file or directory
        output_dir: Directory to save the trained model and artifacts
        model_type: Type of model to train ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'stacking')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with training and evaluation results
    """
    # Initialize the trainer
    trainer = GaitModelTrainer(model_type=model_type, random_state=random_state)
    
    # Load and preprocess the data
    X, y = trainer.load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Preprocess the data
    X_train_processed, y_train_processed = trainer.preprocess_data(X_train, y_train, fit=True)
    X_test_processed, y_test_processed = trainer.preprocess_data(X_test, y_test, fit=False)
    
    # Train the model
    train_metrics = trainer.train(X_train_processed, y_train_processed)
    
    # Evaluate on test set
    eval_metrics = trainer.evaluate(X_test_processed, y_test_processed)
    
    # Perform cross-validation
    cv_metrics = trainer.cross_validate(X_train_processed, y_train_processed)
    
    # Save the trained model and artifacts
    trainer.save_model(output_dir)
    
    # Combine all metrics
    results = {
        'training': train_metrics,
        'evaluation': eval_metrics,
        'cross_validation': cv_metrics,
        'model_type': model_type,
        'data_path': str(data_path),
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save the complete results
    results_path = Path(output_dir) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a gait analysis model')
    parser.add_argument('data_path', type=str, help='Path to the training data file or directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the trained model')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'stacking'],
                       help='Type of model to train')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Train the model
    results = train_gait_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Print summary
    print("\nTraining completed successfully!")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nTraining Accuracy: {results['training']['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['evaluation']['accuracy']:.4f}")
    print(f"Cross-validated Accuracy: {results['cross_validation']['mean_accuracy']:.4f} "
          f"(+/- {results['cross_validation']['std_accuracy']:.4f})")


