"""
Enhanced Model Evaluation for Gait and Balance Analysis

This module provides comprehensive evaluation metrics and visualizations
specifically designed for clinical gait and balance analysis models.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score, KFold
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """Supported evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    COHEN_KAPPA = "cohen_kappa"
    MATTHEWS_CORR = "matthews_corr"

@dataclass
class ModelEvaluationResults:
    """Container for model evaluation results."""
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importances: Optional[Dict[str, float]] = None
    shap_values: Optional[np.ndarray] = None
    time_series_metrics: Optional[Dict[str, float]] = None
    clinical_metrics: Optional[Dict[str, float]] = None
    plots: Optional[Dict[str, Any]] = None

class ClinicalGaitEvaluator:
    """Comprehensive evaluator for gait and balance analysis models."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        is_time_series: bool = True,
        target_variable: str = "gait_parameter"
    ):
        """Initialize the evaluator.
        
        Args:
            model: Trained model (sklearn, xgboost, or keras)
            feature_names: List of feature names
            class_names: List of class names for classification
            is_time_series: Whether the data is time-series
            target_variable: Name of the target variable
        """
        self.model = model
        self.feature_names = feature_names
        # Only set class_names for classification tasks
        if class_names is not None:
            self.class_names = class_names
        else:
            # For regression tasks, we don't need class names
            self.class_names = None
        self.is_time_series = is_time_series
        self.target_variable = target_variable
        self.explainer = None
        
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        shap_values: Optional[np.ndarray] = None,
        generate_shap: bool = False,
        explainer_type: str = "tree"
    ) -> ModelEvaluationResults:
        """
        Evaluate the model and generate comprehensive evaluation metrics and plots.
        
        Args:
            X_test: Test features
            y_test: True labels or values
            y_pred: Predicted labels or values
            y_prob: Predicted probabilities (for classification)
            sample_weight: Sample weights
            feature_importances: Dictionary of feature importances
            shap_values: Pre-computed SHAP values (optional)
            generate_shap: Whether to generate SHAP values
            explainer_type: Type of SHAP explainer to use ("tree", "kernel", or "deep")
            
        Returns:
            ModelEvaluationResults: Results of the evaluation
        """
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob, sample_weight)
        clinical_metrics = self._calculate_clinical_metrics(X_test, y_test, y_pred)
        
        # Generate plots
        plots = self._generate_plots(
            y_test, y_pred, y_prob, feature_importances, shap_values
        )
        
        # Store X_test as instance variable for SHAP plotting
        self.X_test = X_test
        
        # Generate SHAP values if requested and not provided
        if generate_shap and shap_values is None:
            try:
                shap_values = self._calculate_shap_values(X_test, explainer_type)
                
                # Add SHAP summary plot
                if shap_values is not None and hasattr(self, 'explainer') and hasattr(self, 'X_test'):
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values, 
                            self.X_test, 
                            feature_names=self.feature_names,
                            show=False
                        )
                        plt.tight_layout()
                        plots["shap_summary"] = fig
                    except Exception as e:
                        warnings.warn(f"Could not generate SHAP summary plot: {str(e)}")
            except Exception as e:
                warnings.warn(f"Failed to generate SHAP values: {str(e)}")
        
        # Only calculate confusion matrix for classification tasks
        cm = None
        if self.class_names is not None:  # Classification task
            try:
                cm = confusion_matrix(y_test, y_pred)
            except Exception as e:
                warnings.warn(f"Could not generate confusion matrix: {str(e)}")
        
        # Generate classification report for classification tasks
        classification_report_str = ""
        if self.class_names is not None:  # Classification task
            try:
                from sklearn.metrics import classification_report as sk_classification_report
                classification_report_str = sk_classification_report(
                    y_test, y_pred, target_names=self.class_names
                )
            except Exception as e:
                warnings.warn(f"Could not generate classification report: {str(e)}")
        
        # Create results object
        results = ModelEvaluationResults(
            metrics=metrics,
            clinical_metrics=clinical_metrics or {},
            feature_importances=feature_importances or {},
            plots=plots or {},
            classification_report=classification_report_str,
            confusion_matrix=cm,
            shap_values=shap_values
        )
        
        # Generate the detailed report
        results.classification_report = self.generate_report(results)
        
        return results
    
    def _predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get model predictions with optional thresholding."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Handle binary and multiclass cases
            if probs.shape[1] == 2:  # Binary classification
                return (probs[:, 1] >= threshold).astype(int)
            else:  # Multiclass
                return np.argmax(probs, axis=1)
        elif hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            raise ValueError("Model must have either predict_proba or predict method")
    
    def _predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get predicted probabilities if available."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            return probs[:, 1] if probs.shape[1] == 2 else probs
        return None
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate standard evaluation metrics.
        
        Handles both classification and regression tasks based on the input data.
        """
        metrics = {}
        
        # Check if this is a classification or regression task
        is_classification = self.class_names is not None and len(np.unique(y_true)) <= len(self.class_names or [])
        
        if is_classification:
            # Classification metrics
            metrics.update({
                "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            })
            
            # Add ROC AUC if probabilities are available and it's a binary classification
            if y_prob is not None and len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            # Regression metrics
            metrics.update({
                "mae": mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
                "mse": mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
                "r2": r2_score(y_true, y_pred, sample_weight=sample_weight)
            })
        
        return metrics
    
    def _calculate_time_series_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate time-series specific metrics."""
        # Calculate autocorrelation of residuals
        residuals = y_true - y_pred
        acf = np.correlate(residuals, residuals, mode='full')[-len(residuals):]
        acf = acf / np.max(acf)  # Normalize
        
        # Time-series specific metrics
        metrics = {
            "residual_autocorr": np.mean(np.abs(acf[1:])),  # Mean absolute autocorrelation
            "residual_std": np.std(residuals),
            "residual_skew": stats.skew(residuals),
            "residual_kurtosis": stats.kurtosis(residuals)
        }
        
        return metrics
    
    def _calculate_clinical_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clinically relevant metrics.
        
        Handles both classification and regression tasks.
        For regression, returns an empty dictionary as clinical metrics
        are not applicable in the same way as for classification.
        """
        # For regression tasks, return an empty dictionary
        if self.class_names is None:
            return {}
            
        # For classification tasks, calculate standard clinical metrics
        try:
            cm = confusion_matrix(y_true, y_pred)
            sensitivity = []
            specificity = []
            
            for i in range(len(cm)):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - (tp + fn + fp)
                
                sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            
            # Calculate positive and negative predictive values
            ppv = []
            npv = []
            
            for i in range(len(cm)):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                tn = np.sum(cm) - (tp + fn + fp)
                
                ppv.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                npv.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
            
            return {
                "sensitivity": np.mean(sensitivity),
                "specificity": np.mean(specificity),
                "ppv": np.mean(ppv),
                "npv": np.mean(npv)
            }
            
        except Exception as e:
            # If there's an error (e.g., in binary classification with single class),
            # return an empty dictionary
            print(f"Warning: Could not calculate clinical metrics: {str(e)}")
            return {}
    
    def _get_feature_importances(self, X: np.ndarray) -> Dict[str, float]:
        """Extract feature importances from the model."""
        if hasattr(self.model, "feature_importances_"):  # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):  # Linear models
            importances = np.abs(self.model.coef_)
            if len(importances.shape) > 1:  # Multiclass
                importances = np.mean(importances, axis=0)
        else:
            return {name: 0 for name in self.feature_names}
        
        # Normalize importances
        importances = importances / np.sum(importances)
        return dict(zip(self.feature_names, importances))
    
    def _calculate_shap_values(
        self,
        X: np.ndarray,
        X_train: np.ndarray,
        sample_size: int = 100
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for model interpretability."""
        try:
            # Sample data if too large
            if len(X) > sample_size:
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(len(X), sample_size, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            # Initialize explainer based on model type
            if "xgboost" in str(type(self.model)).lower():
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            elif "keras" in str(type(self.model)).lower():
                # For Keras models, use DeepExplainer or GradientExplainer
                if len(X_train) > 1000:
                    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
                else:
                    background = X_train
                explainer = shap.DeepExplainer(self.model, background)
                shap_values = explainer.shap_values(X_sample)
            else:
                # For other models, use KernelExplainer
                explainer = shap.KernelExplainer(self.model.predict, X_train[:100])
                shap_values = explainer.shap_values(X_sample)
            
            self.explainer = explainer
            return shap_values
            
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values: {str(e)}")
            return None
    
    def _generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        shap_values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate evaluation plots.
        
        Handles both classification and regression tasks.
        """
        plots = {}
        is_classification = self.class_names is not None
        
        if is_classification:
            # Classification Plots
            
            # Confusion Matrix
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                plots["confusion_matrix"] = fig
            except Exception as e:
                print(f"Warning: Could not generate confusion matrix: {str(e)}")
            
            # ROC Curve (for binary classification)
            if y_prob is not None and len(np.unique(y_true)) == 2:
                try:
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc="lower right")
                    plots["roc_curve"] = fig
                except Exception as e:
                    print(f"Warning: Could not generate ROC curve: {str(e)}")
        else:
            # Regression Plots
            
            # Actual vs Predicted Scatter Plot
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_true, y_pred, alpha=0.5)
                # Add a diagonal line for perfect predictions
                min_val = min(np.min(y_true), np.min(y_pred))
                max_val = max(np.max(y_true), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Actual vs Predicted Values')
                plots["actual_vs_predicted"] = fig
            except Exception as e:
                print(f"Warning: Could not generate actual vs predicted plot: {str(e)}")
            
            # Residual Plot
            try:
                residuals = y_true - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred, residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot')
                plots["residual_plot"] = fig
            except Exception as e:
                print(f"Warning: Could not generate residual plot: {str(e)}")
        
        # Feature Importance (for both classification and regression)
        if feature_importances:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                sorted_importances = dict(sorted(feature_importances.items(), 
                                              key=lambda x: x[1], reverse=True)[:10])
                sns.barplot(x=list(sorted_importances.values()), 
                          y=list(sorted_importances.keys()), 
                          ax=ax)
                plot_type = 'Classification' if is_classification else 'Regression'
                ax.set_title(f'Top 10 Feature Importances ({plot_type})')
                ax.set_xlabel('Importance')
                plots["feature_importance"] = fig
            except Exception as e:
                print(f"Warning: Could not generate feature importance plot: {str(e)}")
        
        # SHAP Summary Plot
        if shap_values is not None and self.explainer is not None:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    X_test, 
                    feature_names=self.feature_names,
                    show=False
                )
                plt.tight_layout()
                plots["shap_summary"] = fig
            except Exception as e:
                logger.warning(f"Failed to generate SHAP summary plot: {str(e)}")
        
        return plots
    
    def generate_report(self, results: ModelEvaluationResults) -> str:
        """Generate a comprehensive evaluation report."""
        report = [
            "# Model Evaluation Report",
            "## Performance Metrics"
        ]
        
        # Add classification metrics if available
        if any(k in results.metrics for k in ['accuracy', 'precision', 'recall', 'f1']):
            report.append("### Classification Metrics")
            if 'accuracy' in results.metrics:
                report.append(f"- Accuracy: {results.metrics['accuracy']:.4f}")
            if 'precision' in results.metrics:
                report.append(f"- Precision: {results.metrics['precision']:.4f}")
            if 'recall' in results.metrics:
                report.append(f"- Recall: {results.metrics['recall']:.4f}")
            if 'f1' in results.metrics:
                report.append(f"- F1 Score: {results.metrics['f1']:.4f}")
            if 'roc_auc' in results.metrics:
                report.append(f"- ROC AUC: {results.metrics['roc_auc']:.4f}")
            report.append("")  # Add empty line
            
        # Add regression metrics if available
        if any(k in results.metrics for k in ['mae', 'mse', 'rmse', 'r2']):
            report.append("### Regression Metrics")
            if 'mae' in results.metrics:
                report.append(f"- MAE: {results.metrics['mae']:.4f}")
            if 'mse' in results.metrics:
                report.append(f"- MSE: {results.metrics['mse']:.4f}")
            if 'rmse' in results.metrics:
                report.append(f"- RMSE: {results.metrics['rmse']:.4f}")
            if 'r2' in results.metrics:
                report.append(f"- R²: {results.metrics['r2']:.4f}")
            report.append("")  # Add empty line
            
        # Add clinical metrics if available
        if results.clinical_metrics:
            report.append("### Clinical Metrics")
            if 'sensitivity' in results.clinical_metrics:
                report.append(f"- Sensitivity: {results.clinical_metrics['sensitivity']:.4f}")
            if 'specificity' in results.clinical_metrics:
                report.append(f"- Specificity: {results.clinical_metrics['specificity']:.4f}")
            if 'ppv' in results.clinical_metrics:
                report.append(f"- PPV: {results.clinical_metrics['ppv']:.4f}")
            if 'npv' in results.clinical_metrics:
                report.append(f"- NPV: {results.clinical_metrics['npv']:.4f}")
            report.append("")  # Add empty line
            
        # Add feature importance section
        report.append("## Feature Importance")
        report.append("Top 5 most important features:")
        
        # Add top features
        if results.feature_importances:
            sorted_features = sorted(
                results.feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            for feature, importance in sorted_features:
                report.append(f"- {feature}: {importance:.4f}")
        
        # Add recommendations
        report.extend([
            "",
            "## Recommendations",
            "1. Review the confusion matrix to understand model errors",
            "2. Check feature importances for model interpretability",
            "3. Consider SHAP values for understanding individual predictions",
            "4. Validate with clinical experts for clinical relevance"
        ])
        
        return "\n".join(report)
