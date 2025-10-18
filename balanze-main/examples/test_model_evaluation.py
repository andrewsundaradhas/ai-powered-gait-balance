"""
Test Script for Model Evaluation Module

This script demonstrates how to use the ClinicalGaitEvaluator class to evaluate
gait and balance analysis models with both synthetic and real data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.model_evaluation import ClinicalGaitEvaluator, ModelEvaluationResults

def generate_synthetic_data(task_type='classification', n_samples=1000, n_features=20, n_classes=3):
    """Generate synthetic data for testing.
    
    Args:
        task_type: 'classification' or 'regression'
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (for classification only)
        
    Returns:
        Tuple of (X, y, feature_names, class_names)
    """
    np.random.seed(42)
    
    # Generate feature names that look like gait/balance metrics
    feature_names = [
        f"stride_length_{i}" for i in range(n_features // 4)
    ] + [
        f"step_width_{i}" for i in range(n_features // 4)
    ] + [
        f"swing_time_{i}" for i in range(n_features // 4)
    ] + [
        f"stance_time_{i}" for i in range(n_features - 3 * (n_features // 4))
    ]
    
    if task_type == 'classification':
        # Generate classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(10, n_features // 2),
            n_redundant=min(5, n_features // 4),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create meaningful class names
        if n_classes == 2:
            class_names = ["Normal Gait", "Abnormal Gait"]
        elif n_classes == 3:
            class_names = ["Mild Impairment", "Moderate Impairment", "Severe Impairment"]
        else:
            class_names = [f"Class_{i}" for i in range(n_classes)]
            
    else:  # regression
        # Generate regression data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(10, n_features // 2),
            noise=10.0,
            random_state=42
        )
        
        # Make y positive (like gait speed in m/s)
        y = np.abs(y) / 10.0
        class_names = None  # Not used for regression
    
    return X, y, feature_names, class_names

def test_classification():
    """Test the evaluator with a classification model."""
    print("=== Testing Classification Model Evaluation ===")
    
    # Generate synthetic classification data
    X, y, feature_names, class_names = generate_synthetic_data(
        task_type='classification',
        n_samples=1000,
        n_features=20,
        n_classes=3
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train a simple classifier
    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = ClinicalGaitEvaluator(
        model=clf,
        feature_names=feature_names,
        class_names=class_names,
        is_time_series=True,
        target_variable="gait_impairment"
    )
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
    
    # Get feature importances if available
    feature_importances = None
    if hasattr(clf, 'feature_importances_'):
        feature_importances = {
            feature_names[i]: clf.feature_importances_[i] 
            for i in range(len(feature_names))
        }
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_importances=feature_importances,
        generate_shap=True
    )
    
    # Generate and print report
    print("\n=== Evaluation Report ===")
    report = evaluator.generate_report(results)
    print(report)
    
    # Save plots
    os.makedirs("evaluation_plots/classification", exist_ok=True)
    for name, fig in results.plots.items():
        fig.savefig(f"evaluation_plots/classification/{name}.png")
        plt.close(fig)
    
    print("\nPlots saved to 'evaluation_plots/classification/'")
    print("=== Classification Test Complete ===\n")

def test_regression():
    """Test the evaluator with synthetic regression data."""
    print("\n=== Testing with Synthetic Regression Data ===")
    
    # Generate synthetic regression data
    X, y, feature_names, _ = generate_synthetic_data(task_type='regression')
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a simple regression model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create feature importances dictionary
    feature_importances = {
        f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)
    }
    
    # Initialize the evaluator for regression
    evaluator = ClinicalGaitEvaluator(
        model=model,
        feature_names=feature_names,  # Use the actual feature names
        class_names=None,  # None for regression
        is_time_series=False,
        target_variable="gait_metric"
    )
    
    # Get feature importances as a dictionary with actual feature names
    feature_importances = {
        feature_names[i]: model.feature_importances_[i] 
        for i in range(len(feature_names))
    }
    
    # Evaluate the model
    results = evaluator.evaluate(
        X_test, 
        y_test,
        y_pred,
        feature_importances=feature_importances,
        generate_shap=True
    )
    
    # Print results
    print("\nRegression Results:")
    print(f"MAE: {results.metrics.get('mae', 'N/A'):.4f}")
    print(f"MSE: {results.metrics.get('mse', 'N/A'):.4f}")
    print(f"R²: {results.metrics.get('r2', 'N/A'):.4f}")
    
    # Print the report
    print("\n=== Evaluation Report ===")
    print(results.classification_report)
    
    # Save the report
    report_path = os.path.join("reports", "regression_evaluation_report.md")
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(results.classification_report)
    print(f"\nRegression evaluation report saved to: {os.path.abspath(report_path)}")
    
    # Save plots
    os.makedirs("evaluation_plots/regression", exist_ok=True)
    for name, fig in results.plots.items():
        fig.savefig(f"evaluation_plots/regression/{name}.png")
        plt.close(fig)
    
    print("\nPlots saved to 'evaluation_plots/regression/'")
    print("=== Regression Test Complete ===\n")

def test_with_real_data():
    """Test the evaluator with real gait data if available."""
    print("=== Testing with Real Gait Data ===")
    
    try:
        # Try to load the dataset
        from src.data.dataset_loader import load_gait_dataset
        
        print("Loading real gait dataset...")
        X, y, feature_names, class_names = load_gait_dataset()
        
        # If we get here, we have real data
        print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train a model
        print("Training model...")
        if len(np.unique(y)) > 10:  # Regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            is_classification = False
        else:  # Classification
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            is_classification = True
            
        model.fit(X_train_scaled, y_train)
        
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = ClinicalGaitEvaluator(
            model=model,
            feature_names=feature_names,
            class_names=class_names if is_classification else None,
            is_time_series=True,
            target_variable="gait_parameter"
        )
        
        # Run evaluation
        print("Running evaluation...")
        results = evaluator.evaluate(
            X_test=X_test_scaled,
            y_test=y_test,
            X_train=X_train_scaled,
            calculate_shap=True,
            shap_sample_size=min(100, len(X_test_scaled))
        )
        
        # Generate and print report
        print("\n=== Evaluation Report ===")
        report = evaluator.generate_report(results)
        print(report)
        
        # Save plots
        output_dir = "evaluation_plots/real_data"
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in results.plots.items():
            if not is_classification and name == "roc_curve":  # Skip ROC for regression
                continue
            fig.savefig(f"{output_dir}/{name}.png")
            plt.close(fig)
        
        print(f"\nPlots saved to '{output_dir}/'")
        
    except (ImportError, FileNotFoundError) as e:
        print(f"Could not load real dataset: {e}")
        print("Skipping real data test. Make sure the dataset is available.")
    
    print("=== Real Data Test Complete ===\n")

if __name__ == "__main__":
    # Run classification test
    test_classification()
    
    # Run regression test
    test_regression()
    
    # Test with real data if available
    test_with_real_data()
    
    print("All tests completed!")
