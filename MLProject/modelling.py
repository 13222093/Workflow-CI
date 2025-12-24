"""
MLflow Project - Ethereum Fraud Detection Model Training
Author: Mohammad Ari Alexander Aziz
Description: Automated training script for CI/CD pipeline using MLflow Projects
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(data_path='ethereum_fraud_preprocessing.csv',
                           test_size=0.2, apply_smote=True, random_state=42):
    """Load and prepare data for training"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")

    # Separate features and target
    X = df.drop('FLAG', axis=1)
    y = df['FLAG']

    print(f"\nDataset Info:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Total samples: {len(y)}")
    print(f"  Fraud cases: {y.sum()} ({y.mean()*100:.2f}%)")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling
    scaler = PowerTransformer()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = 'scaler_trained.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to: {scaler_path}")

    # Apply SMOTE
    if apply_smote:
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {X_train_scaled.shape[0]} training samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler_path


def create_confusion_matrix_plot(y_true, y_pred, filename='confusion_matrix.png'):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def create_feature_importance_plot(model, feature_names, filename='feature_importance.png'):
    """Create and save feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices],
                   rotation=90)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    return None


def train_model(X_train, X_test, y_train, y_test, feature_names,
                n_estimators=100, max_depth=6, learning_rate=0.1):
    """Train XGBoost model with MLflow tracking"""

    print("\n" + "="*70)
    print("Training XGBoost Model")
    print("="*70)

    # Log parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    mlflow.log_param("smote_applied", "True")

    # Train model
    print(f"\nTraining with parameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"\nModel Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Create and log confusion matrix
    cm_file = create_confusion_matrix_plot(y_test, y_pred)
    mlflow.log_artifact(cm_file)
    print(f"\nConfusion matrix saved: {cm_file}")

    # Create and log feature importance
    fi_file = create_feature_importance_plot(model, feature_names)
    if fi_file:
        mlflow.log_artifact(fi_file)
        print(f"Feature importance saved: {fi_file}")

    # Log model
    mlflow.xgboost.log_model(model, "xgboost_model")

    # Save model locally
    model_path = 'xgboost_fraud_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    mlflow.log_artifact(model_path)
    print(f"Model saved: {model_path}")

    return model, model_path


def main():
    """Main training pipeline"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Ethereum Fraud Detection Model')
    parser.add_argument('--data-path', type=str, default='ethereum_fraud_preprocessing.csv',
                        help='Path to preprocessed data')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum depth of trees')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')

    args = parser.parse_args()

    print("="*70)
    print("Ethereum Fraud Detection - MLflow Project Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  N estimators: {args.n_estimators}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Test size: {args.test_size}")
    print(f"  Random state: {args.random_state}")

    # Set experiment
    experiment_name = "Ethereum_Fraud_CI_CD"
    mlflow.set_experiment(experiment_name)
    print(f"\nMLflow Experiment: {experiment_name}")

    # Start MLflow run
    with mlflow.start_run(run_name="CI_CD_Training"):
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names, scaler_path = load_and_prepare_data(
            data_path=args.data_path,
            test_size=args.test_size,
            apply_smote=True,
            random_state=args.random_state
        )

        # Log scaler
        mlflow.log_artifact(scaler_path)

        # Train model
        model, model_path = train_model(
            X_train, X_test, y_train, y_test, feature_names,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate
        )

        print("\n" + "="*70)
        print("Training Completed Successfully!")
        print("="*70)
        print(f"\nArtifacts:")
        print(f"  - Model: {model_path}")
        print(f"  - Scaler: {scaler_path}")
        print(f"  - Confusion Matrix: confusion_matrix.png")
        print(f"  - Feature Importance: feature_importance.png")
        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
