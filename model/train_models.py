"""
ML Classification Models Training Script
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository / sklearn
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, SVM

This script trains all 6 classification models with model-specific preprocessing,
evaluates them on test data, and saves the trained pipelines for deployment.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_breast_cancer_data():
    """
    Load the original Breast Cancer Wisconsin (Diagnostic) dataset.
    No augmentation - uses the dataset as-is from sklearn/UCI.
    569 instances, 30 features - meets assignment requirements (min 12 features, min 500 instances).
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"    Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"    Source: UCI Machine Learning Repository / sklearn")
    print(f"    Task: Binary Classification (Malignant=0 vs Benign=1)")
    print(f"    Note: Original dataset used without any augmentation")

    return df


def build_model_pipelines():
    """
    Build model-specific preprocessing pipelines.
    - StandardScaler for distance/gradient-based models (LR, KNN, SVM)
    - No scaling for tree-based models (Decision Tree, Random Forest)
    - No scaling for Naive Bayes (works on raw feature distributions)
    """
    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=5000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Naive Bayes': Pipeline([
            ('classifier', GaussianNB())
        ]),
        'Random Forest': Pipeline([
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
        ])
    }

    return pipelines


def train_and_evaluate_models(pipelines, X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate them."""

    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        preprocessing = "StandardScaler" if 'scaler' in pipeline.named_steps else "None (raw features)"
        print(f"Preprocessing: {preprocessing}")
        print(f"{'='*60}")

        # Train pipeline
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        # Note: Using weighted average for Precision, Recall, and F1
        # to account for class imbalance
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        results[name] = {
            'Accuracy': round(accuracy, 4),
            'AUC': round(auc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1, 4),
            'MCC': round(mcc, 4)
        }

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Precision: {precision:.4f} (weighted avg)")
        print(f"  Recall:    {recall:.4f} (weighted avg)")
        print(f"  F1 Score:  {f1:.4f} (weighted avg)")
        print(f"  MCC:       {mcc:.4f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")

        # Save pipeline (includes both scaler and model)
        model_filename = f"{name.lower().replace(' ', '_')}_pipeline.pkl"
        joblib.dump(pipeline, os.path.join(os.path.dirname(__file__), model_filename))
        print(f"  Pipeline saved as: {model_filename}")

    return results


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("   BREAST CANCER CLASSIFICATION - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Load data (original, no augmentation)
    print("\n[1] Loading original Breast Cancer Wisconsin dataset...")
    df = load_breast_cancer_data()
    print(f"    Dataset shape: {df.shape}")
    print(f"    Features: {df.shape[1] - 1}")
    print(f"    Instances: {df.shape[0]}")
    print(f"    Target distribution:")
    print(f"      Malignant (0): {(df['target'] == 0).sum()}")
    print(f"      Benign (1):    {(df['target'] == 1).sum()}")

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split (80-20), stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[2] Train-Test Split (80/20, stratified):")
    print(f"    Training set: {X_train.shape[0]} samples")
    print(f"    Test set: {X_test.shape[0]} samples")

    # Save test data for Streamlit app
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['target'] = y_test.values
    test_data_path = os.path.join(os.path.dirname(__file__), '..', 'test_data.csv')
    test_df.to_csv(test_data_path, index=False)
    print(f"\n[3] Test data saved to: test_data.csv ({len(test_df)} rows)")

    # Save feature names
    feature_names_path = os.path.join(os.path.dirname(__file__), 'feature_names.pkl')
    joblib.dump(list(X.columns), feature_names_path)

    # Build model pipelines (model-specific preprocessing)
    print(f"\n[4] Building model pipelines with model-specific preprocessing...")
    print(f"    - StandardScaler applied to: Logistic Regression, KNN, SVM")
    print(f"    - No scaling applied to: Decision Tree, Random Forest, Naive Bayes")
    pipelines = build_model_pipelines()

    # Train and evaluate all models
    print(f"\n[5] Training and evaluating all 6 models...")
    results = train_and_evaluate_models(pipelines, X_train, X_test, y_train, y_test)

    # Print comparison table
    print("\n\n" + "=" * 70)
    print("   MODEL COMPARISON TABLE")
    print("   (Precision, Recall, F1 use weighted average)")
    print("=" * 70)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # Find best model
    best_model = results_df['Accuracy'].idxmax()
    print(f"\n   Best Model (by Accuracy): {best_model} ({results_df.loc[best_model, 'Accuracy']:.4f})")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'model_results.pkl')
    joblib.dump(results, results_path)
    print(f"\n[6] Results saved to: model_results.pkl")

    print("\n" + "=" * 70)
    print("   TRAINING COMPLETE!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
