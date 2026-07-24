"""
Breast Cancer Wisconsin Classification - Streamlit Web Application
Author: Abhay Kumar
Description: Interactive ML classification app demonstrating 6 models with evaluation metrics.
BITS ID: 2025AC05310
Programme: M.Tech (AI & ML) - BITS Pilani (WILP)
Assignment: Machine Learning - Classification Assignment 2
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Models use sklearn Pipelines with model-specific preprocessing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_pipelines():
    """Load all pre-trained model pipelines from the model/ directory."""
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    pipelines = {}
    pipeline_files = {
        'Logistic Regression': 'logistic_regression_pipeline.pkl',
        'Decision Tree': 'decision_tree_pipeline.pkl',
        'KNN': 'knn_pipeline.pkl',
        'Naive Bayes': 'naive_bayes_pipeline.pkl',
        'Random Forest': 'random_forest_pipeline.pkl',
        'SVM': 'svm_pipeline.pkl'
    }

    for name, filename in pipeline_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            pipelines[name] = joblib.load(filepath)

    return pipelines


@st.cache_resource
def load_feature_names():
    """Load feature names used during training."""
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    path = os.path.join(model_dir, 'feature_names.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def validate_uploaded_data(data, expected_features):
    """
    Validate the uploaded CSV data comprehensively.
    Returns (is_valid, error_messages, warnings).
    """
    errors = []
    warns = []

    # Check if 'target' column exists
    if 'target' not in data.columns:
        errors.append("Missing required column: 'target'")

    # Check for expected feature columns
    feature_cols = [col for col in data.columns if col != 'target']
    missing_cols = [col for col in expected_features if col not in data.columns]
    extra_cols = [col for col in feature_cols if col not in expected_features]

    if missing_cols:
        errors.append(f"Missing feature columns ({len(missing_cols)}): {', '.join(missing_cols[:5])}" +
                     (f"... and {len(missing_cols)-5} more" if len(missing_cols) > 5 else ""))

    if extra_cols:
        warns.append(f"Extra columns detected ({len(extra_cols)}): {', '.join(extra_cols[:5])}" +
                    (f"... and {len(extra_cols)-5} more" if len(extra_cols) > 5 else "") +
                    ". These will be ignored.")

    # Check for missing values (only on columns that actually exist)
    available_features = [col for col in expected_features if col in data.columns]
    if available_features:
        null_counts = data[available_features].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            total_nulls = int(cols_with_nulls.sum())
            errors.append(
                f"Missing values detected: {total_nulls} null values "
                f"across {len(cols_with_nulls)} columns. "
                f"Columns: {', '.join(cols_with_nulls.index[:3].tolist())}"
            )

    # Check for infinite values
    if available_features and np.isinf(data[available_features].to_numpy()).any():
        errors.append("Infinite values detected in feature columns.")

    # Check for invalid target values
    if 'target' in data.columns:
        unique_targets = data['target'].unique()
        invalid_targets = [t for t in unique_targets if t not in [0, 1]]
        if len(invalid_targets) > 0:
            errors.append(f"Invalid target values found: {invalid_targets}. Expected only 0 (Malignant) and 1 (Benign).")

        if data['target'].nunique() < 2:
            errors.append(f"Only one class found in target column (value: {unique_targets[0]}). "
                         "Need both classes (0 and 1) for meaningful evaluation.")

    # Check for non-numeric features
    for col in available_features:
        if not pd.api.types.is_numeric_dtype(data[col]):
            errors.append(f"Column '{col}' contains non-numeric data. All features must be numeric.")
            break

    # Check minimum row count
    if len(data) < 5:
        errors.append(f"Too few samples ({len(data)}). Need at least 5 rows for evaluation.")

    is_valid = len(errors) == 0
    return is_valid, errors, warns


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate all 6 evaluation metrics.
    Precision, Recall, F1 use weighted average to handle class imbalance.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_prob),
        'Precision (weighted)': precision_score(y_true, y_pred, average='weighted'),
        'Recall (weighted)': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score (weighted)': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'{model_name} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison(all_results):
    """Plot comparison bar chart for all models."""
    df = pd.DataFrame(all_results).T

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = ['Accuracy', 'AUC Score', 'Precision (weighted)', 'Recall (weighted)', 'F1 Score (weighted)', 'MCC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(df.index, df[metric], color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.suptitle('Model Performance Comparison (6 Models)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ==================== MAIN APPLICATION ====================

def main():
    # Header
    st.markdown('<p class="main-header">Breast Cancer Classification Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive ML Model Comparison | 6 Classifiers | Wisconsin Diagnostic Dataset</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Load pipelines and feature names
    pipelines = load_pipelines()
    feature_names = load_feature_names()

    if not pipelines:
        st.error("No trained models found! Please run `python model/train_models.py` first.")
        return

    # ==================== SIDEBAR ====================
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")

    # Model selection dropdown
    st.sidebar.subheader("Select Model")
    selected_model_name = st.sidebar.selectbox(
        "Choose a classification model:",
        list(pipelines.keys()),
        index=0  # Default to Logistic Regression (best performer)
    )

    # Show preprocessing info for selected model
    preprocessing_info = {
        'Logistic Regression': 'StandardScaler + LogisticRegression',
        'Decision Tree': 'No scaling + DecisionTreeClassifier',
        'KNN': 'StandardScaler + KNeighborsClassifier',
        'Naive Bayes': 'No scaling + GaussianNB',
        'Random Forest': 'No scaling + RandomForestClassifier',
        'SVM': 'StandardScaler + SVC (RBF kernel)'
    }
    st.sidebar.caption(f"Pipeline: {preprocessing_info[selected_model_name]}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Test Data")
    st.sidebar.markdown("Upload a CSV file with test data to evaluate the model.")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your test data CSV. Must contain the same 30 features as training data + 'target' column."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Author")
    st.sidebar.markdown("**Abhay Kumar**")
    st.sidebar.markdown("BITS ID: 2025AC05310")
    st.sidebar.markdown("BITS Pilani WILP")
    st.sidebar.markdown("Machine Learning")
    st.sidebar.markdown("Classification Assignment 2")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset")
    st.sidebar.info(
        "**Breast Cancer Wisconsin (Diagnostic)**\n\n"
        "**Source:** UCI ML Repository / sklearn\n\n"
        "**Features:** 30 measurements\n\n"
        "**Instances:** 569 (no augmentation)\n\n"
        "**Classes:** Malignant (0) / Benign (1)\n\n"
        "**Metrics:** Weighted average"
    )

    # ==================== DATA LOADING & VALIDATION ====================
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please upload a valid file.")
            return
        except pd.errors.ParserError:
            st.error("The uploaded CSV file is malformed or could not be parsed. Please check the file format.")
            return
        except UnicodeDecodeError:
            st.error("The uploaded file has an unsupported encoding. Please use UTF-8 encoded CSV files.")
            return
        except Exception as e:
            st.error(f"Error reading the uploaded file: {str(e)}")
            return

        if data.empty:
            st.error("The uploaded CSV file contains no data rows.")
            return

        st.success(f"Uploaded: {uploaded_file.name} ({data.shape[0]} rows, {data.shape[1]} columns)")

        # Validate uploaded data
        is_valid, errors, warns = validate_uploaded_data(data, feature_names)

        if warns:
            for w in warns:
                st.warning(f"Warning: {w}")

        if not is_valid:
            st.error("Data validation failed:")
            for err in errors:
                st.error(f"  - {err}")
            st.info(f"Expected features ({len(feature_names)}): {', '.join(feature_names[:5])}... "
                   f"[Upload test_data.csv from the repository for reference]")
            return
    else:
        # Use default test data
        test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
        if os.path.exists(test_data_path):
            data = pd.read_csv(test_data_path)
            st.info("Using default test data (test_data.csv). Upload your own CSV from the sidebar.")
        else:
            st.error("No test data found! Please upload a CSV file.")
            return

    # Split features and target
    X = data[feature_names]  # Use only expected features in correct order
    y = data['target']

    # ==================== MODEL EVALUATION ====================

    # Get selected pipeline (includes preprocessing + model)
    pipeline = pipelines[selected_model_name]

    # Make predictions (pipeline handles preprocessing internally)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y, y_pred, y_prob)

    # ==================== DISPLAY RESULTS ====================

    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Model Metrics", "Confusion Matrix & Report",
        "All Models Comparison", "Feature Importance",
        "Dataset Overview", "About Project"
    ])

    with tab1:
        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        st.caption(f"Pipeline: {preprocessing_info[selected_model_name]}")
        st.markdown("---")

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
        with col2:
            st.metric("Precision (weighted)", f"{metrics['Precision (weighted)']:.4f}")
            st.metric("Recall (weighted)", f"{metrics['Recall (weighted)']:.4f}")
        with col3:
            st.metric("F1 Score (weighted)", f"{metrics['F1 Score (weighted)']:.4f}")
            st.metric("MCC", f"{metrics['MCC']:.4f}")

        st.markdown("---")

        # ROC Curve
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y, y_prob, selected_model_name)
        st.pyplot(fig_roc)
        plt.close()

    with tab2:
        st.subheader(f"Confusion Matrix: {selected_model_name}")
        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_cm = plot_confusion_matrix(y, y_pred, selected_model_name)
            st.pyplot(fig_cm)
            plt.close()

        with col2:
            st.subheader("Classification Report")
            report = classification_report(
                y, y_pred,
                target_names=['Malignant (0)', 'Benign (1)'],
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

            # Confusion matrix values
            cm = confusion_matrix(y, y_pred)
            st.markdown("**Confusion Matrix Values:**")
            cm_df = pd.DataFrame(cm,
                                index=['Actual Malignant', 'Actual Benign'],
                                columns=['Predicted Malignant', 'Predicted Benign'])
            st.dataframe(cm_df, use_container_width=True)

    with tab3:
        st.subheader("All Models Comparison (6 Classifiers)")
        st.markdown("---")

        # Evaluate all models
        all_results = {}
        for name, pipe in pipelines.items():
            y_p = pipe.predict(X)
            y_pr = pipe.predict_proba(X)[:, 1]
            all_results[name] = calculate_metrics(y, y_p, y_pr)

        # Display comparison table
        st.subheader("Metrics Comparison Table")
        st.caption("Precision, Recall, and F1 Score use **weighted average** to account for class imbalance.")
        comparison_df = pd.DataFrame(all_results).T
        comparison_df = comparison_df.round(4)

        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen')
                              .highlight_min(axis=0, color='lightyellow'),
            use_container_width=True
        )

        # Best model callout (handle ties)
        best_accuracy = comparison_df['Accuracy'].max()
        best_models = comparison_df[comparison_df['Accuracy'] == best_accuracy].index.tolist()

        if len(best_models) == 1:
            st.success(f"**Best Model (by Accuracy):** {best_models[0]} with {best_accuracy:.4f}")
        else:
            st.success(f"**Best Models (by Accuracy):** {', '.join(best_models)} with a tied score of {best_accuracy:.4f}")

        # Preprocessing summary
        st.subheader("Model Preprocessing Summary")
        prep_df = pd.DataFrame({
            'Model': list(preprocessing_info.keys()),
            'Pipeline': list(preprocessing_info.values()),
            'Requires Scaling': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes']
        })
        st.dataframe(prep_df, use_container_width=True, hide_index=True)

        # Comparison chart
        st.subheader("Visual Comparison")
        fig_comp = plot_model_comparison(all_results)
        st.pyplot(fig_comp)
        plt.close()

    with tab4:
        st.subheader("Feature Importance (Random Forest)")
        st.markdown("---")
        st.markdown("The chart below shows the **top 10 most important features** as determined by the Random Forest model. "
                   "Feature importance is measured by the average decrease in impurity (Gini importance) across all trees in the ensemble.")

        # Get feature importances from Random Forest pipeline
        if 'Random Forest' in pipelines:
            rf_pipeline = pipelines['Random Forest']
            rf_model = rf_pipeline.named_steps['classifier']
            importances = rf_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1],
                          color='#2ca02c', edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.set_xlabel('Importance (Gini)', fontsize=12)
            ax.set_title('Top 10 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Interpretation:** Features with higher importance contribute more to the model's ability to "
                       "distinguish between malignant and benign tumors. The worst perimeter, worst concave points, and "
                       "mean concave points tend to be the most discriminative features in this dataset, reflecting "
                       "the physical characteristics that differentiate cancerous cells.")
        else:
            st.warning("Random Forest model not available for feature importance visualization.")

    with tab5:
        st.subheader("Dataset Overview")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Samples Loaded", data.shape[0])
        with col2:
            st.metric("Features", len(feature_names))
        with col3:
            st.metric("Target Classes", 2)
        with col4:
            st.metric("Augmentation", "None")

        st.markdown("---")

        # Show data preview
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(data.head(10), use_container_width=True)

        # Target distribution
        st.subheader("Target Distribution")
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            target_counts = data['target'].value_counts().sort_index()
            bars = ax.bar(['Malignant (0)', 'Benign (1)'], target_counts.values,
                         color=['#d62728', '#2ca02c'], edgecolor='black', linewidth=0.5)
            ax.set_title('Target Class Distribution', fontsize=13, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Feature Statistics (first 5 features):**")
            st.dataframe(data[feature_names[:5]].describe().round(3), use_container_width=True)

        # Feature descriptions
        st.subheader("Feature Descriptions")
        st.markdown("""
        The dataset contains 30 features computed from a digitized image of a fine needle aspirate (FNA)
        of a breast mass. Features describe characteristics of cell nuclei present in the image.

        For each of the 10 real-valued features (radius, texture, perimeter, area, smoothness,
        compactness, concavity, concave points, symmetry, fractal dimension), three statistics
        are computed: **mean**, **standard error (SE)**, and **worst** (largest value).
        """)

    with tab6:
        st.subheader("About This Project")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Project Details")
            st.markdown("""
            | Property | Value |
            |----------|-------|
            | **Assignment** | ML Classification Assignment 2 |
            | **Course** | Machine Learning |
            | **Programme** | M.Tech (AI & ML) |
            | **University** | BITS Pilani (WILP) |
            | **Author** | Abhay Kumar |
            | **BITS ID** | 2025AC05310 |
            """)

        with col2:
            st.markdown("#### Technical Details")
            import sklearn
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | **Python Version** | 3.11 |
            | **scikit-learn** | {sklearn.__version__} |
            | **Streamlit** | {st.__version__} |
            | **Dataset** | Breast Cancer Wisconsin |
            | **Total Samples** | 569 |
            | **Total Features** | 30 |
            | **Models Implemented** | 6 |
            """)

        st.markdown("---")
        st.markdown("#### Models Implemented")
        st.markdown("""
        1. Logistic Regression (with StandardScaler)
        2. Decision Tree Classifier (raw features)
        3. K-Nearest Neighbors (with StandardScaler)
        4. Gaussian Naive Bayes (raw features)
        5. Random Forest Classifier (raw features)
        6. Support Vector Machine (with StandardScaler)
        """)

        st.markdown("---")
        st.markdown("#### Technologies Used")
        st.markdown("Python | Streamlit | scikit-learn | pandas | numpy | matplotlib | seaborn | joblib")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 10px;'>"
        "Developed by <b>Abhay Kumar</b> (2025AC05310) | Machine Learning Assignment 2 | BITS Pilani WILP"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
