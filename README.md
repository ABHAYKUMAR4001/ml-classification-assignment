# Breast Cancer Wisconsin Classification - ML Assignment 2

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models for **Breast Cancer Diagnosis** using the Wisconsin Diagnostic dataset. The goal is to classify tumors as **Malignant (0)** or **Benign (1)** based on 30 diagnostic features computed from digitized images of fine needle aspirate (FNA) of breast masses.

This project demonstrates an end-to-end ML pipeline including data preprocessing, model training with model-specific preprocessing (sklearn Pipelines), evaluation using 6 metrics, and deployment as an interactive Streamlit web application.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Dataset Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository / sklearn.datasets |
| **Total Instances** | 569 (original dataset, no augmentation) |
| **Number of Features** | 30 |
| **Target Variable** | Binary (0 = Malignant, 1 = Benign) |
| **Class Distribution** | Malignant: 212 (37.3%), Benign: 357 (62.7%) |
| **Train/Test Split** | 80/20 stratified (455 train, 114 test) |
| **Missing Values** | None |

### Features (30 total):

The dataset contains 30 features computed from digitized images of FNA of breast masses. For each of the 10 real-valued cell nucleus properties, three statistics are computed:

| Property | Mean | Standard Error | Worst (Largest) |
|----------|------|---------------|-----------------|
| Radius | mean radius | radius error | worst radius |
| Texture | mean texture | texture error | worst texture |
| Perimeter | mean perimeter | perimeter error | worst perimeter |
| Area | mean area | area error | worst area |
| Smoothness | mean smoothness | smoothness error | worst smoothness |
| Compactness | mean compactness | compactness error | worst compactness |
| Concavity | mean concavity | concavity error | worst concavity |
| Concave Points | mean concave points | concave points error | worst concave points |
| Symmetry | mean symmetry | symmetry error | worst symmetry |
| Fractal Dimension | mean fractal dimension | fractal dimension error | worst fractal dimension |

---


## c. GitHub Repository Link

**Repository:** [https://github.com/ABHAYKUMAR4001/ml-classification-assignment](https://github.com/ABHAYKUMAR4001/ml-classification-assignment)

### Repository Structure:
```
ml-classification-assignment/
|-- app.py                              # Streamlit web application
|-- requirements.txt                    # Python dependencies (minimum compatible versions)
|-- README.md                           # Project documentation
|-- test_data.csv                       # Test dataset (114 rows, 31 columns)
|-- .gitignore                          # Git ignore rules
|-- model/                              # Trained model pipelines
    |-- train_models.py                 # Model training script
    |-- logistic_regression_pipeline.pkl
    |-- decision_tree_pipeline.pkl
    |-- knn_pipeline.pkl
    |-- naive_bayes_pipeline.pkl
    |-- random_forest_pipeline.pkl
    |-- svm_pipeline.pkl
    |-- feature_names.pkl
    |-- model_results.pkl
```

---

## d. Models Used

### Model-Specific Preprocessing

Each model uses an sklearn Pipeline with appropriate preprocessing:

| Model | Preprocessing | Reason |
|-------|--------------|--------|
| Logistic Regression | StandardScaler | Gradient-based; sensitive to feature scale |
| KNN | StandardScaler | Distance-based; requires normalized features |
| SVM | StandardScaler | Kernel computation sensitive to scale |
| Decision Tree | None (raw features) | Tree splits are scale-invariant |
| Random Forest | None (raw features) | Ensemble of trees; scale-invariant |
| Naive Bayes | None (raw features) | Works on raw feature distributions |


### Evaluation Metrics Comparison Table

All 6 ML models were implemented on the same Breast Cancer Wisconsin dataset.

**Note:** Precision, Recall, and F1 Score use **weighted average** to account for class imbalance (37.3% Malignant vs 62.7% Benign).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-------|------|
| **Logistic Regression** | 0.9561 | 0.9821 | 0.9569 | 0.9561 | 0.9558 | 0.9058 |
| **Decision Tree** | 0.8333 | 0.8185 | 0.8326 | 0.8333 | 0.8329 | 0.6402 |
| **KNN** | 0.9386 | 0.9659 | 0.9408 | 0.9386 | 0.9377 | 0.8688 |
| **Naive Bayes** | 0.9123 | 0.9735 | 0.9181 | 0.9123 | 0.9100 | 0.8138 |
| **Random Forest (Ensemble)** | 0.9123 | 0.9760 | 0.9119 | 0.9123 | 0.9118 | 0.8102 |
| **SVM** | 0.9561 | 0.9812 | 0.9569 | 0.9561 | 0.9558 | 0.9058 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Achieves the highest accuracy (95.61%) tied with SVM. The linear model performs exceptionally well here because the scaled features exhibit strong linear separability. AUC of 0.9821 (highest overall) confirms excellent class-ranking ability. The Pipeline with StandardScaler ensures optimal convergence of the gradient-based solver (max_iter=5000). MCC of 0.9058 indicates near-perfect balanced classification. |
| **Decision Tree** | The Decision Tree achieved the lowest test-set performance among the evaluated models. A standalone tree can be sensitive to variations in the training data and may form unstable decision boundaries. Its lower AUC (0.8185) indicates weaker class-discrimination performance. No preprocessing needed since tree splits are scale-invariant. Its main advantage is interpretability through explicit decision rules. |
| **KNN** | Strong performance (93.86% accuracy) with k=5 neighbors. Distance-based classification benefits greatly from StandardScaler normalization (all features contribute equally to distance computation). High AUC (0.9659) shows good class-discrimination ability. Slightly lower than LR/SVM because some test instances lie in ambiguous boundary regions. |
| **Naive Bayes** | Gaussian Naive Bayes achieves good performance (91.23% accuracy) despite the conditional-independence assumption being imperfect for this dataset, since features such as radius, perimeter, and area are strongly related. Its high AUC (0.9735) indicates good class-ranking ability, although correlated features may affect probability estimates. No scaling applied since GaussianNB models raw feature distributions directly. |
| **Random Forest (Ensemble)** | Random Forest improves substantially over the standalone Decision Tree through ensemble averaging (83.33% to 91.23%). Its high AUC (0.9760) indicates strong discrimination between the two classes. However, it does not outperform simpler linear models on this dataset because the features are well-separable linearly after scaling. No scaling needed as tree-based methods are scale-invariant. |
| **SVM** | SVM ties with Logistic Regression in Accuracy (95.61%) and MCC (0.9058). The RBF kernel with StandardScaler effectively creates non-linear decision boundaries that generalize well. Its AUC (0.9812) is slightly lower than Logistic Regression, indicating marginally weaker class-ranking performance on this particular test split. |
| **Overall Winner for your dataset?** | **Logistic Regression** and **SVM** are tied as the best models (Accuracy: 0.9561, MCC: 0.9058). Logistic Regression is recommended as the overall winner because: (1) it has marginally higher AUC (0.9821 vs 0.9812), (2) it is simpler and more interpretable, (3) it trains faster, and (4) the dataset's features are well-suited to linear classification after scaling. |

**Note on AUC:** AUC is calculated using class 1 (Benign) as the positive class, following the original sklearn target encoding.


---

## e. Streamlit App Features

The deployed Streamlit app includes:
1. **CSV Upload Option** - Upload custom test data with comprehensive validation (checks for missing columns, invalid values, data types, etc.)
2. **Model Selection Dropdown** - Choose from 6 trained ML models
3. **Evaluation Metrics Display** - Shows Accuracy, AUC, Precision (weighted), Recall (weighted), F1 (weighted), MCC
4. **Confusion Matrix & Classification Report** - Visual heatmap and detailed tabular report
5. **All Models Comparison** - Side-by-side metric comparison table with visual bar charts
6. **Dataset Overview** - Feature statistics, target distribution, and feature descriptions
7. **Pipeline Info** - Shows preprocessing details for each selected model

---

## f. How to Run Locally

```bash
# Clone the repository
git clone https://github.com/ABHAYKUMAR4001/ml-classification-assignment.git
cd ml-classification-assignment

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models
python model/train_models.py

# Run the Streamlit app
streamlit run app.py
```

---

## g. Technologies Used

- **Python 3.11**
- **Streamlit** - Web application framework
- **scikit-learn** - ML models, Pipelines, evaluation metrics
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization
- **joblib** - Model/Pipeline serialization

---

## h. Author

**Abhay Kumar**  
M.Tech (AIML/DSE) - BITS Pilani (WILP)  
Machine Learning - Assignment 2
