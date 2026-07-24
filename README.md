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

## Experimental Design Decisions

The following decisions were made during the design of this experiment:

1. The original Breast Cancer Wisconsin Diagnostic dataset was used without any augmentation or synthetic data generation, ensuring reproducibility and academic integrity.
2. All 30 numerical features were retained to allow models to leverage the full information available from cell nucleus measurements.
3. A stratified 80:20 train-test split was used to preserve the original class distribution in both training and evaluation sets.
4. StandardScaler was applied only to Logistic Regression, KNN, and SVM because these algorithms are sensitive to feature magnitudes (gradient descent, distance calculations, and kernel operations respectively).
5. Decision Tree, Random Forest, and Gaussian Naive Bayes were evaluated on raw unscaled features since their algorithms are inherently scale-invariant or operate on feature distributions directly.
6. sklearn Pipelines were used to encapsulate preprocessing and model training together, ensuring that no data leakage occurs between training and test sets.
7. Weighted Precision, Recall, and F1 were selected as evaluation metrics because the dataset has a mild class imbalance (37.3% Malignant vs 62.7% Benign), and weighted averaging accounts for this without artificially inflating any single class.
8. Logistic Regression was selected as the final recommended model after considering Accuracy, MCC, AUC, interpretability, and computational simplicity across all experiments.

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
| **Logistic Regression** | During experimentation, Logistic Regression consistently produced one of the highest overall Accuracy and MCC values. With an AUC of 0.9821, it demonstrates excellent class-ranking ability on this dataset. The StandardScaler preprocessing ensures all features contribute proportionally to the linear decision boundary. The high MCC (0.9058) confirms that the model handles both classes reliably, not just the majority class. |
| **Decision Tree** | The Decision Tree yielded the lowest test-set performance among the six models evaluated. A single tree tends to be sensitive to small variations in training data and can form brittle decision boundaries that do not generalize as well. Its AUC of 0.8185 reflects weaker discrimination ability compared to ensemble or linear approaches. Its primary strength remains interpretability through readable decision rules. |
| **KNN** | KNN performed well at 93.86% accuracy. The StandardScaler normalization was critical here because KNN relies on Euclidean distance, and unscaled features with larger magnitudes would dominate the distance calculation. With k=5, the model captures local neighborhood patterns effectively, though it falls slightly behind linear models on this particular dataset. |
| **Naive Bayes** | Gaussian Naive Bayes achieved 91.23% accuracy despite the conditional-independence assumption being imperfect for this dataset, since features such as radius, perimeter, and area are strongly correlated by nature. Its high AUC (0.9735) suggests good overall class-ranking ability. The model works directly on raw feature distributions without requiring any scaling. |
| **Random Forest (Ensemble)** | Random Forest improves noticeably over the standalone Decision Tree (83.33% to 91.23%) by averaging predictions across 100 individual trees, which reduces variance and instability. Its AUC of 0.9760 indicates strong discrimination between the two classes. The ensemble did not surpass linear models on this dataset because the underlying feature space is already well-separable with a linear boundary. |
| **SVM** | SVM ties with Logistic Regression at 95.61% accuracy and 0.9058 MCC. The RBF kernel allows it to capture non-linear patterns, though on this dataset the improvement over a linear boundary is marginal. Its AUC (0.9812) is slightly lower than Logistic Regression, indicating marginally weaker class-ranking on this particular test split. |
| **Overall Winner for your dataset?** | During experimentation, Logistic Regression and SVM consistently produced the highest overall Accuracy and MCC. **Logistic Regression** was selected as the recommended model because it provides comparable predictive performance while remaining simpler and easier to interpret. It also has the highest AUC (0.9821), trains faster, and its coefficients can be directly examined for feature importance. |

**Note on AUC:** AUC is calculated using class 1 (Benign) as the positive class, following the original sklearn target encoding.

---

## Model Limitations

Each model has inherent limitations that should be acknowledged:

- **Logistic Regression** assumes that the relationship between features and log-odds of the target is approximately linear. It may underperform when complex non-linear interactions exist in the data.
- **Decision Tree** can easily overfit to training data noise, especially without pruning or depth constraints. Small changes in the training set can produce substantially different tree structures.
- **KNN** is highly sensitive to feature scaling and the choice of k. It also becomes computationally expensive during prediction as the dataset grows, since it must compute distances to all training points.
- **Gaussian Naive Bayes** assumes that all features are conditionally independent given the class label. When features are correlated (as in this dataset), the model's probability estimates may be unreliable even if classifications remain reasonable.
- **Random Forest** provides strong performance but at the cost of interpretability. With 100 trees, it is difficult to trace why a particular prediction was made, unlike a single decision tree.
- **SVM** with an RBF kernel can be slow to train on larger datasets due to its computational complexity. The model also requires careful tuning of the regularization parameter C and kernel parameter gamma.

---

## Project Highlights

- Six ML classification models implemented and evaluated on the same dataset
- Model-specific preprocessing using sklearn Pipelines (no data leakage)
- Robust CSV validation in the Streamlit app (missing columns, invalid values, infinite values, type checks)
- Interactive Streamlit dashboard with multiple visualization tabs
- Comparative model analysis with tied-model detection
- Confusion matrix heatmap, ROC curve, and model comparison bar charts
- Feature importance visualization using Random Forest
- Complete GitHub documentation with experimental design rationale

---

## e. Streamlit App Features

The deployed Streamlit app includes:
1. **CSV Upload Option** - Upload custom test data with comprehensive validation
2. **Model Selection Dropdown** - Choose from 6 trained ML models
3. **Evaluation Metrics Display** - Shows Accuracy, AUC, Precision (weighted), Recall (weighted), F1 (weighted), MCC
4. **Confusion Matrix & Classification Report** - Visual heatmap and detailed tabular report
5. **All Models Comparison** - Side-by-side metric comparison table with visual bar charts
6. **Dataset Overview** - Feature statistics, target distribution, and feature descriptions
7. **Feature Importance** - Top 10 features from Random Forest with explanation
8. **About Project** - Assignment details, technologies used, and author information

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

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV to find optimal model configurations
- K-Fold Cross Validation (e.g., 5-fold or 10-fold) for more robust and reliable evaluation estimates
- Model explainability using SHAP values or LIME to understand individual predictions
- Probability calibration analysis to verify that predicted probabilities match observed frequencies
- Docker containerization for reproducible deployment across environments
- CI/CD pipeline integration for automated testing and deployment on code changes
- Support for multiple datasets with dynamic feature detection in the Streamlit interface

---

## h. Author

**Abhay Kumar**  
BITS ID: 2025AC05310  
M.Tech (AI & ML) - BITS Pilani (WILP)  
Machine Learning - Assignment 2
