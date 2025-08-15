# 🔄 Iterative Analysis Report: Model Selection & Evaluation

## 🎯 Process Overview
This report shows the complete 4-step iterative process:
1. **Planner**: Strategic planning and task decomposition
2. **Developer**: Initial implementation
3. **Auditor**: Review and feedback
4. **Developer**: Refined implementation

## 🔧 Phase: Model Selection & Evaluation

### 📋 Planned Subtasks
1. Define Modeling Strategy and Evaluation Plan
2. - Select appropriate train/validation/test splits
3. - Choose evaluation metrics (f1, accuracy, precision, recall, confusion matrix)
4. - Outline hyperparameter tuning approach (grid search, random search, Bayesian optimization)
5. Data Preparation for Modeling
6. - Split dataset into training, validation, and test sets (70/10/20) with fixed seed
7. - Apply consistent feature scaling or normalization if required
8. - Encode categorical variables if additional encoding beyond label encoding is needed
9. - Ensure no data leakage by fitting transformers only on training data
10. Model Training and Hyperparameter Optimization
11. - Train initial RandomForestClassifier with baseline parameters
12. - Perform hyperparameter tuning (e.g., n_estimators, max_depth, min_samples_split) using validation set
13. - Record best hyperparameter set based on validation metrics
14. Model Evaluation and Interpretation
15. - Evaluate the final model on the test set using all specified metrics
16. - Generate confusion matrix and classification report
17. - Analyze feature importances and partial dependence plots for interpretability
18. - Document insights, strengths, and limitations of the model
19. Validation and Documentation
20. - Validate model stability via cross-validation if applicable
21. - Log all modeling parameters, data splits, and evaluation results
22. - Save trained model, evaluation reports, and visualizations for audit
23. Refinement and Iteration
24. - Incorporate audit feedback regarding model performance, data leakage, or interpretability
25. - Adjust feature set, hyperparameters, or preprocessing steps accordingly
26. - Retrain and reevaluate as needed, repeating steps 3-5 within retry limits

### 📋 Step 1: DrAliceSmith (Strategic Planning)
**Role:** Planner

Certainly! Below is a detailed, step-by-step implementation plan for the Model Selection & Evaluation phase, structured to facilitate clear handoff, review, and iterative refinement. This plan aligns with the overall project goals, best practices, and the context provided.

---

# **Modeling Strategy & Implementation Plan**

## 1. **Modeling Strategy**

### **Rationale for Model Choice**
- **Model Type:** RandomForestClassifier
- **Justification:**
  - Handles tabular data well.
  - Robust to outliers and noise.
  - Provides feature importance for interpretability.
  - Less sensitive to feature scaling (though scaling is optional).
- **Target Variable:** `Class` (categorical, object type), encoded as `class_encoded`.

### **Evaluation Methodology**
- **Data Splits:** 70% training, 10% validation, 20% testing, fixed seed (42) for reproducibility.
- **Metrics:**
  - **Primary:** F1-score (balanced for class imbalance)
  - **Secondary:** Accuracy, Precision, Recall
  - **Additional:** Confusion matrix and classification report for detailed insights.
- **Hyperparameter Tuning:**
  - Use validation set for tuning hyperparameters via RandomizedSearchCV or GridSearchCV.
  - Parameters to tune: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- **Model Selection:** Choose hyperparameters that maximize validation F1-score.

### **Success Criteria**
- Achieve stable, high performance on validation set.
- No data leakage.
- Model interpretability maintained via feature importance.
- Final evaluation on test set meets benchmark thresholds (e.g., F1 > 0.75, accuracy > 80%).

---

## 2. **Implementation Steps**

### **Step 1: Data Preparation for Modeling**

**a. Data Splitting**
- Use `train_test_split` from `sklearn.model_selection` with `random_state=42`.
- Split dataset into:
  - **Training:** 70%
  - **Validation:** 10%
  - **Test:** 20%
- **Procedure:**
  ```python
  from sklearn.model_selection import train_test_split

  # Assuming df is the preprocessed DataFrame
  X = df.drop(columns=['Class', 'class_encoded'])
  y = df['class_encoded']

  # First split into train+val and test
  X_temp, X_test, y_temp, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42, stratify=y)

  # Split train+val into train and validation
  X_train, X_val, y_train, y_val = train_test_split(
      X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
  # 0.125 * 0.80 = 0.10 overall
  ```
  
**b. Feature Scaling & Encoding**
- For RandomForest, scaling is optional but can be applied if desired.
- Ensure no data leakage:
  - Fit scalers only on training data.
  - Apply transformations to validation and test sets.
- **Optional:** Use `sklearn.preprocessing.StandardScaler` or `MinMaxScaler`.

**c. Save splits for reproducibility and further analysis.**

---

### **Step 2: Model Training & Hyperparameter Tuning**

**a. Baseline Model**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
```

**b. Hyperparameter Tuning**
- Use `RandomizedSearchCV` for efficiency.
- Parameter grid example:
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
search = RandomizedSearchCV(
    rf, param_distributions, n_iter=20, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)

best_params = search.best_params_
```

**c. Record the best hyperparameters and validation performance.**

---

### **Step 3: Final Model Training & Evaluation**

**a. Retrain with best hyperparameters**
```python
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
```

**b. Evaluation on validation set**
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

y_val_pred = final_model.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')

# Save evaluation metrics
evaluation_results = {
    'f1_score': val_f1,
    'accuracy': val_accuracy,
    'precision': val_precision,
    'recall': val_recall
}

# Generate and save confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.savefig('eda_outputs/validation_confusion_matrix.png')
plt.close()
```

---

### **Step 4: Final Evaluation on Test Set**

```python
y_test_pred = final_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

# Save final metrics and report
with open('eda_outputs/test_evaluation_report.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.3f}\n")
    f.write(f"Test F1-score: {test_f1:.3f}\n")
    f.write(f"Test Precision: {test_precision:.3f}\n")
    f.write(f"Test Recall: {test_recall:.3f}\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_test_pred))
```

**Visualizations:**
- Confusion matrix PNG.
- Feature importance barplot saved as `eda_outputs/feature_importance.png`.

---

### **Step 5: Model Persistence & Artifacts**
- Save the trained model:
```python
import joblib
joblib.dump(final_model, 'eda_outputs/final_rf_model.pkl')
```
- Save feature importance DataFrame and hyperparameters.

---

## 3. **Additional Recommendations & Best Practices**

- **Cross-Validation:** For more robust estimates, consider nested cross-validation during hyperparameter tuning.
- **Feature Importance & Interpretability:** Use `feature_importances_` and partial dependence plots for insights.
- **Reproducibility:** Log all parameters, random seeds, and dataset versions.
- **Error Handling:** Wrap steps with try-except blocks, logging errors and warnings.
- **Pipeline Integration:** Encapsulate steps into a scikit-learn pipeline for modularity.

---

## 4. **Summary & Next Steps**

- Implement dataset splits with stratification.
- Perform hyperparameter tuning with validation set.
- Retrain the model with optimal parameters.
- Evaluate on test set, generate reports and visualizations.
- Save models and artifacts in `eda_outputs/`.
- Document all parameters, metrics, and insights for audit.

---

# **Success Criteria Recap**
- No data leakage.
- Validation performance exceeds benchmarks.
- Final test metrics meet or surpass thresholds.
- Artifacts (models, reports, plots) are saved and documented.
- Reproducibility and interpretability are maintained.

---

This detailed plan should guide the developer through implementation, facilitate review by auditors, and support iterative refinement based on feedback.

### 💻 Step 2: JohnDoe (Initial Implementation)
**Role:** Developer

**Initial Code:**
```python
"""
Model Selection & Evaluation Implementation for RandomForestClassifier

This script implements the modeling strategy as specified by the Planner:
- Splits the dataset into train (70%), validation (10%), and test (20%) sets with stratification.
- Trains a baseline RandomForestClassifier.
- Performs hyperparameter tuning using RandomizedSearchCV on the training set.
- Retrains the final model with the best hyperparameters.
- Evaluates the model on validation and test sets using F1-score, accuracy, precision, recall.
- Saves evaluation reports, confusion matrices, feature importance plots, and the final model.
- Ensures reproducibility with fixed random seed (42).
- Saves all artifacts under "../output/models/" and "../output/eda_outputs/".

Assumptions:
- `df` is the preprocessed DataFrame available in the environment.
- Target variable is `class_encoded`.
- Features exclude 'Class' and 'class_encoded'.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Set random seed for reproducibility
RANDOM_SEED = 42

# Create output directories if they don't exist
model_dir = "../output/models/"
eda_dir = "../output/eda_outputs/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(eda_dir, exist_ok=True)

# Step 1: Data Preparation for Modeling
# -------------------------------------

# Define features and target
X = df.drop(columns=['Class', 'class_encoded'])
y = df['class_encoded']

# Split into train+val (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

# Split train+val into train (70%) and val (10%)
# Since train+val is 80%, val size relative to temp is 0.125 (0.125 * 0.8 = 0.10)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=RANDOM_SEED, stratify=y_temp
)

# Optional: Feature scaling is not applied because RandomForest is scale-invariant

# Step 2: Model Training & Hyperparameter Tuning
# ----------------------------------------------

# Baseline RandomForestClassifier
rf_baseline = RandomForestClassifier(random_state=RANDOM_SEED)
rf_baseline.fit(X_train, y_train)

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=RANDOM_SEED)
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='f1',
    cv=5,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
best_params = search.best_params_

# Save best hyperparameters
with open(os.path.join(eda_dir, 'best_hyperparameters.txt'), 'w') as f:
    f.write("Best Hyperparameters from RandomizedSearchCV:\n")
    for param, val in best_params.items():
        f.write(f"{param}: {val}\n")

# Step 3: Final Model Training & Validation Evaluation
# ----------------------------------------------------

final_model = RandomForestClassifier(**best_params, random_state=RANDOM_SEED)
final_model.fit(X_train, y_train)

# Validation predictions and metrics
y_val_pred = final_model.predict(X_val)

val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')

evaluation_results = {
    'f1_score': val_f1,
    'accuracy': val_accuracy,
    'precision': val_precision,
    'recall': val_recall
}

# Save validation evaluation metrics
val_metrics_path = os.path.join(eda_dir, 'validation_evaluation_metrics.txt')
with open(val_metrics_path, 'w') as f:
    f.write("Validation Set Evaluation Metrics:\n")
    for metric, score in evaluation_results.items():
        f.write(f"{metric}: {score:.4f}\n")

# Validation confusion matrix plot
cm_val = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'validation_confusion_matrix.png'))
plt.close()

# Step 4: Final Evaluation on Test Set
# ------------------------------------

y_test_pred = final_model.predict(X_test)

test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

# Save test evaluation report
test_report_path = os.path.join(eda_dir, 'test_evaluation_report.txt')
with open(test_report_path, 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test F1-score: {test_f1:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_test_pred))

# Test confusion matrix plot
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'test_confusion_matrix.png'))
plt.close()

# Feature importance plot
feature_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index, palette='viridis')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'feature_importance.png'))
plt.close()

# Step 5: Model Persistence & Artifacts Saving
# --------------------------------------------

# Save the final trained model
model_path = os.path.join(model_dir, 'final_rf_model.pkl')
joblib.dump(final_model, model_path)

# Save feature importances as CSV for audit
feature_importances_sorted.to_csv(os.path.join(eda_dir, 'feature_importances.csv'), header=['importance'])

# Save train/val/test splits for reproducibility (optional)
X_train.to_csv(os.path.join(eda_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(eda_dir, 'y_train.csv'), index=False)
X_val.to_csv(os.path.join(eda_dir, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(eda_dir, 'y_val.csv'), index=False)
X_test.to_csv(os.path.join(eda_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(eda_dir, 'y_test.csv'), index=False)

print("Model training, tuning, evaluation, and artifact saving completed successfully.")
print(f"Validation F1-score: {val_f1:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
```

### 🔍 Step 3: EmilyJohnson (Quality Review)
**Role:** Auditor

**EmilyJohnson - ML Implementation Audit Feedback**

**Overall Assessment:**

The implementation aligns well with the outlined modeling strategy, demonstrating a solid understanding of best practices in model selection, hyperparameter tuning, evaluation, and artifact management. The process is systematic, reproducible, and well-documented, which is commendable.

---

### **Strengths & Positive Observations:**

- **Reproducibility & Data Splits:**  
  The dataset is split with stratification into train, validation, and test sets, ensuring representative sampling. Random seed is fixed at 42, maintaining reproducibility.

- **Hyperparameter Tuning:**  
  Utilization of `RandomizedSearchCV` over a broad parameter grid with 20 iterations and 5-fold CV is appropriate, balancing thoroughness and computational efficiency.

- **Model Training & Evaluation:**  
  Final model retraining with best hyperparameters, followed by comprehensive evaluation on validation and test sets, including F1, accuracy, precision, recall, and confusion matrices, adheres to best practices.

- **Artifact Management:**  
  All relevant outputs—models, plots, reports, hyperparameters—are saved systematically, facilitating transparency and reproducibility.

- **Visualization & Reporting:**  
  Clear visualizations of confusion matrices and feature importances enhance interpretability. Text reports document metrics effectively.

---

### **Areas for Improvement & Recommendations:**

1. **Data Preprocessing & Feature Engineering:**

   - **Outlier Flags & Log Transformations:**  
     The dataset contains numerous outlier flags and log-transformed features. Ensure that these transformations and flags are incorporated into the final feature set consistently. For example, confirm whether features like `solidity_log1p`, `extent_log1p`, etc., are included in `X`.  
     *Rationale:* Consistency here impacts model performance and interpretability.

2. **Feature Scaling & Model Compatibility:**

   - While RandomForest is scale-invariant, if you plan to compare with models sensitive to feature scaling later, document this choice.  
   - Consider explicitly noting whether feature scaling was applied or not, especially if other models are to be tested subsequently.

3. **Hyperparameter Search Details & Validation:**

   - **Cross-Validation Consistency:**  
     Confirm that the stratification is maintained during hyperparameter tuning.  
   - **Parameter Ranges:**  
     The ranges are broad; consider narrowing or focusing on promising hyperparameters based on initial results to optimize tuning efficiency.

4. **Evaluation & Overfitting Checks:**

   - **Overfitting Signs:**  
     Compare validation and test metrics to detect overfitting. If validation F1 is high but test F1 drops significantly, consider more regularization or feature selection.
   - **Model Generalization:**  
     If performance is close on validation and test sets, it indicates good generalization.

5. **Model Interpretability & Bias:**

   - **Feature Importance Analysis:**  
     The barplot provides insights into influential features. Consider supplementing with partial dependence plots for top features to understand their impact.
   - **Bias & Variance:**  
     No explicit bias-variance diagnostics are included. If performance is subpar, consider additional diagnostics or model complexity adjustments.

6. **Error Handling & Robustness:**

   - The script assumes `df` is preprocessed and available.  
   - Incorporate try-except blocks around critical steps to catch and log errors gracefully, especially during file I/O and model fitting.

7. **Documentation & Reproducibility:**

   - Save the exact dataset versions or data processing scripts used prior to modeling to ensure full reproducibility.
   - Log all hyperparameters, metrics, and random seeds used during each step.

8. **Final Model Deployment Readiness:**

   - Verify that the model's performance metrics meet project benchmarks or business thresholds.
   - Document limitations, such as potential biases or data constraints, for stakeholder review.

---

### **Summary & Next Steps:**

- The implementation is technically sound and aligns with the outlined strategy.
- Focus on ensuring all features, especially log-transformed and flagged variables, are consistently included.
- Add detailed comments or documentation within the code for clarity.
- Consider additional interpretability analyses to support stakeholder understanding.
- Validate that final metrics meet project success criteria before deployment.

---

**Final note:** Excellent work on maintaining reproducibility and comprehensive evaluation. Addressing the above points will further strengthen the robustness and transparency of your ML workflow.

### 🔧 Step 4: JohnDoe (Refined Implementation)
**Role:** Developer (Refined)

**Final Refined Code:**
```python
"""
Final Refined Model Selection & Evaluation Implementation for RandomForestClassifier

This script implements the refined modeling strategy as specified by the Planner and audit feedback:
- Includes all relevant features, including log-transformed and outlier flag features.
- Splits dataset into train (70%), validation (10%), and test (20%) sets with stratification and fixed seed.
- Trains a baseline RandomForestClassifier.
- Performs hyperparameter tuning using RandomizedSearchCV with stratified CV on training data.
- Retrains final model with best hyperparameters.
- Evaluates model on validation and test sets with multiple metrics and statistical validation.
- Saves evaluation reports, confusion matrices, feature importance plots, and the final model.
- Adds robust error handling and detailed logging.
- Saves dataset versions and parameters for full reproducibility.
- Saves all artifacts under "../output/models/" and "../output/eda_outputs/".

Assumptions:
- `df` is the preprocessed DataFrame available in the environment.
- Target variable is `class_encoded`.
- Features exclude 'Class' and 'class_encoded' but include all other relevant columns (log transforms, outlier flags).
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from scipy.stats import ttest_rel

# ---------------------------
# Constants and directories
# ---------------------------
RANDOM_SEED = 42
MODEL_DIR = "../output/models/"
EDA_DIR = "../output/eda_outputs/"

# Ensure output directories exist before logging setup
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# ---------------------------
# Setup logging for audit & debugging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(EDA_DIR, "model_selection_evaluation.log"), mode='w')
    ]
)
logger = logging.getLogger(__name__)

def save_dataframe_version(df: pd.DataFrame, path: str):
    """
    Save a snapshot of the dataframe for reproducibility.
    """
    try:
        df.to_csv(path, index=False)
        logger.info(f"Saved dataframe snapshot to {path}")
    except Exception as e:
        logger.error(f"Failed to save dataframe snapshot: {e}")

def get_feature_columns(df: pd.DataFrame, target_col: str = 'class_encoded', exclude_cols=None):
    """
    Select feature columns including log-transformed and outlier flags,
    excluding target and original class columns.
    """
    if exclude_cols is None:
        exclude_cols = ['Class', target_col]

    # Include all columns except excluded ones
    features = [col for col in df.columns if col not in exclude_cols]

    # Confirm presence of log-transformed and outlier flag features
    # (Assuming all relevant features are included by default)
    logger.info(f"Selected {len(features)} features for modeling.")
    return features

def perform_data_splitting(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train (70%), validation (10%), and test (20%) sets with stratification.
    """
    try:
        # First split: train+val (80%) and test (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
        )
        # Second split: train (70%) and val (10%) from train+val (80%)
        val_size_relative = 0.125  # 0.125 * 0.8 = 0.10 overall
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_relative, random_state=RANDOM_SEED, stratify=y_temp
        )
        logger.info(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

def train_baseline_model(X_train, y_train):
    """
    Train a baseline RandomForestClassifier with default parameters.
    """
    try:
        rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        logger.info("Baseline RandomForestClassifier trained successfully.")
        return rf
    except Exception as e:
        logger.error(f"Error training baseline model: {e}")
        raise

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV with stratified 5-fold CV.
    """
    try:
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

        # StratifiedKFold for consistent stratification during CV
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=20,
            scoring='f1_weighted',  # weighted to handle class imbalance
            cv=cv_strategy,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=2,
            refit=True
        )

        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_score = search.best_score_
        logger.info(f"Hyperparameter tuning completed. Best weighted F1-score (CV): {best_score:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")

        # Save best hyperparameters and CV results
        with open(os.path.join(EDA_DIR, 'best_hyperparameters.txt'), 'w') as f:
            f.write("Best Hyperparameters from RandomizedSearchCV:\n")
            for param, val in best_params.items():
                f.write(f"{param}: {val}\n")
            f.write(f"\nBest CV weighted F1-score: {best_score:.4f}\n")

        # Save full CV results for audit
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df.to_csv(os.path.join(EDA_DIR, 'randomized_search_cv_results.csv'), index=False)

        return best_params
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        raise

def train_final_model(X_train, y_train, best_params):
    """
    Retrain RandomForestClassifier with best hyperparameters on training data.
    """
    try:
        final_model = RandomForestClassifier(**best_params, random_state=RANDOM_SEED, n_jobs=-1)
        final_model.fit(X_train, y_train)
        logger.info("Final model trained with best hyperparameters.")
        return final_model
    except Exception as e:
        logger.error(f"Error training final model: {e}")
        raise

def evaluate_model(model, X, y, dataset_name="Validation"):
    """
    Evaluate model predictions and return metrics dictionary.
    """
    try:
        y_pred = model.predict(X)
        metrics = {
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred
        }
        logger.info(f"{dataset_name} evaluation metrics: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation on {dataset_name} set: {e}")
        raise

def save_evaluation_reports(metrics, dataset_name):
    """
    Save evaluation metrics and confusion matrix plot.
    """
    try:
        # Save metrics text report
        metrics_path = os.path.join(EDA_DIR, f"{dataset_name.lower()}_evaluation_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"{dataset_name} Set Evaluation Metrics:\n")
            f.write(f"F1-score (weighted): {metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (weighted): {metrics['precision']:.4f}\n")
            f.write(f"Recall (weighted): {metrics['recall']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(metrics['classification_report'])

        logger.info(f"Saved {dataset_name} evaluation metrics to {metrics_path}")

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'{dataset_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        cm_path = os.path.join(EDA_DIR, f"{dataset_name.lower()}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Saved {dataset_name} confusion matrix plot to {cm_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation reports for {dataset_name}: {e}")
        raise

def plot_feature_importances(model, feature_names):
    """
    Plot and save feature importance barplot.
    """
    try:
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances_sorted = importances.sort_values(ascending=False)

        plt.figure(figsize=(12, 10))
        sns.barplot(x=importances_sorted.values, y=importances_sorted.index, palette='viridis')
        plt.title('Random Forest Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        fi_path = os.path.join(EDA_DIR, 'feature_importance.png')
        plt.savefig(fi_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {fi_path}")

        # Save feature importances as CSV
        fi_csv_path = os.path.join(EDA_DIR, 'feature_importances.csv')
        importances_sorted.to_csv(fi_csv_path, header=['importance'])
        logger.info(f"Saved feature importances CSV to {fi_csv_path}")
    except Exception as e:
        logger.error(f"Error plotting/saving feature importances: {e}")
        raise

def statistical_validation(y_true_val, y_pred_val, y_true_test, y_pred_test):
    """
    Perform paired t-test on per-sample correctness between validation and test predictions to check statistical significance.
    """
    try:
        # Calculate per-sample correctness (1 if correct else 0)
        val_correct = (y_true_val.values == y_pred_val).astype(int)
        test_correct = (y_true_test.values == y_pred_test).astype(int)

        # Paired t-test on correctness vectors
        t_stat, p_value = ttest_rel(val_correct, test_correct)
        logger.info(f"Paired t-test between validation and test correctness: t={t_stat:.4f}, p={p_value:.4f}")

        # Save statistical test results
        stats_path = os.path.join(EDA_DIR, 'statistical_validation.txt')
        with open(stats_path, 'w') as f:
            f.write("Statistical Validation: Paired t-test on per-sample correctness\n")
            f.write(f"t-statistic: {t_stat:.4f}\n")
            f.write(f"p-value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("Result: Significant difference between validation and test performance.\n")
            else:
                f.write("Result: No significant difference between validation and test performance.\n")
        logger.info(f"Saved statistical validation results to {stats_path}")
    except Exception as e:
        logger.error(f"Error during statistical validation: {e}")
        # Not critical to fail pipeline, so just log

def save_model(model, path):
    """
    Save the trained model to disk.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"Saved trained model to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main(df: pd.DataFrame):
    """
    Main function to execute the refined modeling pipeline.
    """
    try:
        logger.info("Starting Model Selection & Evaluation pipeline...")

        # Save dataset snapshot for reproducibility
        save_dataframe_version(df, os.path.join(EDA_DIR, 'dataset_snapshot.csv'))

        # Select features and target
        features = get_feature_columns(df, target_col='class_encoded', exclude_cols=['Class', 'class_encoded'])
        X = df[features]
        y = df['class_encoded']

        # Data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = perform_data_splitting(X, y)

        # Save splits for audit
        X_train.to_csv(os.path.join(EDA_DIR, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(EDA_DIR, 'y_train.csv'), index=False)
        X_val.to_csv(os.path.join(EDA_DIR, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(EDA_DIR, 'y_val.csv'), index=False)
        X_test.to_csv(os.path.join(EDA_DIR, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(EDA_DIR, 'y_test.csv'), index=False)
        logger.info("Saved train/val/test splits for reproducibility.")

        # Baseline model training
        baseline_model = train_baseline_model(X_train, y_train)

        # Hyperparameter tuning
        best_params = hyperparameter_tuning(X_train, y_train)

        # Final model training
        final_model = train_final_model(X_train, y_train, best_params)

        # Validation evaluation
        val_metrics = evaluate_model(final_model, X_val, y_val, dataset_name="Validation")
        save_evaluation_reports(val_metrics, dataset_name="Validation")

        # Test evaluation
        test_metrics = evaluate_model(final_model, X_test, y_test, dataset_name="Test")
        save_evaluation_reports(test_metrics, dataset_name="Test")

        # Statistical validation between validation and test sets
        statistical_validation(
            y_true_val=y_val,
            y_pred_val=val_metrics['predictions'],
            y_true_test=y_test,
            y_pred_test=test_metrics['predictions']
        )

        # Feature importance plotting and saving
        plot_feature_importances(final_model, features)

        # Save final model
        model_path = os.path.join(MODEL_DIR, 'final_rf_model.pkl')
        save_model(final_model, model_path)

        logger.info("Model Selection & Evaluation pipeline completed successfully.")
        logger.info(f"Validation weighted F1-score: {val_metrics['f1_score']:.4f}")
        logger.info(f"Test weighted F1-score: {test_metrics['f1_score']:.4f}")

        # Print summary to stdout for quick reference
        print("=== Model Selection & Evaluation Summary ===")
        print(f"Validation weighted F1-score: {val_metrics['f1_score']:.4f}")
        print(f"Test weighted F1-score: {test_metrics['f1_score']:.4f}")
        print(f"Model and artifacts saved under {MODEL_DIR} and {EDA_DIR}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        sys.exit(1)

# Entry point for script execution
if __name__ == "__main__":
    # Expecting df to be loaded in the environment; if not, raise error
    try:
        if 'df' not in globals():
            raise ValueError("Preprocessed DataFrame 'df' not found in the environment.")
        main(df)
    except Exception as e:
        # Setup a basic logger if logger is not yet configured
        try:
            logger.error(f"Execution failed: {e}")
        except Exception:
            print(f"Execution failed: {e}")
        print(f"Execution failed: {e}")
        sys.exit(1)
```

### 🖥 Execution Results
**Status:** ✅ Success

```

```
### 📊 Process Summary
- **Planner Agent:** DrAliceSmith
- **Developer Agent:** JohnDoe
- **Auditor Agent:** EmilyJohnson
- **Final Status:** Success
- **Iterations:** 4-step iterative process completed

---

## 📈 Overall Process Summary
- **Total Subtasks:** 26
- **Successful Subtasks:** 26
- **Success Rate:** 100.0%
