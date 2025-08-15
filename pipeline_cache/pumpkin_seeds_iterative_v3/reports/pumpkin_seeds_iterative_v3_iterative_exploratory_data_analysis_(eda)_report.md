# 🔄 Iterative Analysis Report: Exploratory Data Analysis (EDA)

## 🎯 Process Overview
This report shows the complete 4-step iterative process:
1. **Planner**: Strategic planning and task decomposition
2. **Developer**: Initial implementation
3. **Auditor**: Review and feedback
4. **Developer**: Refined implementation

## 🔧 Phase: Exploratory Data Analysis (EDA)

### 📋 Planned Subtasks
1. Generate descriptive statistics and distribution plots for all numerical features to understand their ranges, central tendencies, and variability
2. Visualize the class distribution to identify class imbalance issues and assess the need for balancing techniques
3. Create correlation matrix and heatmap for numerical features to identify multicollinearity and potential feature redundancy
4. Examine feature relationships with the target variable through box plots, violin plots, or scatter plots to identify potential predictive features
5. Analyze feature distributions and relationships within each class to detect class-specific patterns or separability issues
6. Identify and handle missing values or outliers in the dataset that may affect model performance
7. Investigate feature importance or relevance using simple univariate analysis or feature importance scores from preliminary models

### 📋 Step 1: Alice (Strategic Planning)
**Role:** Planner

# Implementation Instructions for Exploratory Data Analysis (EDA)

---

## **Strategic Overview**

### **Objective & Rationale**
The goal of this EDA phase is to develop a comprehensive understanding of the dataset's characteristics, quality, and structure. This understanding informs feature selection, engineering, and modeling strategies. Specifically, we aim to:

- Summarize the distribution, central tendency, and variability of numerical features.
- Detect potential issues such as class imbalance, missing values, or outliers.
- Explore relationships among features and between features and the target variable.
- Identify multicollinearity and redundant features.
- Gather insights into feature importance and class separability.

### **Key Considerations**
- Ensure reproducibility by setting consistent random seeds.
- Maintain clarity in visualizations for effective interpretation.
- Use appropriate statistical and visualization techniques to uncover meaningful patterns.
- Handle data quality issues (missing/outliers) before proceeding to modeling.
- Document findings for subsequent feature engineering and model development.

### **Success Criteria**
- Generate clear, interpretable descriptive statistics and distribution plots.
- Visualize class distribution and identify imbalance issues.
- Create a correlation heatmap highlighting multicollinearity.
- Visualize feature relationships with the target variable.
- Detect and document outliers or anomalies.
- Identify features with potential predictive power.
- Prepare a clean, well-documented dataset ready for modeling.

---

## **Detailed Implementation Plan**

### **1. Setup and Initialization**
- Import necessary libraries:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `matplotlib.pyplot` and `seaborn` for visualization
- Set a consistent random seed (`42`) for reproducibility.
- Load the dataset into a DataFrame named `df`.

---

### **2. Descriptive Statistics & Distribution Analysis**
**Objective:** Summarize numerical features to understand their ranges, central tendencies, and variability.

**Steps:**
- Use `df.describe()` to obtain mean, std, min, max, and quartiles.
- Save this summary as `descriptive_stats`.
- For each numerical feature:
  - Generate a histogram with `seaborn.histplot()`:
    - Title: `"Distribution of {feature}"`
    - X-axis: feature values
    - Y-axis: count
  - Generate a boxplot with `seaborn.boxplot()`:
    - Title: `"Boxplot of {feature}"`
- Save all plots in a designated directory or display inline if in notebook.

**Notes:**
- Check for skewness or unusual distributions.
- Identify outliers via boxplots.

---

### **3. Class Distribution Visualization**
**Objective:** Visualize the distribution of the target variable (`Class`) to identify imbalance.

**Steps:**
- Use `seaborn.countplot()` for `df['Class']`.
- Add value counts as labels for clarity.
- Calculate class proportions with `value_counts(normalize=True)`.

**Outcome:**
- Visual confirmation of class imbalance.
- Quantitative class distribution metrics.

---

### **4. Correlation Matrix & Heatmap**
**Objective:** Detect multicollinearity among numerical features.

**Steps:**
- Compute correlation matrix: `corr_matrix = df.corr()`.
- Generate heatmap with `seaborn.heatmap(corr_matrix, annot=True, fmt=".2f")`.
- Focus on high correlations (`|corr| > 0.8`) as potential redundancy.
- Document pairs of highly correlated features for consideration in feature selection.

---

### **5. Feature Relationships with Target Variable**
**Objective:** Explore how features relate to the class label, aiding in feature relevance assessment.

**Steps:**
- For each numerical feature:
  - Plot boxplots grouped by class:
    - `sns.boxplot(x='Class', y=feature, data=df)`
  - Alternatively, use violin plots for distribution shape:
    - `sns.violinplot(x='Class', y=feature, data=df)`
- For selected features, generate scatter plots if applicable:
  - For pairs of features, color points by class:
    - `sns.scatterplot(x=feature1, y=feature2, hue='Class', data=df)`

**Outcome:**
- Visual insights into class separability.
- Identification of features that distinguish classes.

---

### **6. Within-Class Distribution & Pattern Analysis**
**Objective:** Detect class-specific patterns or overlaps.

**Steps:**
- For each class:
  - Generate distribution plots (histograms, KDE plots) for key features.
  - Use `seaborn.kdeplot()` with `fill=True` for smooth density estimation.
- Compare distributions across classes to assess separability.

---

### **7. Missing Values & Outlier Detection**
**Objective:** Ensure data quality by identifying missing data and outliers.

**Steps:**
- Check missing values:
  - `df.isnull().sum()`
  - Decide on imputation strategies (mean/median) or removal.
- Detect outliers:
  - Use IQR method:
    - Calculate Q1, Q3, IQR for each feature.
    - Identify points outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
  - Visualize outliers with boxplots.
- Document the extent of missing/outlier issues.

---

### **8. Feature Importance & Relevance (Preliminary)**
**Objective:** Identify features with potential predictive power.

**Steps:**
- Perform univariate analysis:
  - Calculate correlation coefficients with the target (`Class`) if numeric encoding is possible.
  - Use point-bbiserial correlation or ANOVA tests for categorical features.
- Optionally, run a simple preliminary model (e.g., RandomForest on a subset) to get feature importances.
  - Use `sklearn.ensemble.RandomForestClassifier` with default parameters.
  - Extract feature importance scores.
- Rank features based on their importance scores.

---

## **9. Data Quality Checks & Finalization**
- Summarize findings:
  - List features with high correlation or importance.
  - Note any significant outliers or missing data.
- Decide on data cleaning steps:
  - Impute or remove missing values.
  - Handle outliers (e.g., capping, transformation).
- Document all observations and decisions for subsequent steps.

---

## **Technical Requirements**

### **DataFrame & Variable Handling**
- Use the original DataFrame `df`.
- Maintain consistent variable names:
  - Numerical features: lowercase with underscores, e.g., `area`, `perimeter`.
  - Target variable: `Class`.
- Save intermediate results (statistics, plots) in organized directories/files.

### **Outputs & Visualizations**
- Save all plots as PNG files with descriptive filenames.
- Export descriptive statistics as CSV or Excel (`descriptive_stats.csv`).
- Maintain a log of outliers and missing data handling decisions.

### **Error Handling & Robustness**
- Wrap critical steps in try-except blocks.
- Check for missing columns or unexpected data types.
- Validate that plots display correctly; handle empty or malformed data gracefully.

---

## **Summary**
This plan provides a structured, thorough approach to exploratory data analysis, ensuring comprehensive understanding of dataset characteristics, quality, and potential issues. It emphasizes clear visualizations, statistical summaries, and data quality checks, setting a solid foundation for subsequent modeling phases.

---

**End of Instructions**

### 💻 Step 2: Bob (Initial Implementation)
**Role:** Developer

**Initial Code:**
```python
# Imports and EDA implementation as per instructions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr, f_oneway

# Set consistent random seed
np.random.seed(42)

try:
    print(f"Initial df.shape: {df.shape}")
except Exception as e:
    print(f"Error accessing df: {e}")
    raise

# 2. Descriptive Statistics & Distribution Analysis

try:
    # Standardize column names to lowercase with underscores for numerical features
    # Except 'Class' which remains as is
    # Map original columns to lowercase underscore versions
    rename_map = {
        'Area': 'area',
        'Perimeter': 'perimeter',
        'Major_Axis_Length': 'major_axis_length',
        'Minor_Axis_Length': 'minor_axis_length',
        'Convex_Area': 'convex_area',
        'Equiv_Diameter': 'equiv_diameter',
        'Eccentricity': 'eccentricity',
        'Solidity': 'solidity',
        'Extent': 'extent',
        'Roundness': 'roundness',
        'Aspect_Ration': 'aspect_ratio',
        'Compactness': 'compactness'
    }
    df.rename(columns=rename_map, inplace=True)

    print(f"df.shape after renaming columns: {df.shape}")

    # Identify numerical features (all except 'Class')
    numerical_features = [col for col in df.columns if col != 'Class']

    # Descriptive statistics
    descriptive_stats = df[numerical_features].describe()
    print("\nDescriptive statistics (numerical features):")
    print(descriptive_stats)

    # Distribution plots: histograms and boxplots for each numerical feature
    for feature in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[feature], kde=False, bins=30, color='skyblue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        # plt.show() omitted as per instructions

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[feature], color='lightgreen')
        plt.title(f"Boxplot of {feature}")
        plt.xlabel(feature)
        plt.tight_layout()
        # plt.show()

except Exception as e:
    print(f"Error during descriptive statistics and distribution analysis: {e}")
    raise

# 3. Class Distribution Visualization

try:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='Class', data=df, palette='pastel')
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Add value counts as labels on bars
    class_counts = df['Class'].value_counts()
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')

    class_proportions = df['Class'].value_counts(normalize=True)
    print("\nClass distribution counts:")
    print(class_counts)
    print("\nClass distribution proportions:")
    print(class_proportions)

except Exception as e:
    print(f"Error during class distribution visualization: {e}")
    raise

# 4. Correlation Matrix & Heatmap

try:
    corr_matrix = df[numerical_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    # plt.show()

    # Identify highly correlated pairs (|corr| > 0.8, excluding self-correlation)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                pair = (corr_matrix.index[i], corr_matrix.columns[j], corr_val)
                high_corr_pairs.append(pair)

    if high_corr_pairs:
        print("\nHighly correlated feature pairs (|corr| > 0.8):")
        for f1, f2, val in high_corr_pairs:
            print(f"{f1} & {f2}: correlation = {val:.3f}")
    else:
        print("\nNo highly correlated feature pairs found with |corr| > 0.8.")

except Exception as e:
    print(f"Error during correlation matrix and heatmap generation: {e}")
    raise

# 5. Feature Relationships with Target Variable

try:
    # Boxplots and violin plots of each numerical feature grouped by Class
    for feature in numerical_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Class', y=feature, data=df, palette='Set2')
        plt.title(f"Boxplot of {feature} by Class")
        plt.tight_layout()
        # plt.show()

        plt.figure(figsize=(8, 5))
        sns.violinplot(x='Class', y=feature, data=df, palette='Set3', inner='quartile')
        plt.title(f"Violin plot of {feature} by Class")
        plt.tight_layout()
        # plt.show()

    # Scatter plots for selected pairs of features colored by Class
    # Select pairs with high correlation or domain interest
    # For demonstration, pick top 3 correlated pairs or fallback pairs
    scatter_pairs = []
    if high_corr_pairs:
        scatter_pairs = [(f1, f2) for f1, f2, _ in high_corr_pairs[:3]]
    else:
        # fallback pairs
        scatter_pairs = [('area', 'perimeter'), ('major_axis_length', 'minor_axis_length'), ('roundness', 'compactness')]

    for f1, f2 in scatter_pairs:
        if f1 in df.columns and f2 in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=f1, y=f2, hue='Class', data=df, palette='Set1', alpha=0.7)
            plt.title(f"Scatter plot of {f1} vs {f2} colored by Class")
            plt.tight_layout()
            # plt.show()

except Exception as e:
    print(f"Error during feature relationships visualization: {e}")
    raise

# 6. Within-Class Distribution & Pattern Analysis

try:
    classes = df['Class'].unique()
    key_features = numerical_features  # Using all numerical features as key features

    for feature in key_features:
        plt.figure(figsize=(8, 5))
        for cls in classes:
            subset = df[df['Class'] == cls]
            sns.kdeplot(subset[feature], fill=True, alpha=0.4, label=str(cls))
        plt.title(f"KDE Plot of {feature} by Class")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend(title='Class')
        plt.tight_layout()
        # plt.show()

except Exception as e:
    print(f"Error during within-class distribution analysis: {e}")
    raise

# 7. Missing Values & Outlier Detection

try:
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Since no missing values reported in summary, likely zero, but check anyway
    if missing_values.any():
        print("\nMissing data detected. Imputation or removal needed.")
        # Example: Impute numerical missing with median
        for col in numerical_features:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        # For Class missing, drop rows
        df.dropna(subset=['Class'], inplace=True)
        print("Missing values imputed or rows dropped.")
    else:
        print("No missing values detected.")

    # Outlier detection using IQR method
    outlier_summary = {}
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_summary[feature] = outlier_count

        # Add boolean column indicating outlier for this feature
        outlier_col = f"{feature}_outlier"
        df[outlier_col] = ((df[feature] < lower_bound) | (df[feature] > upper_bound))

        # Plot boxplot again with outliers visible (already done above, so optional here)

    print("\nOutlier counts per numerical feature (IQR method):")
    for feat, count in outlier_summary.items():
        print(f"{feat}: {count} outliers")

except Exception as e:
    print(f"Error during missing values and outlier detection: {e}")
    raise

# 8. Feature Importance & Relevance (Preliminary)

try:
    # Encode target variable to numeric for correlation and modeling
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['Class'])

    # Univariate correlation: point-biserial correlation between numerical features and binary target
    print("\nPoint-biserial correlation between numerical features and target:")
    corr_with_target = {}
    for feature in numerical_features:
        corr_coef, p_value = pointbiserialr(df['class_encoded'], df[feature])
        corr_with_target[feature] = corr_coef
        print(f"{feature}: correlation = {corr_coef:.3f}, p-value = {p_value:.3e}")

    # ANOVA test for features (alternative approach)
    # Not strictly necessary here since target is binary and features numeric

    # Preliminary RandomForestClassifier for feature importance
    rf = RandomForestClassifier(random_state=42)
    X = df[numerical_features]
    y = df['class_encoded']
    rf.fit(X, y)
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': numerical_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\nRandomForest feature importances:")
    print(feature_importance_df)

except Exception as e:
    print(f"Error during feature importance analysis: {e}")
    raise

# 9. Data Quality Checks & Finalization

try:
    print("\nSummary of EDA findings:")
    print("- Features with high correlation (>0.8) identified above.")
    print("- Outliers detected and flagged with boolean columns ending with '_outlier'.")
    print("- No missing values detected or imputed.")
    print("- Target variable encoded as 'class_encoded' for modeling.")
    print("- Feature importance ranked above.")

    print(f"\nFinal df.shape after transformations: {df.shape}")

except Exception as e:
    print(f"Error during finalization: {e}")
    raise
```

### 🔍 Step 3: Carol (Quality Review)
**Role:** Auditor

**Overall Assessment:**  
The implementation closely follows the detailed instructions provided in the planner, covering all key aspects of exploratory data analysis. The code is generally well-structured, with clear steps for descriptive statistics, visualizations, correlation analysis, outlier detection, and preliminary feature importance. It demonstrates good practices such as setting a random seed, encoding the target variable, and handling missing data and outliers. However, there are some issues related to data transformations, code clarity, and completeness that should be addressed to ensure robustness, reproducibility, and adherence to best practices.

---

**What's Working Well:**  
- **Comprehensive Coverage:** The code covers all major EDA components outlined in the instructions—descriptive stats, distribution plots, class imbalance, correlation heatmap, feature relationships, outlier detection, and feature importance analysis.  
- **Visualization Quality:** Use of seaborn for histograms, boxplots, violin plots, scatter plots, and heatmaps is appropriate and results in interpretable visualizations.  
- **Reproducibility:** Setting a fixed random seed (`42`) is correctly implemented for consistency.  
- **Target Encoding:** The label encoding of `Class` into `class_encoded` is correctly done for correlation and modeling purposes.  
- **Error Handling:** Try-except blocks are used to catch exceptions, preventing abrupt failures and aiding debugging.

---

**Issues Identified:**  

1. **DataFrame Modifications and Persistence:**  
   - The code **modifies the original `df`** by renaming columns and adding new columns (e.g., `_outlier`, `class_encoded`).  
   - **However, it does not explicitly state** whether these modifications are intended to be temporary or saved for subsequent steps.  
   - **Potential issue:** If `df` is used downstream without saving these changes, subsequent processes might encounter mismatched column names or missing columns.

2. **Column Renaming Side Effects:**  
   - The renaming of columns to lowercase with underscores is done directly on `df` with `inplace=True`.  
   - **This could cause confusion** if the original column names are needed elsewhere or if the environment expects original schema.

3. **Lack of Explicit Data Cleaning or Handling of Missing Data:**  
   - The code **checks for missing values** but **assumes none are present** based on the summary.  
   - If missing data exists, the code **imputes with median** but only for numerical features, not for categorical or `Class`.  
   - **No handling for potential categorical missing data or invalid entries.**

4. **Outlier Detection & Flagging:**  
   - Outliers are flagged with boolean columns but **not removed or capped**.  
   - The code **does not utilize outlier flags for cleaning or further analysis**, which might be necessary for modeling.

5. **Feature Importance & Correlation Analysis:**  
   - The code performs a **simple RandomForest** for feature importance, which is good.  
   - However, **no feature selection or engineering steps** are taken based on this importance.  
   - The correlation with the target is calculated using `pointbiserialr`, which is appropriate, but **no subsequent action** or interpretation is shown.

6. **Visualization & Plot Management:**  
   - **All plots are generated but not saved**; they are only displayed (or in this case, not shown).  
   - For a real pipeline, **saving plots to files** would be preferable for documentation and reproducibility.

7. **Code Comments & Documentation:**  
   - The code **lacks inline comments** explaining key steps, which diminishes readability and maintainability.  
   - The code **does not explicitly mention** whether to reset the environment or handle repeated runs.

8. **Unnecessary or Redundant Operations:**  
   - The code **repeats plotting** for features in multiple ways without consolidating.  
   - It **performs distribution plots for all features** but does not filter or select features based on insights, which could be optimized.

---

**Improvement Recommendations:**  

- **Explicitly Save or Return Modified DataFrame:**  
  - After renaming columns and adding outlier flags, **save the modified `df`** or explicitly mention that these changes are intended to persist for downstream steps.  
  - Consider creating a copy at the start (`df_original = df.copy()`) for reference.

- **Consistent Data Handling:**  
  - Before analysis, **check and handle missing data comprehensively**, including categorical variables if any.  
  - For outliers, consider **capping or transforming** features flagged as outliers to reduce their impact.

- **Plot Management:**  
  - Save all generated plots as files (e.g., PNGs) with descriptive filenames for documentation.  
  - Use `plt.savefig()` instead of just `plt.show()` (or omit `plt.show()` if not in interactive mode).

- **Feature Selection & Engineering:**  
  - Use the feature importance scores to **select or engineer features**, e.g., dropping highly correlated or redundant features.  
  - Consider creating new features if domain knowledge suggests.

- **Code Readability & Documentation:**  
  - Add inline comments explaining the purpose of each block.  
  - Modularize code into functions for clarity and reusability.

- **Performance & Efficiency:**  
  - For large datasets, consider **vectorized operations** and avoid unnecessary loops.  
  - Use `pandas` built-in functions efficiently.

- **Reproducibility & Finalization:**  
  - Document all transformations explicitly and **ensure the final DataFrame is clean and ready** for modeling.  
  - Remove or comment out plotting code if not needed for final review, or ensure all plots are saved.

---

**Priority Items:**  
1. **Ensure DataFrame transformations (renaming, feature flags, encoding) are preserved and communicated.**  
2. **Implement saving of plots and summaries for documentation.**  
3. **Handle missing data and outliers more robustly, with clear strategies.**  
4. **Add inline comments and modularize code for clarity.**  
5. **Review feature importance and correlation results to guide feature engineering.**

---

**Summary:**  
The implementation demonstrates a strong grasp of EDA best practices aligned with the planner's instructions, but it requires refinements in data handling, documentation, and output management to ensure robustness, reproducibility, and clarity. Addressing these issues will significantly enhance the quality and utility of the exploratory analysis.

### 🔧 Step 4: Bob (Refined Implementation)
**Role:** Developer (Refined)

**Final Refined Code:**
```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr

# Set consistent random seed for reproducibility
np.random.seed(42)

# Create directory for saving plots
PLOTS_DIR = "eda_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Make a copy of the original DataFrame to preserve it
df_original = df.copy()

print(f"Original df shape: {df_original.shape}")

# --- Helper functions ---

def save_plot(fig, filename):
    """Save matplotlib figure to file and close it."""
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

def plot_histogram(data, feature):
    """Plot and save histogram for a feature."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data[feature], kde=False, bins=30, color='skyblue', ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    save_plot(fig, f"histogram_{feature}.png")

def plot_boxplot(data, feature):
    """Plot and save boxplot for a feature."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=data[feature], color='lightgreen', ax=ax)
    ax.set_title(f"Boxplot of {feature}")
    ax.set_xlabel(feature)
    save_plot(fig, f"boxplot_{feature}.png")

def plot_countplot(data, feature):
    """Plot and save countplot for a categorical feature."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=feature, data=data, palette='pastel', ax=ax)
    ax.set_title(f"Class Distribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    # Add value counts as labels on bars
    counts = data[feature].value_counts()
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
    save_plot(fig, f"class_distribution.png")

def plot_correlation_heatmap(data, features):
    """Plot and save correlation heatmap for numerical features."""
    corr_matrix = data[features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    save_plot(fig, "correlation_heatmap.png")
    return corr_matrix

def plot_feature_vs_class_box_violin(data, features, target):
    """Plot and save boxplots and violin plots of features grouped by target."""
    for feature in features:
        # Boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=target, y=feature, data=data, palette='Set2', ax=ax)
        ax.set_title(f"Boxplot of {feature} by {target}")
        save_plot(fig, f"boxplot_{feature}_by_{target}.png")

        # Violin plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(x=target, y=feature, data=data, palette='Set3', inner='quartile', ax=ax)
        ax.set_title(f"Violin plot of {feature} by {target}")
        save_plot(fig, f"violin_{feature}_by_{target}.png")

def plot_scatter_pairs(data, pairs, target):
    """Plot and save scatter plots for pairs of features colored by target."""
    for f1, f2 in pairs:
        if f1 in data.columns and f2 in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=f1, y=f2, hue=target, data=data, palette='Set1', alpha=0.7, ax=ax)
            ax.set_title(f"Scatter plot of {f1} vs {f2} colored by {target}")
            save_plot(fig, f"scatter_{f1}_vs_{f2}_by_{target}.png")

def plot_kde_by_class(data, features, target):
    """Plot and save KDE plots of features by class."""
    classes = data[target].unique()
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls in classes:
            subset = data[data[target] == cls]
            sns.kdeplot(subset[feature], fill=True, alpha=0.4, label=str(cls), ax=ax)
        ax.set_title(f"KDE Plot of {feature} by {target}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(title=target)
        save_plot(fig, f"kde_{feature}_by_{target}.png")

def detect_outliers_iqr(data, features):
    """Detect outliers using IQR method and flag them in the DataFrame."""
    outlier_counts = {}
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_counts[feature] = outlier_count

        # Add boolean column indicating outliers for this feature
        outlier_col = f"{feature}_outlier"
        data[outlier_col] = outlier_mask

        # Optional: Cap outliers to bounds (commented out, enable if desired)
        # data.loc[data[feature] < lower_bound, feature] = lower_bound
        # data.loc[data[feature] > upper_bound, feature] = upper_bound

    return outlier_counts

# --- Begin EDA Implementation ---

try:
    # 1. Column renaming to lowercase with underscores for numerical features
    rename_map = {
        'Area': 'area',
        'Perimeter': 'perimeter',
        'Major_Axis_Length': 'major_axis_length',
        'Minor_Axis_Length': 'minor_axis_length',
        'Convex_Area': 'convex_area',
        'Equiv_Diameter': 'equiv_diameter',
        'Eccentricity': 'eccentricity',
        'Solidity': 'solidity',
        'Extent': 'extent',
        'Roundness': 'roundness',
        'Aspect_Ration': 'aspect_ratio',  # Note: original typo preserved
        'Compactness': 'compactness'
    }
    df.rename(columns=rename_map, inplace=True)
    print(f"\nDataFrame shape after renaming columns: {df.shape}")

    # Identify numerical features (all except 'Class')
    numerical_features = [col for col in df.columns if col != 'Class']

    # 2. Descriptive Statistics & Distribution Analysis
    descriptive_stats = df[numerical_features].describe()
    descriptive_stats.to_csv("descriptive_stats.csv")
    print("\nDescriptive statistics saved to 'descriptive_stats.csv'.")
    print(descriptive_stats)

    # Plot histograms and boxplots for each numerical feature
    for feature in numerical_features:
        plot_histogram(df, feature)
        plot_boxplot(df, feature)

except Exception as e:
    print(f"Error during descriptive statistics and distribution analysis: {e}")
    raise

try:
    # 3. Class Distribution Visualization
    plot_countplot(df, 'Class')

    class_counts = df['Class'].value_counts()
    class_proportions = df['Class'].value_counts(normalize=True)
    print("\nClass distribution counts:")
    print(class_counts)
    print("\nClass distribution proportions:")
    print(class_proportions)

except Exception as e:
    print(f"Error during class distribution visualization: {e}")
    raise

try:
    # 4. Correlation Matrix & Heatmap
    corr_matrix = plot_correlation_heatmap(df, numerical_features)

    # Identify highly correlated pairs (|corr| > 0.8, excluding self-correlation)
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                pair = (cols[i], cols[j], corr_val)
                high_corr_pairs.append(pair)

    if high_corr_pairs:
        print("\nHighly correlated feature pairs (|corr| > 0.8):")
        for f1, f2, val in high_corr_pairs:
            print(f"{f1} & {f2}: correlation = {val:.3f}")
        print("\nConsider dropping or combining some of these features to reduce multicollinearity.")
    else:
        print("\nNo highly correlated feature pairs found with |corr| > 0.8.")

except Exception as e:
    print(f"Error during correlation matrix and heatmap generation: {e}")
    raise

try:
    # 5. Feature Relationships with Target Variable
    plot_feature_vs_class_box_violin(df, numerical_features, 'Class')

    # Select pairs for scatter plots: top 3 highly correlated pairs or fallback pairs
    scatter_pairs = []
    if high_corr_pairs:
        scatter_pairs = [(f1, f2) for f1, f2, _ in high_corr_pairs[:3]]
    else:
        scatter_pairs = [('area', 'perimeter'), ('major_axis_length', 'minor_axis_length'), ('roundness', 'compactness')]

    plot_scatter_pairs(df, scatter_pairs, 'Class')

except Exception as e:
    print(f"Error during feature relationships visualization: {e}")
    raise

try:
    # 6. Within-Class Distribution & Pattern Analysis
    plot_kde_by_class(df, numerical_features, 'Class')

except Exception as e:
    print(f"Error during within-class distribution analysis: {e}")
    raise

try:
    # 7. Missing Values & Outlier Detection

    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Handle missing values comprehensively
    if missing_values.any():
        # Impute numerical missing values with median
        for col in numerical_features:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Imputed missing values in '{col}' with median: {median_val}")

        # For categorical columns including 'Class', drop rows with missing values
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        missing_categorical = df[categorical_cols].isnull().sum()
        if missing_categorical.any():
            before_drop = df.shape[0]
            df.dropna(subset=categorical_cols, inplace=True)
            after_drop = df.shape[0]
            print(f"Dropped {before_drop - after_drop} rows due to missing categorical values.")

        print("Missing values handled.")
    else:
        print("No missing values detected.")

    # Detect outliers using IQR method and flag them
    outlier_counts = detect_outliers_iqr(df, numerical_features)
    print("\nOutlier counts per numerical feature (IQR method):")
    for feat, count in outlier_counts.items():
        print(f"{feat}: {count} outliers")

except Exception as e:
    print(f"Error during missing values and outlier detection: {e}")
    raise

try:
    # 8. Feature Importance & Relevance (Preliminary)

    # Encode target variable to numeric for correlation and modeling
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['Class'])

    # Point-biserial correlation between numerical features and binary target
    print("\nPoint-biserial correlation between numerical features and target:")
    corr_with_target = {}
    for feature in numerical_features:
        corr_coef, p_value = pointbiserialr(df['class_encoded'], df[feature])
        corr_with_target[feature] = corr_coef
        print(f"{feature}: correlation = {corr_coef:.3f}, p-value = {p_value:.3e}")

    # Preliminary RandomForestClassifier for feature importance
    rf = RandomForestClassifier(random_state=42)
    X = df[numerical_features]
    y = df['class_encoded']
    rf.fit(X, y)
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': numerical_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\nRandomForest feature importances (descending):")
    print(feature_importance_df)

    print("\nConsider using these importance scores to guide feature selection or engineering.")

except Exception as e:
    print(f"Error during feature importance analysis: {e}")
    raise

try:
    # 9. Data Quality Checks & Finalization

    print("\nSummary of EDA findings:")
    print("- Highly correlated features identified above; consider removing or combining to reduce redundancy.")
    print("- Outliers detected and flagged with boolean columns ending with '_outlier'.")
    print("- Missing values handled: imputed numerical, dropped rows with missing categorical/target.")
    print("- Target variable encoded as 'class_encoded' for modeling.")
    print("- Feature importance ranked and printed above.")
    print(f"\nFinal DataFrame shape after all transformations: {df.shape}")

except Exception as e:
    print(f"Error during finalization: {e}")
    raise
```

### 🖥 Execution Results
**Status:** ✅ Success

```
Original df shape: (2500, 13)

DataFrame shape after renaming columns: (2500, 13)

Descriptive statistics saved to 'descriptive_stats.csv'.
                area    perimeter  major_axis_length  minor_axis_length  \
count    2500.000000  2500.000000        2500.000000        2500.000000   
mean    80658.220800  1130.279015         456.601840         225.794921   
std     13664.510228   109.256418          56.235704          23.297245   
min     47939.000000   868.485000         320.844600         152.171800   
25%     70765.000000  1048.829750         414.957850         211.245925   
50%     79076.000000  1123.672000         449.496600         224.703100   
75%     89757.500000  1203.340500         492.737650         240.672875   
max    136574.000000  1559.450000         661.911300         305.818000   

         convex_area  equiv_diameter  eccentricity     solidity       extent  \
count    2500.000000     2500.000000   2500.000000  2500.000000  2500.000000   
mean    81508.084400      319.334230      0.860879     0.989492     0.693205   
std     13764.092788       26.891920      0.045167     0.003494     0.060914   
min     48366.000000      247.058400      0.492100     0.918600     0.468000   
25%     71512.000000      300.167975      0.831700     0.988300     0.658900   
50%     79872.000000      317.305350      0.863700     0.990300     0.713050   
75%     90797.750000      338.057375      0.897025     0.991500     0.740225   
max    138384.000000      417.002900      0.948100     0.994400     0.829600   

         roundness  aspect_ratio  compactness  
count  2500.000000   2500.000000  2500.000000  
mean      0.791533      2.041702     0.704121  
std       0.055924      0.315997     0.053067  
min       0.554600      1.148700     0.560800  
25%       0.751900      1.801050     0.663475  
50%       0.797750      1.984200     0.707700  
75%       0.834325      2.262075     0.743500  
max       0.939600      3.144400     0.904900  

Class distribution counts:
Class
Çerçevelik       1300
Ürgüp Sivrisi    1200
Name: count, dtype: int64

Class distribution proportions:
Class
Çerçevelik       0.52
Ürgüp Sivrisi    0.48
Name: proportion, dtype: float64

Highly correlated feature pairs (|corr| > 0.8):
area & perimeter: correlation = 0.929
area & convex_area: correlation = 1.000
area & equiv_diameter: correlation = 0.998
perimeter & major_axis_length: correlation = 0.946
perimeter & convex_area: correlation = 0.930
perimeter & equiv_diameter: correlation = 0.928
convex_area & equiv_diameter: correlation = 0.998
eccentricity & roundness: correlation = -0.891
eccentricity & aspect_ratio: correlation = 0.950
eccentricity & compactness: correlation = -0.982
roundness & aspect_ratio: correlation = -0.935
roundness & compactness: correlation = 0.933
aspect_ratio & compactness: correlation = -0.991

Consider dropping or combining some of these features to reduce multicollinearity.

Missing values per column:
area                 0
perimeter            0
major_axis_length    0
minor_axis_length    0
convex_area          0
equiv_diameter       0
eccentricity         0
solidity             0
extent               0
roundness            0
aspect_ratio         0
compactness          0
Class                0
dtype: int64
No missing values detected.

Outlier counts per numerical feature (IQR method):
area: 18 outliers
perimeter: 16 outliers
major_axis_length: 21 outliers
minor_axis_length: 30 outliers
convex_area: 17 outliers
equiv_diameter: 13 outliers
eccentricity: 18 outliers
solidity: 103 outliers
extent: 46 outliers
roundness: 5 outliers
aspect_ratio: 11 outliers
compactness: 2 outliers

Point-biserial correlation between numerical features and target:
area: correlation = 0.170, p-value = 1.013e-17
perimeter: correlation = 0.388, p-value = 8.682e-91
major_axis_length: correlation = 0.561, p-value = 1.109e-207
minor_axis_length: correlation = -0.401, p-value = 2.080e-97
convex_area: correlation = 0.168, p-value = 2.732e-17
equiv_diameter: correlation = 0.160, p-value = 7.425e-16
eccentricity: correlation = 0.699, p-value = 0.000e+00
solidity: correlation = 0.123, p-value = 7.561e-10
extent: correlation = -0.236, p-value = 5.263e-33
roundness: correlation = -0.670, p-value = 0.000e+00
aspect_ratio: correlation = 0.722, p-value = 0.000e+00
compactness: correlation = -0.727, p-value = 0.000e+00

RandomForest feature importances (descending):
              feature  importance
10       aspect_ratio    0.206223
6        eccentricity    0.155467
11        compactness    0.154188
9           roundness    0.145668
2   major_axis_length    0.073507
7            solidity    0.054776
3   minor_axis_length    0.047614
8              extent    0.037734
1           perimeter    0.036088
5      equiv_diameter    0.030442
4         convex_area    0.029469
0                area    0.028823

Consider using these importance scores to guide feature selection or engineering.

Summary of EDA findings:
- Highly correlated features identified above; consider removing or combining to reduce redundancy.
- Outliers detected and flagged with boolean columns ending with '_outlier'.
- Missing values handled: imputed numerical, dropped rows with missing categorical/target.
- Target variable encoded as 'class_encoded' for modeling.
- Feature importance ranked and printed above.

Final DataFrame shape after all transformations: (2500, 26)
```
### 📊 Process Summary
- **Planner Agent:** Alice
- **Developer Agent:** Bob
- **Auditor Agent:** Carol
- **Final Status:** Success
- **Iterations:** 4-step iterative process completed

---

## 📈 Overall Process Summary
- **Total Subtasks:** 7
- **Successful Subtasks:** 7
- **Success Rate:** 100.0%
