# 🔄 Iterative Analysis Report: Exploratory Data Analysis (EDA)

## 🎯 Process Overview
This report shows the complete 4-step iterative process:
1. **Planner**: Strategic planning and task decomposition
2. **Developer**: Initial implementation
3. **Auditor**: Review and feedback
4. **Developer**: Refined implementation

## 🔧 Phase: Exploratory Data Analysis (EDA)

### 📋 Planned Subtasks
1. Generate descriptive statistics and visualize distributions for all numerical features to understand their ranges, central tendencies, and variability
2. Analyze the class distribution to identify potential imbalance issues and visualize class proportions
3. Create pairwise scatter plots and correlation heatmaps of numerical features to identify relationships, multicollinearity, and potential feature redundancies
4. Investigate feature distributions across different classes using boxplots or violin plots to detect class-separating features
5. Examine the presence of outliers and their potential impact on features using boxplots and z-score analysis
6. Assess feature importance and relevance through initial feature importance analysis or univariate statistical tests to prioritize features for modeling
7. Explore potential feature transformations or engineering opportunities based on insights from distributions, correlations, and class separability

### 📋 Step 1: Alice (Strategic Planning)
**Role:** Planner

# Exploratory Data Analysis (EDA) Implementation Instructions

---

## **Strategic Overview**

### **Objective:**
The goal of this phase is to thoroughly understand the dataset's features, their distributions, relationships, and potential issues such as class imbalance and outliers. Insights gained here will inform feature engineering, model selection, and hyperparameter tuning, ultimately leading to a robust classification model.

### **Why:**
A comprehensive EDA helps identify data quality issues, feature relevance, and potential redundancies. It also uncovers patterns that influence model performance, ensuring informed decisions and effective feature engineering.

### **Key Considerations:**
- Focus on both numerical and categorical features.
- Visualizations should be clear, labeled, and interpretable.
- Handle missing or anomalous data carefully.
- Maintain reproducibility with consistent random states.
- Document findings and observations during each step.

### **Success Criteria:**
- Complete descriptive statistics summaries for all numerical features.
- Visualizations of feature distributions, class distributions, and feature relationships.
- Identification of outliers and their potential impact.
- Insights into feature importance and class separability.
- Clear documentation of data issues and potential feature transformations.

---

## **Detailed Implementation Plan**

### **1. Generate Descriptive Statistics & Visualize Distributions**

**Goals:**
- Summarize each numerical feature (mean, median, std, min, max, quartiles).
- Visualize distributions to assess range, skewness, and modality.

**Steps:**
- Use `df.describe()` for summary statistics.
- For each numerical feature:
  - Plot histograms with KDE overlay (`sns.histplot` or `sns.kdeplot`) to visualize distribution shape.
  - Use `matplotlib` for clear axis labels, titles, and legends.
- Save plots with descriptive filenames (e.g., `'hist_area.png'`).

**Libraries:**
- pandas (`df.describe()`)
- seaborn (`sns.histplot`, `sns.kdeplot`)
- matplotlib (`plt`)

---

### **2. Analyze Class Distribution**

**Goals:**
- Quantify class imbalance.
- Visualize class proportions.

**Steps:**
- Count class occurrences: `df['Class'].value_counts()`
- Plot pie chart or bar plot to visualize class distribution.
- Calculate class imbalance metrics (e.g., ratio of majority to minority class).

**Libraries:**
- pandas
- seaborn/matplotlib

---

### **3. Generate Pairwise Scatter Plots & Correlation Heatmap**

**Goals:**
- Visualize relationships between features.
- Detect multicollinearity and feature redundancies.

**Steps:**
- Use `sns.pairplot()` for a subset of features (e.g., top 5-8 most relevant based on domain knowledge or variance).
- Compute correlation matrix: `corr = df.corr()`
- Plot heatmap: `sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')`
- Highlight highly correlated pairs (|correlation| > 0.8).

**Note:**
- Limit pairwise plots to avoid clutter.
- Use `hue='Class'` if class separation visualization is desired.

**Libraries:**
- seaborn
- pandas

---

### **4. Investigate Feature Distributions Across Classes**

**Goals:**
- Identify features that differentiate classes.
- Detect potential features for classification.

**Steps:**
- For each numerical feature:
  - Plot boxplots or violin plots grouped by class (`sns.boxplot(x='Class', y='Feature', data=df)`).
- Note features with distinct distributions across classes.

**Libraries:**
- seaborn
- pandas

---

### **5. Outlier Detection & Impact Analysis**

**Goals:**
- Identify outliers that may skew model training.
- Decide on potential outlier handling strategies.

**Steps:**
- Use boxplots for each numerical feature.
- Calculate z-scores: `(value - mean) / std`, flag values with |z| > 3.
- Alternatively, use IQR method:
  - Compute Q1, Q3, IQR.
  - Identify points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
- Document features with significant outliers.

**Libraries:**
- scipy.stats (`zscore`)
- pandas

---

### **6. Initial Feature Importance & Statistical Tests**

**Goals:**
- Prioritize features based on relevance.
- Use univariate tests to assess feature relevance.

**Steps:**
- Perform univariate statistical tests:
  - Use ANOVA (`scipy.stats.f_oneway`) for continuous features across classes.
  - Use chi-squared test for categorical features if any.
- Alternatively, use RandomForest feature importance on a preliminary model.

**Libraries:**
- scipy.stats
- sklearn.ensemble (`RandomForestClassifier` for importance)

---

### **7. Explore Feature Transformations & Engineering Opportunities**

**Goals:**
- Identify features that benefit from transformations (e.g., log, sqrt).
- Detect potential new features based on existing ones.

**Steps:**
- Based on distribution plots, consider transformations for skewed features.
- Calculate ratios or interaction terms if meaningful (e.g., `Major_Axis_Length / Minor_Axis_Length`).
- Document potential features for engineering.

---

## **Technical Requirements**

### **DataFrame & Variables:**
- Input: Pandas DataFrame `df` with all features and target `'Class'`.
- Expected: No missing data; handle missing data if present (imputation or removal).

### **Variable Naming & Outputs:**
- Use consistent variable names (`df`, `corr`, etc.).
- Save all plots in a designated directory (`'eda_outputs/'`) with descriptive filenames.
- Summaries should be printed and/or saved as CSV files (`'descriptive_stats.csv'`).

### **Error Handling:**
- Wrap plotting and analysis steps in try-except blocks to catch errors.
- Log errors with descriptive messages.
- Validate data types before analysis (e.g., ensure numerical features are float/int).

### **Additional Notes:**
- Set `random_state=42` for reproducibility in any sampling or model-based steps.
- Maintain clear, commented code for readability.
- Document observations and anomalies during each step.

---

## **Summary of Deliverables:**
- Descriptive statistics CSV.
- Distribution histograms and KDE plots.
- Class distribution pie/bar chart.
- Pairwise scatter plots and correlation heatmap.
- Boxplots/violin plots for feature-class comparison.
- Outlier detection results and plots.
- Feature importance rankings and statistical test results.
- Recommendations for feature transformations and engineering.

---

This detailed plan ensures a comprehensive, systematic approach to exploratory data analysis, setting a solid foundation for subsequent modeling steps.

### 💻 Step 2: Bob (Initial Implementation)
**Role:** Developer

**Initial Code:**
```python
# Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# For reproducibility
RANDOM_STATE = 42

# Print initial shape
print(f"Initial df.shape: {df.shape}")

# 1. Descriptive Statistics & Visualize Distributions
try:
    print("\n1. Descriptive Statistics Summary:")
    desc_stats = df.describe().T
    print(desc_stats)

    # Store descriptive stats in df for potential later use (no saving to CSV as per instructions)
    df_descriptive_stats = desc_stats.copy()

    # List numerical features (exclude target 'Class')
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Class' in num_features:
        num_features.remove('Class')

    # Plot histograms with KDE for each numerical feature
    for feature in num_features:
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[feature], kde=True, stat="density", color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.tight_layout()
            # plt.savefig(f'eda_outputs/hist_{feature.lower()}.png')  # Not saving as per instructions
            plt.close()
        except Exception as e:
            print(f"Error plotting distribution for {feature}: {e}")

except Exception as e:
    print(f"Error in descriptive statistics and distribution plotting: {e}")

# 2. Analyze Class Distribution
try:
    print("\n2. Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)

    majority_class_count = class_counts.max()
    minority_class_count = class_counts.min()
    imbalance_ratio = majority_class_count / minority_class_count
    print(f"Class imbalance ratio (majority/minority): {imbalance_ratio:.2f}")

    plt.figure(figsize=(6,6))
    class_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Class Distribution Pie Chart')
    plt.ylabel('')
    plt.tight_layout()
    # plt.savefig('eda_outputs/class_distribution_pie.png')
    plt.close()

except Exception as e:
    print(f"Error in class distribution analysis: {e}")

# 3. Pairwise Scatter Plots & Correlation Heatmap
try:
    print("\n3. Pairwise Scatter Plots & Correlation Heatmap:")

    # Select subset of features for pairplot based on variance (top 8)
    variances = df[num_features].var().sort_values(ascending=False)
    top_features = variances.head(8).index.tolist()

    # Pairplot with hue='Class'
    try:
        pairplot_fig = sns.pairplot(df, vars=top_features, hue='Class', corner=True, plot_kws={'alpha':0.5, 's':20})
        pairplot_fig.fig.suptitle('Pairwise Scatter Plots of Top Features by Variance', y=1.02)
        plt.close()
    except Exception as e:
        print(f"Error creating pairplot: {e}")

    # Correlation matrix and heatmap
    corr = df[num_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    # plt.savefig('eda_outputs/correlation_heatmap.png')
    plt.close()

    # Highlight highly correlated pairs (|corr| > 0.8)
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr_pairs:
        print("Highly correlated feature pairs (|corr| > 0.8):")
        for f1, f2, val in high_corr_pairs:
            print(f"  {f1} & {f2}: correlation = {val:.2f}")
    else:
        print("No highly correlated feature pairs found (|corr| > 0.8).")

except Exception as e:
    print(f"Error in pairwise scatter plots and correlation heatmap: {e}")

# 4. Investigate Feature Distributions Across Classes
try:
    print("\n4. Feature Distributions Across Classes (Boxplots):")
    for feature in num_features:
        try:
            plt.figure(figsize=(7, 5))
            sns.boxplot(x='Class', y=feature, data=df, palette='Set2')
            plt.title(f'Boxplot of {feature} by Class')
            plt.tight_layout()
            # plt.savefig(f'eda_outputs/boxplot_{feature.lower()}_by_class.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting boxplot for {feature}: {e}")

except Exception as e:
    print(f"Error in feature distribution across classes: {e}")

# 5. Outlier Detection & Impact Analysis
try:
    print("\n5. Outlier Detection:")

    outlier_summary = {}

    for feature in num_features:
        try:
            # Calculate IQR
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers using IQR method
            outliers_iqr = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]

            # Calculate z-scores
            z_scores = zscore(df[feature])
            outliers_z = df[np.abs(z_scores) > 3][feature]

            outlier_summary[feature] = {
                'IQR_outliers_count': outliers_iqr.shape[0],
                'Zscore_outliers_count': outliers_z.shape[0],
                'IQR_lower_bound': lower_bound,
                'IQR_upper_bound': upper_bound
            }

            # Plot boxplot for outlier visualization
            plt.figure(figsize=(7, 4))
            sns.boxplot(x=df[feature], color='lightcoral')
            plt.title(f'Boxplot for Outlier Detection: {feature}')
            plt.tight_layout()
            # plt.savefig(f'eda_outputs/outlier_boxplot_{feature.lower()}.png')
            plt.close()

        except Exception as e:
            print(f"Error detecting outliers for {feature}: {e}")

    # Print summary of outliers
    print("Outlier counts per feature (IQR method and Z-score method):")
    for feat, stats in outlier_summary.items():
        print(f"  {feat}: IQR outliers = {stats['IQR_outliers_count']}, Z-score outliers = {stats['Zscore_outliers_count']}")

except Exception as e:
    print(f"Error in outlier detection: {e}")

# 6. Initial Feature Importance & Statistical Tests
try:
    print("\n6. Initial Feature Importance & Statistical Tests:")

    # Encode target for RF
    le = LabelEncoder()
    df['Class_encoded'] = le.fit_transform(df['Class'])

    # RandomForest for feature importance
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
    rf.fit(df[num_features], df['Class_encoded'])
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': num_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("RandomForest Feature Importances:")
    print(feature_importance_df)

    # ANOVA tests for each numerical feature across classes
    print("\nANOVA test results (F-statistic and p-value):")
    classes = df['Class'].unique()
    anova_results = []
    for feature in num_features:
        try:
            groups = [df[df['Class'] == cls][feature] for cls in classes]
            f_stat, p_val = f_oneway(*groups)
            anova_results.append({'Feature': feature, 'F_statistic': f_stat, 'p_value': p_val})
        except Exception as e:
            print(f"Error performing ANOVA for {feature}: {e}")

    anova_df = pd.DataFrame(anova_results).sort_values(by='p_value')
    print(anova_df)

except Exception as e:
    print(f"Error in feature importance and statistical tests: {e}")

# 7. Explore Feature Transformations & Engineering Opportunities
try:
    print("\n7. Feature Transformations & Engineering:")

    # Identify skewed features (skewness > 1 or < -1)
    skewness = df[num_features].skew()
    skewed_features = skewness[abs(skewness) > 1].index.tolist()
    print(f"Skewed features (|skew| > 1): {skewed_features}")

    # Apply log1p transformation to skewed features (add new columns)
    for feature in skewed_features:
        new_col = f'{feature}_log1p'
        # Avoid transforming features with negative or zero values
        if (df[feature] <= 0).any():
            print(f"Skipping log transform for {feature} due to non-positive values.")
            continue
        df[new_col] = np.log1p(df[feature])
        print(f"Added log1p transformed feature: {new_col}")

    # Create ratio feature: Major_Axis_Length / Minor_Axis_Length (Aspect_Ration is similar but we add for completeness)
    # Check for zero in denominator to avoid division errors
    if (df['Minor_Axis_Length'] == 0).any():
        print("Zero values found in Minor_Axis_Length, cannot create ratio feature safely.")
    else:
        df['Major_to_Minor_Axis_Ratio'] = df['Major_Axis_Length'] / df['Minor_Axis_Length']
        print("Added feature: Major_to_Minor_Axis_Ratio")

    # Create interaction term: Area * Solidity (example of interaction)
    df['Area_Solidity_Interaction'] = df['Area'] * df['Solidity']
    print("Added feature: Area_Solidity_Interaction")

    print(f"\nShape after feature engineering: {df.shape}")

except Exception as e:
    print(f"Error in feature transformations and engineering: {e}")

# Final shape printout
print(f"\nFinal df.shape: {df.shape}")
```

### 🔍 Step 3: Carol (Quality Review)
**Role:** Auditor

**Overall Assessment:**  
The implementation aligns well with the detailed instructions, covering all key aspects of exploratory data analysis. The code is structured, comprehensive, and demonstrates good use of data science libraries. It correctly performs descriptive statistics, visualizations, correlation analysis, outlier detection, feature importance assessment, and feature engineering. The transformations are applied directly to the DataFrame `df`, and the code maintains clarity and robustness.

---

**What's Working Well:**  
- **Adherence to instructions:** The code systematically covers each EDA step, including summaries, visualizations, outlier detection, and feature engineering.  
- **Data transformations:** New features are added directly to `df`, as specified.  
- **Visualization quality:** Use of seaborn and matplotlib for clear, labeled plots.  
- **Statistical analysis:** Use of ANOVA, feature importance, and correlation heatmaps provides solid insights.  
- **Reproducibility:** Consistent `RANDOM_STATE` enhances reproducibility.  
- **Error handling:** Try-except blocks are used to catch and report errors without halting execution.

---

**Issues Identified:**  
1. **Plot Saving & Output Management:**  
   - The instructions specify saving plots with descriptive filenames but the implementation comments out the save commands (`plt.savefig()`). If the goal is to review outputs later, these should be enabled or adjusted accordingly.  
   
2. **Outlier Detection Clarity:**  
   - Outliers are identified via IQR and z-score methods, but no explicit flagging or removal is performed—only counts and plots. Consider adding optional outlier removal or marking for further analysis.

3. **Feature Engineering Checks:**  
   - The code skips log transformations if negative or zero values are present, which is good. However, it does not handle or document how to handle such features (e.g., shifting to positive domain).  
   - The ratio feature creation assumes no zeros in `Minor_Axis_Length`. While checked, it might be better to handle zeros explicitly or avoid creating ratios with zeros.

4. **Univariate Tests & Importance:**  
   - The feature importance via RandomForest is good, but the code does not normalize importance or visualize it. A bar plot could enhance interpretability.  
   - The ANOVA results are printed but not visualized or summarized further.

5. **Potential Missing Data Handling:**  
   - The code assumes no missing data; if present, it should include imputation or removal steps.

6. **Code Comments & Documentation:**  
   - While generally readable, some sections could benefit from more inline comments explaining the rationale, especially for feature engineering choices.

7. **Unused Variables & Data:**  
   - The variable `df_descriptive_stats` is created but not saved or used further. Consider saving if needed.

---

**Improvement Recommendations:**  
- **Uncomment and implement plot saving:** Enable `plt.savefig()` with the correct filenames to preserve outputs for review.  
- **Add feature importance visualization:** e.g., plot importance scores as a bar chart for quick interpretation.  
- **Explicit outlier handling:** Consider flagging or removing outliers based on the counts, or documenting their potential impact.  
- **Handle missing data:** If missing values exist, add imputation or removal steps early in the process.  
- **Enhance feature engineering:**  
   - For skewed features, consider alternative transformations (Box-Cox) if appropriate.  
   - For ratio features, handle zeros or near-zero denominators explicitly.  
- **Summarize statistical test results:** Create a summary table or plot to compare features’ relevance.  
- **Document assumptions and decisions:** For example, why certain features are transformed or engineered.

---

**Priority Items:**  
1. **Plot saving and output management** — Ensure all visualizations are saved systematically.  
2. **Outlier handling** — Decide whether to remove or flag outliers based on counts and domain knowledge.  
3. **Feature importance visualization** — Add a bar plot for importance scores.  
4. **Data completeness check** — Verify and handle missing data if present.  
5. **Comments and documentation** — Improve inline explanations for clarity.

---

**Summary:**  
The implementation demonstrates a solid, methodical approach to EDA, fulfilling the core requirements. Addressing the identified issues and enhancing interpretability and robustness will further improve its effectiveness and readiness for subsequent modeling steps.

### 🔧 Step 4: Bob (Refined Implementation)
**Role:** Developer (Refined)

**Final Refined Code:**
```python
# Necessary imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# For reproducibility
RANDOM_STATE = 42

# Create output directory if it doesn't exist
output_dir = 'eda_outputs'
os.makedirs(output_dir, exist_ok=True)

# Print initial shape
print(f"Initial df.shape: {df.shape}")

# 0. Check for missing data and handle if present
try:
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    if total_missing > 0:
        print(f"\nMissing data detected: {total_missing} missing values across columns.")
        print(missing_summary[missing_summary > 0])
        # Impute numerical features with median
        num_features_missing = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in num_features_missing:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Imputed missing values in {col} with median: {median_val}")
        # Impute categorical features with mode
        cat_features_missing = df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_features_missing:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Imputed missing values in {col} with mode: {mode_val}")
        print("Missing data imputation completed.\n")
    else:
        print("\nNo missing data detected.\n")
except Exception as e:
    print(f"Error checking or imputing missing data: {e}")

# 1. Descriptive Statistics & Visualize Distributions
try:
    print("\n1. Descriptive Statistics Summary:")
    desc_stats = df.describe().T
    print(desc_stats)

    # Save descriptive stats to CSV
    desc_stats.to_csv(os.path.join(output_dir, 'descriptive_stats.csv'))

    # List numerical features (exclude target 'Class')
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Class' in num_features:
        num_features.remove('Class')

    # Plot histograms with KDE for each numerical feature and save
    for feature in num_features:
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[feature], kde=True, stat="density", color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'hist_{feature.lower()}.png'))
            plt.close()
        except Exception as e:
            print(f"Error plotting distribution for {feature}: {e}")

except Exception as e:
    print(f"Error in descriptive statistics and distribution plotting: {e}")

# 2. Analyze Class Distribution
try:
    print("\n2. Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)

    majority_class_count = class_counts.max()
    minority_class_count = class_counts.min()
    imbalance_ratio = majority_class_count / minority_class_count
    print(f"Class imbalance ratio (majority/minority): {imbalance_ratio:.2f}")

    plt.figure(figsize=(6,6))
    class_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Class Distribution Pie Chart')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_pie.png'))
    plt.close()

except Exception as e:
    print(f"Error in class distribution analysis: {e}")

# 3. Pairwise Scatter Plots & Correlation Heatmap
try:
    print("\n3. Pairwise Scatter Plots & Correlation Heatmap:")

    # Select subset of features for pairplot based on variance (top 8)
    variances = df[num_features].var().sort_values(ascending=False)
    top_features = variances.head(8).index.tolist()

    # Pairplot with hue='Class'
    try:
        pairplot_fig = sns.pairplot(df, vars=top_features, hue='Class', corner=True, plot_kws={'alpha':0.5, 's':20})
        pairplot_fig.fig.suptitle('Pairwise Scatter Plots of Top Features by Variance', y=1.02)
        pairplot_fig.savefig(os.path.join(output_dir, 'pairplot_top_features.png'))
        plt.close(pairplot_fig.fig)
    except Exception as e:
        print(f"Error creating pairplot: {e}")

    # Correlation matrix and heatmap
    corr = df[num_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Highlight highly correlated pairs (|corr| > 0.8)
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr_pairs:
        print("Highly correlated feature pairs (|corr| > 0.8):")
        for f1, f2, val in high_corr_pairs:
            print(f"  {f1} & {f2}: correlation = {val:.2f}")
    else:
        print("No highly correlated feature pairs found (|corr| > 0.8).")

except Exception as e:
    print(f"Error in pairwise scatter plots and correlation heatmap: {e}")

# 4. Investigate Feature Distributions Across Classes
try:
    print("\n4. Feature Distributions Across Classes (Boxplots):")
    for feature in num_features:
        try:
            plt.figure(figsize=(7, 5))
            sns.boxplot(x='Class', y=feature, data=df, palette='Set2')
            plt.title(f'Boxplot of {feature} by Class')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'boxplot_{feature.lower()}_by_class.png'))
            plt.close()
        except Exception as e:
            print(f"Error plotting boxplot for {feature}: {e}")

except Exception as e:
    print(f"Error in feature distribution across classes: {e}")

# 5. Outlier Detection & Impact Analysis
try:
    print("\n5. Outlier Detection:")

    outlier_summary = {}

    # Add outlier flags to df for each numerical feature using IQR method
    for feature in num_features:
        try:
            # Calculate IQR
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers using IQR method
            outliers_iqr = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]

            # Calculate z-scores
            z_scores = zscore(df[feature])
            outliers_z = df[np.abs(z_scores) > 3][feature]

            outlier_summary[feature] = {
                'IQR_outliers_count': outliers_iqr.shape[0],
                'Zscore_outliers_count': outliers_z.shape[0],
                'IQR_lower_bound': lower_bound,
                'IQR_upper_bound': upper_bound
            }

            # Add boolean column flagging IQR outliers for this feature
            flag_col = f'{feature}_outlier_flag'
            df[flag_col] = ((df[feature] < lower_bound) | (df[feature] > upper_bound))

            # Plot boxplot for outlier visualization
            plt.figure(figsize=(7, 4))
            sns.boxplot(x=df[feature], color='lightcoral')
            plt.title(f'Boxplot for Outlier Detection: {feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'outlier_boxplot_{feature.lower()}.png'))
            plt.close()

        except Exception as e:
            print(f"Error detecting outliers for {feature}: {e}")

    # Print summary of outliers
    print("Outlier counts per feature (IQR method and Z-score method):")
    for feat, stats in outlier_summary.items():
        print(f"  {feat}: IQR outliers = {stats['IQR_outliers_count']}, Z-score outliers = {stats['Zscore_outliers_count']}")

except Exception as e:
    print(f"Error in outlier detection: {e}")

# 6. Initial Feature Importance & Statistical Tests
try:
    print("\n6. Initial Feature Importance & Statistical Tests:")

    # Encode target for RF
    le = LabelEncoder()
    df['Class_encoded'] = le.fit_transform(df['Class'])

    # RandomForest for feature importance
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
    rf.fit(df[num_features], df['Class_encoded'])
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': num_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)  # ascending for horizontal bar plot

    print("RandomForest Feature Importances:")
    print(feature_importance_df.sort_values(by='Importance', ascending=False))

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('RandomForest Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
    plt.close()

    # ANOVA tests for each numerical feature across classes
    print("\nANOVA test results (F-statistic and p-value):")
    classes = df['Class'].unique()
    anova_results = []
    for feature in num_features:
        try:
            groups = [df[df['Class'] == cls][feature] for cls in classes]
            f_stat, p_val = f_oneway(*groups)
            anova_results.append({'Feature': feature, 'F_statistic': f_stat, 'p_value': p_val})
        except Exception as e:
            print(f"Error performing ANOVA for {feature}: {e}")

    anova_df = pd.DataFrame(anova_results).sort_values(by='p_value')
    print(anova_df)

    # Save ANOVA results to CSV
    anova_df.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)

    # Plot -log10(p-value) for features to visualize significance
    plt.figure(figsize=(10, 6))
    anova_df['neg_log_p'] = -np.log10(anova_df['p_value'].replace(0, 1e-300))  # avoid log(0)
    sns.barplot(x='neg_log_p', y='Feature', data=anova_df, palette='magma')
    plt.xlabel('-log10(p-value)')
    plt.title('ANOVA Test Significance by Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anova_significance.png'))
    plt.close()

except Exception as e:
    print(f"Error in feature importance and statistical tests: {e}")

# 7. Explore Feature Transformations & Engineering Opportunities
try:
    print("\n7. Feature Transformations & Engineering:")

    # Identify skewed features (skewness > 1 or < -1)
    skewness = df[num_features].skew()
    skewed_features = skewness[abs(skewness) > 1].index.tolist()
    print(f"Skewed features (|skew| > 1): {skewed_features}")

    # Apply log1p transformation to skewed features (add new columns)
    for feature in skewed_features:
        new_col = f'{feature}_log1p'
        # Handle non-positive values by shifting
        min_val = df[feature].min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6  # small epsilon to ensure positivity
            print(f"Shifting {feature} by {shift:.6f} before log transform due to non-positive values.")
            df[new_col] = np.log1p(df[feature] + shift)
        else:
            df[new_col] = np.log1p(df[feature])
        print(f"Added log1p transformed feature: {new_col}")

    # Create ratio feature: Major_Axis_Length / Minor_Axis_Length (handle zeros safely)
    denom = df['Minor_Axis_Length'].replace(0, np.nan)
    if denom.isnull().any():
        # Replace zeros with small epsilon to avoid division by zero
        denom = df['Minor_Axis_Length'].replace(0, 1e-6)
        print("Replaced zeros in Minor_Axis_Length with small epsilon for ratio feature creation.")
    df['Major_to_Minor_Axis_Ratio'] = df['Major_Axis_Length'] / denom
    print("Added feature: Major_to_Minor_Axis_Ratio")

    # Create interaction term: Area * Solidity (example of interaction)
    df['Area_Solidity_Interaction'] = df['Area'] * df['Solidity']
    print("Added feature: Area_Solidity_Interaction")

    print(f"\nShape after feature engineering: {df.shape}")

except Exception as e:
    print(f"Error in feature transformations and engineering: {e}")

# Final shape printout
print(f"\nFinal df.shape: {df.shape}")
```

### 🖥 Execution Results
**Status:** ✅ Success

```
Initial df.shape: (2500, 13)

No missing data detected.


1. Descriptive Statistics Summary:
                    count          mean           std         min  \
Area               2500.0  80658.220800  13664.510228  47939.0000   
Perimeter          2500.0   1130.279015    109.256418    868.4850   
Major_Axis_Length  2500.0    456.601840     56.235704    320.8446   
Minor_Axis_Length  2500.0    225.794921     23.297245    152.1718   
Convex_Area        2500.0  81508.084400  13764.092788  48366.0000   
Equiv_Diameter     2500.0    319.334230     26.891920    247.0584   
Eccentricity       2500.0      0.860879      0.045167      0.4921   
Solidity           2500.0      0.989492      0.003494      0.9186   
Extent             2500.0      0.693205      0.060914      0.4680   
Roundness          2500.0      0.791533      0.055924      0.5546   
Aspect_Ration      2500.0      2.041702      0.315997      1.1487   
Compactness        2500.0      0.704121      0.053067      0.5608   

                            25%          50%           75%          max  
Area               70765.000000  79076.00000  89757.500000  136574.0000  
Perimeter           1048.829750   1123.67200   1203.340500    1559.4500  
Major_Axis_Length    414.957850    449.49660    492.737650     661.9113  
Minor_Axis_Length    211.245925    224.70310    240.672875     305.8180  
Convex_Area        71512.000000  79872.00000  90797.750000  138384.0000  
Equiv_Diameter       300.167975    317.30535    338.057375     417.0029  
Eccentricity           0.831700      0.86370      0.897025       0.9481  
Solidity               0.988300      0.99030      0.991500       0.9944  
Extent                 0.658900      0.71305      0.740225       0.8296  
Roundness              0.751900      0.79775      0.834325       0.9396  
Aspect_Ration          1.801050      1.98420      2.262075       3.1444  
Compactness            0.663475      0.70770      0.743500       0.9049  

2. Class Distribution:
Class
Çerçevelik       1300
Ürgüp Sivrisi    1200
Name: count, dtype: int64
Class imbalance ratio (majority/minority): 1.08

3. Pairwise Scatter Plots & Correlation Heatmap:
Highly correlated feature pairs (|corr| > 0.8):
  Area & Perimeter: correlation = 0.93
  Area & Convex_Area: correlation = 1.00
  Area & Equiv_Diameter: correlation = 1.00
  Perimeter & Major_Axis_Length: correlation = 0.95
  Perimeter & Convex_Area: correlation = 0.93
  Perimeter & Equiv_Diameter: correlation = 0.93
  Convex_Area & Equiv_Diameter: correlation = 1.00
  Eccentricity & Roundness: correlation = -0.89
  Eccentricity & Aspect_Ration: correlation = 0.95
  Eccentricity & Compactness: correlation = -0.98
  Roundness & Aspect_Ration: correlation = -0.94
  Roundness & Compactness: correlation = 0.93
  Aspect_Ration & Compactness: correlation = -0.99

4. Feature Distributions Across Classes (Boxplots):

5. Outlier Detection:
Outlier counts per feature (IQR method and Z-score method):
  Area: IQR outliers = 18, Z-score outliers = 13
  Perimeter: IQR outliers = 16, Z-score outliers = 8
  Major_Axis_Length: IQR outliers = 21, Z-score outliers = 8
  Minor_Axis_Length: IQR outliers = 30, Z-score outliers = 9
  Convex_Area: IQR outliers = 17, Z-score outliers = 13
  Equiv_Diameter: IQR outliers = 13, Z-score outliers = 9
  Eccentricity: IQR outliers = 18, Z-score outliers = 14
  Solidity: IQR outliers = 103, Z-score outliers = 29
  Extent: IQR outliers = 46, Z-score outliers = 13
  Roundness: IQR outliers = 5, Z-score outliers = 4
  Aspect_Ration: IQR outliers = 11, Z-score outliers = 8
  Compactness: IQR outliers = 2, Z-score outliers = 2

6. Initial Feature Importance & Statistical Tests:
RandomForest Feature Importances:
              Feature  Importance
10      Aspect_Ration    0.206223
6        Eccentricity    0.155467
11        Compactness    0.154188
9           Roundness    0.145668
2   Major_Axis_Length    0.073507
7            Solidity    0.054776
3   Minor_Axis_Length    0.047614
8              Extent    0.037734
1           Perimeter    0.036088
5      Equiv_Diameter    0.030442
4         Convex_Area    0.029469
0                Area    0.028823

ANOVA test results (F-statistic and p-value):
              Feature  F_statistic        p_value
6        Eccentricity  2390.900770   0.000000e+00
9           Roundness  2029.403166   0.000000e+00
10      Aspect_Ration  2716.923219   0.000000e+00
11        Compactness  2795.022444   0.000000e+00
2   Major_Axis_Length  1149.965997  1.108998e-207
3   Minor_Axis_Length   479.679978   2.079533e-97
1           Perimeter   443.634119   8.682216e-91
8              Extent   147.434883   5.263276e-33
0                Area    74.593266   1.013240e-17
4         Convex_Area    72.577417   2.732434e-17
5      Equiv_Diameter    65.883942   7.425085e-16
7            Solidity    38.166489   7.560815e-10

7. Feature Transformations & Engineering:
Skewed features (|skew| > 1): ['Solidity', 'Extent']
Added log1p transformed feature: Solidity_log1p
Added log1p transformed feature: Extent_log1p
Added feature: Major_to_Minor_Axis_Ratio
Added feature: Area_Solidity_Interaction

Shape after feature engineering: (2500, 30)

Final df.shape: (2500, 30)
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
