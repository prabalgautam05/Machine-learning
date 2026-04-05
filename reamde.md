# Insurance Charges Analysis — Notebook README

This notebook performs **exploratory data analysis (EDA), data cleaning, preprocessing, and feature engineering** on a medical insurance charges dataset. The goal is to understand the factors that influence insurance charges and prepare a clean, model-ready dataset.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Library Imports](#2-library-imports)
3. [Data Loading & Initial Exploration](#3-data-loading--initial-exploration)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Data Cleaning & Preprocessing](#5-data-cleaning--preprocessing)
6. [Feature Engineering & Selection](#6-feature-engineering--selection)

---

## 1. Dataset Overview

The dataset (`insurance.csv`) contains **1,338 rows and 7 columns** representing individual insurance policyholders.

| Column     | Type    | Description                                         |
|------------|---------|-----------------------------------------------------|
| `age`      | int     | Age of the primary beneficiary                      |
| `sex`      | string  | Gender of the beneficiary (`male` / `female`)       |
| `bmi`      | float   | Body Mass Index (BMI)                               |
| `children` | int     | Number of children/dependents covered               |
| `smoker`   | string  | Whether the beneficiary is a smoker (`yes` / `no`)  |
| `region`   | string  | US region of residence (northeast/northwest/southeast/southwest) |
| `charges`  | float   | Individual medical insurance charges billed         |

---

## 2. Library Imports

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
```

Standard Python data science libraries are imported:
- **NumPy** — numerical operations
- **Pandas** — data loading and manipulation
- **Seaborn** — statistical data visualization
- **Matplotlib** — plotting
- **warnings** — suppresses non-critical warnings during execution

---

## 3. Data Loading & Initial Exploration

### Load Data
```python
df = pd.read_csv('insurance.csv')
```
Loads the CSV file into a Pandas DataFrame.

### Shape
```python
df.shape  # (1338, 7)
```
Confirms 1,338 rows and 7 columns.

### Info / Data Types
```python
df.info
```
Displays column names with their bound method reference (note: `df.info` without parentheses shows the method object rather than calling it — `df.info()` would call it properly).

### Descriptive Statistics
```python
df.describe()
```
Outputs summary statistics for numerical columns:

| Stat  | age    | bmi    | children | charges       |
|-------|--------|--------|----------|---------------|
| count | 1338   | 1338   | 1338     | 1338          |
| mean  | 39.21  | 30.66  | 1.09     | 13,270.42     |
| std   | 14.05  | 6.10   | 1.21     | 12,110.01     |
| min   | 18     | 15.96  | 0        | 1,121.87      |
| 25%   | 27     | 26.30  | 0        | 4,740.29      |
| 50%   | 39     | 30.40  | 1        | 9,382.03      |
| 75%   | 51     | 34.69  | 2        | 16,639.91     |
| max   | 64     | 53.13  | 5        | 63,770.43     |

### Missing Values
```python
df.isnull().sum()
```
Confirms there are **no missing values** in any column.

### Columns
```python
df.columns
# Index(['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'])
```

---

## 4. Exploratory Data Analysis (EDA)

### 4a. Histograms of Numeric Features

```python
numeric_columns = ['age', 'bmi', 'children', 'charges']
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True, bins=20)
```

Plots KDE-overlaid histograms for each numeric column, showing distribution shape:
- **Age**: Roughly uniform distribution between 18 and 64
- **BMI**: Approximately normal, centred around 30
- **Children**: Heavily right-skewed; most people have 0 children
- **Charges**: Strongly right-skewed; most charges are low, but a long tail extends to ~$64K

### 4b. Count Plots for Categorical Features

```python
sns.countplot(x=df['children'])
sns.countplot(x=df['sex'])
sns.countplot(x=df['smoker'])
```

Shows the frequency distribution of:
- **Children**: 0 is by far the most common count
- **Sex**: Roughly balanced between male (~676) and female (~662)
- **Smoker**: Majority are non-smokers (~1,064 no vs. ~274 yes)

### 4c. Box Plots for Numeric Features

```python
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col])
```

Box plots highlight the spread and outliers:
- **Age**: No outliers; spans 18–64
- **BMI**: A few high-end outliers above ~50
- **Children**: Outliers at 4 and 5
- **Charges**: Significant right-skewed outliers (very high medical costs)

### 4d. Correlation Heatmap

```python
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

Shows Pearson correlation between numeric features and `charges`:
- **Age** has a moderate positive correlation with charges (~0.30)
- **BMI** has a mild positive correlation (~0.20)
- **Children** has a weak positive correlation (~0.07)

---

## 5. Data Cleaning & Preprocessing

### 5a. Create a Working Copy

```python
df_cleaned = df.copy()
```

All transformations are performed on `df_cleaned` to preserve the original `df`.

### 5b. Encode `sex` Column

```python
df_cleaned['sex'] = df_cleaned['sex'].map({"male": 0, "female": 1})
```

Maps the `sex` column to binary integers: `male → 0`, `female → 1`.

### 5c. Remove Duplicates

```python
df_cleaned.drop_duplicates(inplace=True)
```

One duplicate row is found and removed, leaving **1,337 rows**.

### 5d. Verify Missing Values

```python
df_cleaned.isnull().sum()
```

Confirms no nulls remain after deduplication.

### 5e. Value Counts for `sex`

```python
df_cleaned['sex'].value_counts()
# male      675
# female    662
```

Confirms the gender split after encoding.

### 5f. Check Column Data Types

```python
df_cleaned.dtypes
```

Shows that `sex`, `smoker`, and `region` are still object/string types at this point.

### 5g. Encode `smoker` Column

```python
df_cleaned['smoker'] = df_cleaned['smoker'].map({"yes": 1, "no": 0})
```

Maps the `smoker` column to binary integers: `yes → 1`, `no → 0`.

Verified with:
```python
df_cleaned['smoker'].value_counts()
# no     1064
# yes     274
```

### 5h. One-Hot Encode `region` Column

```python
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
```

Converts the 4-level `region` column into 3 binary dummy variables, dropping the first category (`northeast`) to avoid multicollinearity:
- `region_northwest`
- `region_southeast`
- `region_southwest`

---

## 6. Feature Engineering & Selection

### 6a. Visualise BMI Distribution

```python
sns.histplot(df['bmi'])
```

Shows that BMI is roughly bell-shaped, suggesting natural cut-points for categorisation.

### 6b. Create BMI Categories

```python
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)
```

Creates an ordinal `bmi_category` feature based on standard WHO thresholds:
- **Underweight**: BMI < 18.5
- **Normal**: 18.5 ≤ BMI < 25
- **Overweight**: 25 ≤ BMI < 30
- **Obese**: BMI ≥ 30

### 6c. One-Hot Encode `bmi_category`

```python
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
```

Produces three dummy variables (dropping `Underweight` as reference):
- `bmi_category_Normal`
- `bmi_category_Overweight`
- `bmi_category_Obese`

### 6d. Convert All Columns to Integer

```python
df_cleaned = df_cleaned.astype(int)
```

Truncates float values (e.g., `charges`, `bmi`) to integers for uniform dtype handling. **Note:** This step loses decimal precision in `charges` and `bmi`.

### 6e. Feature Scaling (StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
```

Standardises the three continuous features to zero mean and unit variance — a common requirement for distance-based and gradient-based ML algorithms.

### 6f. Rename Columns for Clarity

```python
df_cleaned.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)
```

Renames binary indicator columns to semantically descriptive names.

### 6g. Pearson Correlation Feature Ranking

```python
from scipy.stats import pearsonr

selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}
```

Computes Pearson correlation of each feature against `charges`. Top results:

| Feature                  | Pearson Correlation |
|--------------------------|---------------------|
| `is_smoker`              | 0.787               |
| `age`                    | 0.299               |
| `bmi_category_Obese`     | 0.197               |
| `bmi`                    | 0.196               |
| `region_southeast`       | 0.074               |
| `children`               | 0.068               |

**Smoking status is by far the strongest predictor of charges.**

### 6h. Chi-Squared Feature Selection

```python
from scipy.stats import chi2_contingency

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
```

Uses chi-squared independence tests (α = 0.05) between each binary feature and a quartile-binned version of `charges`. Results:

| Feature                   | χ² Statistic | p-value | Decision              |
|---------------------------|-------------|---------|-----------------------|
| `is_smoker`               | 854.02      | 0.000   | **Keep Feature**      |
| `region_southeast`        | 15.21       | 0.0016  | **Keep Feature**      |
| `is_female`               | 9.53        | 0.023   | **Keep Feature**      |
| `bmi_category_Obese`      | 7.62        | 0.055   | Drop Feature          |
| `region_southwest`        | 5.53        | 0.137   | Drop Feature          |
| `bmi_category_Overweight` | 4.63        | 0.201   | Drop Feature          |
| `bmi_category_Normal`     | 4.13        | 0.247   | Drop Feature          |
| `region_northwest`        | 1.23        | 0.747   | Drop Feature          |

### 6i. Construct Final Feature Set

```python
final_df = df_cleaned[[
    'age', 'is_female', 'bmi', 'children', 'is_smoker',
    'charges', 'region_southeast', 'bmi_category_Obese'
]]
```

The final dataset retains 7 predictors + 1 target (`charges`), combining:
- Features significant by both Pearson and/or Chi-squared tests
- Scaled continuous features: `age`, `bmi`, `children`
- Binary indicators: `is_female`, `is_smoker`, `region_southeast`, `bmi_category_Obese`

---

## Summary

| Stage                  | Key Action                                                      |
|------------------------|-----------------------------------------------------------------|
| **Loading**            | Read 1,338-row CSV with 7 features                              |
| **EDA**                | Histograms, countplots, boxplots, heatmap                       |
| **Cleaning**           | Encoded sex & smoker, removed 1 duplicate, verified no nulls    |
| **Encoding**           | One-hot encoded region (drop first) and bmi_category            |
| **Scaling**            | StandardScaler applied to age, bmi, children                    |
| **Feature Selection**  | Pearson correlation + chi-squared → 7 final predictors retained |

The resulting `final_df` is ready for downstream regression modelling of insurance charges.