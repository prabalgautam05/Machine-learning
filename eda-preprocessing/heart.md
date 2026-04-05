# Heart Disease Prediction — Notebook README

This notebook performs **exploratory data analysis (EDA)** and **data preprocessing** on a heart disease dataset (`heart.csv`), preparing it for machine learning classification.

---

## Table of Contents

1. [Setup & Imports](#1-setup--imports)
2. [Data Loading](#2-data-loading)
3. [Initial Data Inspection](#3-initial-data-inspection)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Data Cleaning](#5-data-cleaning)
6. [Feature Engineering & Preprocessing](#6-feature-engineering--preprocessing)

---

## Dataset

The dataset (`heart.csv`) contains **918 rows × 12 columns** describing patient health metrics used to predict heart disease.

| Column | Type | Description |
|---|---|---|
| `Age` | int | Patient age in years |
| `Sex` | str | `M` = Male, `F` = Female |
| `ChestPainType` | str | `ATA`, `NAP`, `ASY`, `TA` |
| `RestingBP` | int | Resting blood pressure (mm Hg) |
| `Cholesterol` | int | Serum cholesterol (mg/dl) |
| `FastingBS` | int | Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False) |
| `RestingECG` | str | Resting ECG results: `Normal`, `ST`, `LVH` |
| `MaxHR` | int | Maximum heart rate achieved |
| `ExerciseAngina` | str | Exercise-induced angina (`Y`/`N`) |
| `Oldpeak` | float | ST depression induced by exercise |
| `ST_Slope` | str | Slope of peak exercise ST segment (`Up`, `Flat`, `Down`) |
| `HeartDisease` | int | Target: `1` = Heart Disease, `0` = Normal |

---

## 1. Setup & Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

Standard data science libraries are imported. Warnings are suppressed for clean output.

---

## 2. Data Loading

```python
df = pd.read_csv('heart.csv')
```

Loads the dataset from a CSV file into a pandas DataFrame.

---

## 3. Initial Data Inspection

### Preview

```python
df.head()
```

Displays the first 5 rows to understand the structure and sample values.

### Column Names

```python
df.columns
```

Lists all 12 column names: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, `HeartDisease`.

### Shape

```python
df.shape  # Output: (918, 12)
```

Confirms 918 records and 12 features.

### Data Types & Null Counts

```python
df.info()
```

Shows column data types and confirms **zero null values** across all columns.

### Statistical Summary

```python
df.describe()
```

Provides descriptive statistics for numeric columns:
- `Age` ranges from 28 to 77 (mean ≈ 53.5)
- `RestingBP` has a min of 0 (invalid — cleaned later)
- `Cholesterol` has a min of 0 (invalid — cleaned later)
- `HeartDisease` mean ≈ 0.55, indicating a slightly imbalanced but workable target

### Duplicates

```python
df.duplicated().sum()  # Output: 0
```

No duplicate rows in the dataset.

### Null Values

```python
df.isnull().sum()
```

All columns return 0 — no missing values.

---

## 4. Exploratory Data Analysis (EDA)

### Target Distribution

```python
df['HeartDisease'].value_counts().plot(kind="bar")
```

Bar chart showing the count of patients with (`1`) and without (`0`) heart disease. The dataset is roughly balanced (~508 positive, ~410 negative).

### Histograms of Continuous Features

```python
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)

plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)
plt.tight_layout()
```

A 2×2 grid of histograms with KDE curves for `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`. Reveals skewness and presence of zero values in `Cholesterol` and `RestingBP`.

### Cholesterol Value Counts

```python
df['Cholesterol'].value_counts()
```

Shows that **172 records have Cholesterol = 0**, which is physiologically impossible — these are treated as missing values to be imputed.

### Categorical Feature vs. Target

```python
sns.countplot(x=df['Sex'], hue=df['HeartDisease'])
sns.countplot(x=df['ChestPainType'], hue=df['HeartDisease'])
sns.countplot(x=df['FastingBS'], hue=df['HeartDisease'])
```

Grouped bar charts showing the distribution of heart disease across:
- **Sex**: Males have higher counts of heart disease.
- **ChestPainType**: `ASY` (asymptomatic) is strongly associated with heart disease.
- **FastingBS**: Patients with fasting blood sugar > 120 mg/dl show higher heart disease rates.

### Cholesterol Box Plot

```python
sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
```

Visualises cholesterol spread by target class after zero-imputation. Patients without heart disease tend to have slightly higher cholesterol values.

### Age Violin Plot

```python
sns.violinplot(x='HeartDisease', y='Age', data=df)
```

Shows that heart disease patients are generally older, with a wider distribution in the positive class.

### Correlation Heatmap

```python
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

Heatmap of Pearson correlations between numeric features. Notable findings:
- `Oldpeak` and `FastingBS` positively correlate with `HeartDisease`
- `MaxHR` negatively correlates with `HeartDisease`

---

## 5. Data Cleaning

### Imputing Zero Cholesterol

```python
ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
```

Zeros in `Cholesterol` are replaced with the mean of non-zero values (≈ **244.64**). This step is run twice in the notebook to ensure full imputation, then values are rounded to 2 decimal places.

### Imputing Zero RestingBP

```python
resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean)
df['RestingBP'] = df['RestingBP'].round(2)
```

Same approach applied to `RestingBP` — zeros are replaced with the column mean of valid readings.

After cleaning, the histograms are re-plotted to confirm improvements in the distributions of `RestingBP` and `Cholesterol`.

### Summary Report (sheryanalysis)

```python
import sheryanalysis as sh
sh.analyze(df)
```

Generates a structured report confirming:
- Shape: `(918, 12)`
- No null values
- Categorical columns: `Sex`, `ChestPainType`, `FastingBS`, `RestingECG`, `ExerciseAngina`, `ST_Slope`, `HeartDisease`
- Numerical columns: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`

---

## 6. Feature Engineering & Preprocessing

### One-Hot Encoding

```python
df_encode = pd.get_dummies(df, drop_first=True)
```

Converts all categorical string columns to binary dummy variables using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity. This expands the DataFrame from 12 to **16 columns**.

New columns added:
- `Sex_M`
- `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`
- `RestingECG_Normal`, `RestingECG_ST`
- `ExerciseAngina_Y`
- `ST_Slope_Flat`, `ST_Slope_Up`

### Convert Boolean to Integer

```python
df_encode = df_encode.astype(int)
```

Converts the boolean dummy columns (`True`/`False`) to integer (`1`/`0`) for compatibility with scikit-learn models.

### Standard Scaling

```python
from sklearn.preprocessing import StandardScaler

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
```

Scales the 5 continuous numeric features to have **mean = 0 and standard deviation = 1**. Categorical and binary columns are left unchanged. The resulting `df_encode` DataFrame is ready for model training.

---

## Key Takeaways

- The dataset is clean with no nulls or duplicates, but contains **invalid zero values** in `Cholesterol` and `RestingBP` that require mean imputation.
- **ASY chest pain**, **male sex**, **high fasting blood sugar**, **low MaxHR**, and **high Oldpeak** are notable risk indicators for heart disease.
- After preprocessing, the final dataset has 918 rows × 16 feature columns, fully numeric and scaled — ready for downstream classification models.