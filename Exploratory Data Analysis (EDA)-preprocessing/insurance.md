# Streamlit EDA & Preprocessing App

A comprehensive exploratory data analysis and preprocessing dashboard for insurance and heart disease datasets.

## Features

### 📊 Home Dashboard
- Quick overview of both datasets
- Easy navigation to individual dataset analysis

### 💰 Insurance Dataset Analysis
- **Overview Tab**: Dataset statistics, shape, data types
- **EDA Tab**: Distribution of categorical features (sex, smoker, region, children)
- **Data Quality Tab**: Missing values, duplicates, statistical summary
- **Distributions Tab**: Histograms for age, BMI, charges, and log-transformed charges
- **Correlations Tab**: Heatmap and correlation analysis with target variable

### ❤️ Heart Disease Dataset Analysis
- **Overview Tab**: Dataset statistics and basic info
- **EDA Tab**: Target distribution, categorical feature analysis
- **Data Quality Tab**: Missing values and duplicates analysis
- **Distributions Tab**: Distribution visualization for all numerical features
- **Correlations Tab**: Correlation matrix and target variable correlations

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd "c:\Users\praba\Documents\GitHub\Machine-learning\Exploratory Data Analysis (EDA)-preprocessing"
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── insurance.csv             # Insurance dataset
├── heart.csv                 # Heart disease dataset
├── insurance.ipynb           # Insurance analysis notebook
├── heart.ipynb               # Heart disease analysis notebook
├── insurance.md              # Insurance dataset documentation
├── heart.md                  # Heart disease dataset documentation
└── README.md                 # This file
```

## Usage Guide

### Navigation
- Use the sidebar to switch between datasets
- Click the "Explore" buttons on the home page or use the sidebar radio buttons

### Dataset Overview
- View basic statistics and data shape
- Check data types and missing values
- Browse the first 10 rows of data

### Exploratory Data Analysis
- Visualize distributions of categorical and numerical features
- Analyze target variable distribution
- Explore relationships between features

### Data Quality Checks
- Identify missing values
- Detect duplicate records
- Review statistical summaries

### Correlation Analysis
- Heatmaps show feature relationships
- Identify strong correlations with target variable
- Understand feature dependencies

## Datasets Description

### Insurance Dataset (1,338 records, 7 features)
- **age**: Age of the primary beneficiary
- **sex**: Gender (male/female)
- **bmi**: Body Mass Index
- **children**: Number of dependents
- **smoker**: Smoking status (yes/no)
- **region**: US region (northeast/northwest/southeast/southwest)
- **charges**: Annual insurance charges (TARGET)

### Heart Disease Dataset (918 records, 12 features)
- **Age**: Patient age in years
- **Sex**: Male or Female
- **ChestPainType**: ASY, ATA, NAP, TA
- **RestingBP**: Resting blood pressure
- **Cholesterol**: Serum cholesterol levels
- **FastingBS**: Fasting blood sugar
- **RestingECG**: Resting ECG results
- **MaxHR**: Maximum heart rate
- **ExerciseAngina**: Exercise-induced angina
- **Oldpeak**: ST depression
- **ST_Slope**: ST segment slope
- **HeartDisease**: Presence of heart disease (TARGET)

## Technologies Used

- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **SciPy**: Statistical analysis

## Tips

1. **Responsive Design**: The dashboard is optimized for different screen sizes
2. **Caching**: Data is cached to improve performance
3. **Export Data**: Use Streamlit's built-in download buttons to export visualizations
4. **Interactivity**: Hover over charts to see detailed values

## Future Enhancements

- Add data preprocessing and transformation workflows
- Implement predictive modeling
- Add feature engineering demonstrations
- Include advanced statistical tests
- Add data export functionality

## Notes

- All visualizations are created with matplotlib and seaborn
- Categorical variables are encoded numerically for correlation analysis
- Missing values are automatically detected and reported
- The app runs in wide mode for better visualization space

## Author

Created for exploratory data analysis and data science education.

---

For questions or improvements, feel free to modify and extend the app!
