import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EDA & Preprocessing App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom theme
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data


@st.cache_data
def load_insurance_data():
    df = pd.read_csv('insurance.csv')
    return df


@st.cache_data
def load_heart_data():
    df = pd.read_csv('heart.csv')
    return df


# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select Dataset & Section",
    ["Home", "Insurance Analysis", "Heart Disease Analysis"]
)

# ==================== HOME PAGE ====================
if page == "Home":
    st.title("📊 Exploratory Data Analysis & Preprocessing")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Insurance Dataset")
        st.markdown("""
        **Medical Insurance Charges Analysis**
        - Records: 1,338
        - Features: 7
        - Target: Charges
        
        Analyze factors influencing insurance charges including age, BMI, smoking status, and more.
        """)
        if st.button("Explore Insurance 🔍", key="btn_insurance"):
            st.switch_page("pages/insurance_analysis.py")

    with col2:
        st.subheader("❤️ Heart Disease Dataset")
        st.markdown("""
        **Heart Disease Prediction**
        - Records: 918
        - Features: 12
        - Target: Heart Disease (Binary)
        
        Predict heart disease presence using clinical health metrics.
        """)
        if st.button("Explore Heart Disease 🔍", key="btn_heart"):
            st.switch_page("pages/heart_analysis.py")

# ==================== INSURANCE ANALYSIS PAGE ====================
elif page == "Insurance Analysis":
    st.title("💰 Insurance Charges Analysis")
    st.markdown("---")

    df = load_insurance_data()

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "📊 EDA", "🧹 Data Quality", "📈 Distributions", "🔗 Correlations"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Types", len(df.dtypes.unique()))

        st.subheader("Dataset Head")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Data Types & Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df.dtypes)
        with col2:
            st.write(df.describe().round(2))

    # TAB 2: EDA
    with tab2:
        st.subheader("Categorical Distribution")
        categorical_cols = df.select_dtypes(include='object').columns

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sex Distribution**")
            fig, ax = plt.subplots()
            df['sex'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
            plt.title("Sex Distribution")
            plt.xlabel("Sex")
            plt.ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.markdown("**Smoker Distribution**")
            fig, ax = plt.subplots()
            df['smoker'].value_counts().plot(kind='bar', ax=ax, color='coral')
            plt.title("Smoker Distribution")
            plt.xlabel("Smoker")
            plt.ylabel("Count")
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Region Distribution**")
            fig, ax = plt.subplots()
            df['region'].value_counts().plot(
                kind='bar', ax=ax, color='lightgreen')
            plt.title("Region Distribution")
            plt.xlabel("Region")
            plt.ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.markdown("**Children Distribution**")
            fig, ax = plt.subplots()
            df['children'].value_counts().sort_index().plot(
                kind='bar', ax=ax, color='plum')
            plt.title("Children Distribution")
            plt.xlabel("Number of Children")
            plt.ylabel("Count")
            st.pyplot(fig)

    # TAB 3: DATA QUALITY
    with tab3:
        st.subheader("Missing Values Analysis")
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(missing[missing['Missing Count'] > 0] if missing['Missing Count'].sum(
        ) > 0 else missing, use_container_width=True)

        st.subheader("Duplicate Rows")
        st.write(f"Total Duplicate Rows: {df.duplicated().sum()}")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # TAB 4: DISTRIBUTIONS
    with tab4:
        st.subheader("Numerical Distributions")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Age Distribution**")
            fig, ax = plt.subplots()
            df['age'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
            plt.title("Age Distribution")
            plt.xlabel("Age")
            plt.ylabel("Frequency")
            st.pyplot(fig)

        with col2:
            st.markdown("**BMI Distribution**")
            fig, ax = plt.subplots()
            df['bmi'].hist(bins=30, ax=ax, color='lightcoral',
                           edgecolor='black')
            plt.title("BMI Distribution")
            plt.xlabel("BMI")
            plt.ylabel("Frequency")
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Charges Distribution**")
            fig, ax = plt.subplots()
            df['charges'].hist(
                bins=30, ax=ax, color='lightgreen', edgecolor='black')
            plt.title("Insurance Charges Distribution")
            plt.xlabel("Charges ($)")
            plt.ylabel("Frequency")
            st.pyplot(fig)

        with col2:
            st.markdown("**Log Charges Distribution**")
            fig, ax = plt.subplots()
            np.log(df['charges']).hist(bins=30, ax=ax,
                                       color='plum', edgecolor='black')
            plt.title("Log Insurance Charges Distribution")
            plt.xlabel("Log Charges")
            plt.ylabel("Frequency")
            st.pyplot(fig)

    # TAB 5: CORRELATIONS
    with tab5:
        st.subheader("Correlation Analysis")

        # Encode categorical variables for correlation
        df_encoded = df.copy()
        df_encoded['sex'] = df_encoded['sex'].map({'male': 1, 'female': 0})
        df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
        df_encoded['region'] = pd.factorize(df_encoded['region'])[0]

        correlation = df_encoded.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title("Correlation Matrix - Insurance Dataset")
        st.pyplot(fig)

        st.subheader("Correlation with Charges")
        charges_corr = correlation['charges'].sort_values(ascending=False)
        st.dataframe(charges_corr, use_container_width=True)

# ==================== HEART DISEASE ANALYSIS PAGE ====================
elif page == "Heart Disease Analysis":
    st.title("❤️ Heart Disease Prediction Analysis")
    st.markdown("---")

    df = load_heart_data()

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "📊 EDA", "🧹 Data Quality", "📈 Distributions", "🔗 Correlations"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Types", len(df.dtypes.unique()))

        st.subheader("Dataset Head")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Data Types & Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df.dtypes)
        with col2:
            st.write(df.describe().round(2))

    # TAB 2: EDA
    with tab2:
        st.subheader("Target Distribution")
        target_counts = df[df.columns[-1]].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Heart Disease Distribution**")
            fig, ax = plt.subplots()
            target_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
            plt.title("Heart Disease Distribution")
            plt.xlabel("Heart Disease (0=No, 1=Yes)")
            plt.ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.markdown("**Heart Disease Proportion**")
            fig, ax = plt.subplots()
            target_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=[
                               '#90EE90', '#FF6B6B'])
            plt.ylabel("")
            plt.title("Heart Disease Proportion")
            st.pyplot(fig)

        st.subheader("Categorical Features Analysis")
        categorical_cols = df.select_dtypes(include='object').columns

        col1, col2 = st.columns(2)
        with col1:
            if 'Sex' in df.columns:
                st.markdown("**Sex Distribution**")
                fig, ax = plt.subplots()
                df['Sex'].value_counts().plot(
                    kind='bar', ax=ax, color='skyblue')
                plt.title("Sex Distribution")
                plt.xlabel("Sex")
                plt.ylabel("Count")
                st.pyplot(fig)

        with col2:
            if 'ChestPainType' in df.columns:
                st.markdown("**Chest Pain Type Distribution**")
                fig, ax = plt.subplots()
                df['ChestPainType'].value_counts().plot(
                    kind='bar', ax=ax, color='coral')
                plt.title("Chest Pain Type Distribution")
                plt.xlabel("Chest Pain Type")
                plt.ylabel("Count")
                st.pyplot(fig)

    # TAB 3: DATA QUALITY
    with tab3:
        st.subheader("Missing Values Analysis")
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(missing[missing['Missing Count'] > 0] if missing['Missing Count'].sum(
        ) > 0 else missing, use_container_width=True)

        st.subheader("Duplicate Rows")
        st.write(f"Total Duplicate Rows: {df.duplicated().sum()}")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # TAB 4: DISTRIBUTIONS
    with tab4:
        st.subheader("Numerical Distributions")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for i in range(0, len(numeric_cols), 2):
            col1, col2 = st.columns(2)

            with col1:
                if i < len(numeric_cols):
                    col_name = numeric_cols[i]
                    st.markdown(f"**{col_name} Distribution**")
                    fig, ax = plt.subplots()
                    df[col_name].hist(
                        bins=30, ax=ax, color='skyblue', edgecolor='black')
                    plt.title(f"{col_name} Distribution")
                    plt.xlabel(col_name)
                    plt.ylabel("Frequency")
                    st.pyplot(fig)

            with col2:
                if i + 1 < len(numeric_cols):
                    col_name = numeric_cols[i + 1]
                    st.markdown(f"**{col_name} Distribution**")
                    fig, ax = plt.subplots()
                    df[col_name].hist(
                        bins=30, ax=ax, color='lightcoral', edgecolor='black')
                    plt.title(f"{col_name} Distribution")
                    plt.xlabel(col_name)
                    plt.ylabel("Frequency")
                    st.pyplot(fig)

    # TAB 5: CORRELATIONS
    with tab5:
        st.subheader("Correlation Analysis")

        # Encode categorical variables for correlation
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = pd.factorize(df_encoded[col])[0]

        correlation = df_encoded.corr()

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title("Correlation Matrix - Heart Disease Dataset")
        st.pyplot(fig)

        st.subheader("Correlation with Heart Disease")
        target_col = df_encoded.columns[-1]
        disease_corr = correlation[target_col].sort_values(ascending=False)
        st.dataframe(disease_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<p>📊 EDA & Preprocessing Dashboard | Built with Streamlit & Python</p>"
    "</div>",
    unsafe_allow_html=True
)
