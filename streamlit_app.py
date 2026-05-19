import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank Churn Intelligence System", page_icon="🏦", layout="wide")

# --- 2. CACHED DATA & MODEL TRAINING ---
@st.cache_resource
def load_data_and_train_model():
    # Load data
    df = pd.read_csv('European_Bank.csv')
    df_cleaned = df.drop(columns=['Year', 'CustomerId', 'Surname'])
    
    # Feature Engineering (New Requirements)
    df_cleaned['Balance_Salary_Ratio'] = df_cleaned['Balance'] / (df_cleaned['EstimatedSalary'] + 1e-5)
    df_cleaned['Product_Density'] = df_cleaned['NumOfProducts'] / (df_cleaned['Tenure'] + 1) # +1 to avoid division by zero
    df_cleaned['Engagement_Product'] = df_cleaned['IsActiveMember'] * df_cleaned['NumOfProducts']
    df_cleaned['Age_Tenure'] = df_cleaned['Age'] * df_cleaned['Tenure']
    
    # Encode Categorical Variables
    df_cleaned['Gender'] = df_cleaned['Gender'].map({'Female': 0, 'Male': 1})
    df_encoded = pd.get_dummies(df_cleaned, columns=['Geography'], drop_first=True)
    
    # Split
    X = df_encoded.drop('Exited', axis=1)
    y = df_encoded['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                      'EstimatedSalary', 'Balance_Salary_Ratio', 'Product_Density', 
                      'Engagement_Product', 'Age_Tenure']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # SMOTE and Train Model
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_smote, y_train_smote)
    
    # Generate test probabilities for the distribution plot
    test_probabilities = rf_model.predict_proba(X_test)[:, 1]
    
    return rf_model, scaler, X_train.columns, test_probabilities

# Load pipeline silently
model, scaler, feature_names, test_probs = load_data_and_train_model()

# --- 3. FRONTEND UI & TABS ---
st.title("🏦 Predictive Churn Intelligence System")
st.markdown("Advanced Risk Scoring & Scenario Analysis")

tab1, tab2, tab3 = st.tabs(["Risk Calculator & What-If Simulator", "Feature Importance Dashboard", "Probability Distribution"])

# --- TAB 1: CALCULATOR & SIMULATOR ---
with tab1:
    st.subheader("Customer What-If Simulator")
    st.write("Adjust engagement metrics below to observe real-time changes in churn probability.")
    
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
        
    with col2:
        balance = st.number_input("Account Balance", min_value=0.0, value=60000.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    # Calculate Derived Features for the Input
    bal_sal_ratio = balance / (salary + 1e-5)
    prod_density = num_products / (tenure + 1)
    is_active_num = 1 if is_active == 'Yes' else 0
    eng_prod = is_active_num * num_products
    age_tenure = age * tenure

    if st.button("Calculate Risk Score", type="primary"):
        input_data = pd.DataFrame({
            'CreditScore': [float(credit_score)],
            'Gender': [int(1 if gender == 'Male' else 0)],
            'Age': [float(age)],
            'Tenure': [float(tenure)],
            'Balance': [float(balance)],
            'NumOfProducts': [float(num_products)],
            'HasCrCard': [int(1 if has_cr_card == 'Yes' else 0)],
            'IsActiveMember': [is_active_num],
            'EstimatedSalary': [float(salary)],
            'Balance_Salary_Ratio': [float(bal_sal_ratio)],
            'Product_Density': [float(prod_density)],
            'Engagement_Product': [float(eng_prod)],
            'Age_Tenure': [float(age_tenure)],
            'Geography_Germany': [True if geography == 'Germany' else False],
            'Geography_Spain': [True if geography == 'Spain' else False]
        })
        
        # Ensure column order matches training
        input_data = input_data[feature_names]
        
        # Scale numerical columns
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                          'EstimatedSalary', 'Balance_Salary_Ratio', 'Product_Density', 
                          'Engagement_Product', 'Age_Tenure']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Predict Probability
        probability = model.predict_proba(input_data)[0][1]
        risk_score = round(probability * 100, 2)
        
        # Determine Risk Band
        if risk_score < 30:
            risk_band, color = "Low Risk", "green"
        elif risk_score < 70:
            risk_band, color = "Medium Risk", "orange"
        else:
            risk_band, color = "High Risk", "red"
            
        st.markdown(f"### Current Risk Probability: <span style='color:{color}'>{risk_score}% ({risk_band})</span>", unsafe_allow_html=True)
        st.progress(int(risk_score))

# --- TAB 2: FEATURE IMPORTANCE ---
with tab2:
    st.subheader("Global Feature Importance (Explainability)")
    st.write("Displays the key drivers influencing customer churn decisions across the entire dataset.")
    
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    # Reduced graph size here
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis', ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

# --- TAB 3: PROBABILITY DISTRIBUTION ---
with tab3:
    st.subheader("System-Wide Risk Distribution")
    st.write("Visualizing the spread of predicted churn probabilities across the tested customer base.")
    
    # Reduced graph size here
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(test_probs, bins=30, kde=True, color='purple', ax=ax2)
    ax2.set_title("Distribution of Customer Churn Probabilities")
    ax2.set_xlabel("Churn Risk Score (Probability)")
    ax2.set_ylabel("Number of Customers")
    ax2.axvline(x=0.3, color='green', linestyle='--', label='Low Risk Threshold (30%)')
    ax2.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold (70%)')
    ax2.legend()
    st.pyplot(fig2)
