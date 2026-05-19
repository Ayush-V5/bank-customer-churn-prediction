import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")

# --- 2. CACHED MODEL TRAINING ---
@st.cache_resource
def load_data_and_train_model():
    # Load data
    df = pd.read_csv('European_Bank.csv')
    df_cleaned = df.drop(columns=['Year', 'CustomerId', 'Surname'])
    
    # Encode
    df_cleaned['Gender'] = df_cleaned['Gender'].map({'Female': 0, 'Male': 1})
    df_encoded = pd.get_dummies(df_cleaned, columns=['Geography'], drop_first=True)
    
    # Split
    X = df_encoded.drop('Exited', axis=1)
    y = df_encoded['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # SMOTE and Train
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_smote, y_train_smote)
    
    return rf_model, scaler

# Load the model and scaler silently
model, scaler = load_data_and_train_model()

# --- 3. FRONTEND USER INTERFACE ---
st.title("🏦 Bank Customer Churn Risk Predictor")
st.write("Enter the customer's details below to calculate their risk of leaving the bank.")

# Input fields arranged in columns
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

# --- 4. PREDICTION LOGIC ---
if st.button("Calculate Risk Score", type="primary"):
    # Format inputs exactly as the model expects
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [1 if gender == 'Male' else 0],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active == 'Yes' else 0],
        'EstimatedSalary': [salary],
        'Geography_Germany': [1 if geography == 'Germany' else 0],
        'Geography_Spain': [1 if geography == 'Spain' else 0]
    })
    
    # Scale numerical columns
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Predict Probability
    probability = model.predict_proba(input_data)[0][1]
    risk_score = round(probability * 100, 2)
    
    # Determine Risk Band and Color
    if risk_score < 30:
        risk_band = "Low Risk"
        color = "green"
    elif risk_score < 70:
        risk_band = "Medium Risk"
        color = "orange"
    else:
        risk_band = "High Risk"
        color = "red"
        
    # Display Results
    st.markdown(f"### Customer Risk Score: <span style='color:{color}'>{risk_score}/100 ({risk_band})</span>", unsafe_allow_html=True)
    st.progress(int(risk_score))
