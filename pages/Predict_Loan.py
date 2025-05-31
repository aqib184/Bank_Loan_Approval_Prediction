import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB

st.set_page_config(page_title="Loan Prediction", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
    df.loc[df['Experience'] < 0, 'Experience'] = abs(df['Experience'])
    return df

data = load_data()

# Prepare model
X = data.drop(columns=["Personal Loan", "ZIP Code", "ID", "Experience"])
y = data["Personal Loan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = BernoulliNB()
model.fit(X_train_scaled, y_train)

st.title("🤖 Predict Personal Loan Approval")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income (in thousands)", min_value=0, max_value=500, value=50)
family = st.selectbox("Family Size", [1, 2, 3, 4])
ccavg = st.number_input("CCAvg", min_value=0.0, max_value=10.0, value=1.5)
education = st.selectbox("Education", [1, 2, 3])
mortgage = st.number_input("Mortgage", min_value=0, max_value=1000, value=0)
securities = st.selectbox("Securities Account", [0, 1])
cd = st.selectbox("CD Account", [0, 1])
online = st.selectbox("Online Banking", [0, 1])
credit_card = st.selectbox("Credit Card", [0, 1])

input_data = pd.DataFrame([[
    age, income, family, ccavg, education, mortgage,
    securities, cd, online, credit_card
]], columns=X.columns)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

if st.button("Predict"):
    st.success("🟢 Loan Approved" if prediction == 1 else "🔴 Loan Not Approved")
    st.info(f"Probability of Approval: {prob:.2%}")
