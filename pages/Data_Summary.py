import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Summary", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
    df.loc[df['Experience'] < 0, 'Experience'] = abs(df['Experience'])
    return df

data = load_data()

st.title("📈 Data Summary")

st.subheader("🧾 Head of Data")
st.dataframe(data.head())

st.subheader("📊 Describe")
st.dataframe(data.describe())

st.subheader("🔍 Null Values")
st.dataframe(data.isnull().sum())

st.subheader("🔢 Unique Value Counts")
st.dataframe(data.nunique())

import io

st.subheader("🧠 Info")
buffer = io.StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.subheader("📈 Correlation Matrix")
st.dataframe(data.corr())
