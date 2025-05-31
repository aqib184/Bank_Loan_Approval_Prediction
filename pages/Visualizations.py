import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Visualizations", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
    df.loc[df['Experience'] < 0, 'Experience'] = abs(df['Experience'])
    return df

data = load_data()

st.title("📊 Visual Explorations")

# Heatmap
st.subheader("🔷 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap="Blues", fmt=".1f", ax=ax)
st.pyplot(fig)

# Pairplot
st.subheader("🔷 Pairplot")
st.info("Pairplot may take a while to render...")
st.pyplot(sns.pairplot(data))

# Histograms
st.subheader("🔷 Distributions")
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 5))
ax1.hist(data["Mortgage"], color="red")
ax1.set_title("Mortgage Distribution")
ax1.axvline(data.Mortgage.mean(), color="black")

ax2.hist(data["Experience"], color="green")
ax2.set_title("Experience Distribution")
ax2.axvline(0, color="black")

ax3.hist(data["Income"], color="orange")
ax3.set_title("Income Distribution")
ax3.axvline(data.Income.mean(), color="black")
st.pyplot(fig)

# Boxplots
st.subheader("🔷 Boxplots for Continuous Variables")
continuous_vars = [col for col in data.columns if data[col].nunique() > 5 and col not in ["ID", "Experience"]]
fig = plt.figure(figsize=(20, 10))
for i, col in enumerate(continuous_vars):
    ax = fig.add_subplot(2, 3, i + 1)
    sns.boxplot(y=data[col], x=data["Personal Loan"], color='cyan')
st.pyplot(fig)

# Barplots
st.subheader("🔷 Barplots for Categorical Variables")
categorical_vars = [col for col in data.columns if data[col].nunique() <= 5 and col != "Personal Loan"]
fig = plt.figure(figsize=(20, 10))
for i, col in enumerate(categorical_vars):
    ax = fig.add_subplot(2, 3, i + 1)
    sns.barplot(x=col, y='Personal Loan', data=data, color='salmon')
st.pyplot(fig)
