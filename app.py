import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.title("Hierarchical Clustering App")

# Load data
df = pd.read_csv("Mall_Customers (2).csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Load model
with open("hierarchical_model.pkl", "rb") as f:
    model = pickle.load(f)

# Re-fit (required for hierarchical)
df['Cluster'] = model.fit_predict(X)

st.subheader("Clustered Data")
st.dataframe(df.head())

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=df['Cluster']
)
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
st.pyplot(fig)
