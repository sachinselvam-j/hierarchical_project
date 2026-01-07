import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Hierarchical Clustering", layout="centered")

st.title("Hierarchical Clustering – Streamlit App")

# Load dataset
df = pd.read_csv("Mall_Customers (2).csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------- DENDROGRAM ----------------
st.subheader("Dendrogram")
fig1, ax1 = plt.subplots(figsize=(8,4))
dendrogram(linkage(X, method='ward'), ax=ax1)
ax1.set_xlabel("Customers")
ax1.set_ylabel("Distance")
st.pyplot(fig1)

# ---------------- CLUSTER SELECTION ----------------
st.subheader("Create Model")
k = st.slider("Select Number of Clusters", 2, 10, 5)

model = AgglomerativeClustering(
    n_clusters=k,
    linkage='ward'
)

df['Cluster'] = model.fit_predict(X)

# ---------------- VISUALIZATION ----------------
st.subheader("Cluster Visualization")
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=df['Cluster']
)
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.set_title("Hierarchical Clustering Result")
st.pyplot(fig2)

# ---------------- OUTPUT ----------------
st.subheader("Clustered Data")
st.dataframe(df.head())

st.success("Model created successfully (Hierarchical Clustering)")
