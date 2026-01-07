import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Hierarchical Clustering App")

st.title("🏦 Credit Card Customer Segmentation")
st.write("Hierarchical Clustering (Agglomerative)")

# ================= LOAD MODEL =================
if not os.path.exists("hierarchical_predictor.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Model or scaler file not found")
    st.stop()

with open("hierarchical_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.success("✅ Model loaded successfully")

st.divider()

# ================= INPUTS =================
n_features = scaler.n_features_in_

st.subheader("Enter Customer Details")

inputs = []
for i in range(n_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# ================= PREDICTION =================
if st.button("🔍 Predict Cluster"):
    try:
        data = np.array(inputs).reshape(1, -1)
        data_scaled = scaler.transform(data)
        cluster = model.predict(data_scaled)

        st.success(f"✅ Customer belongs to Cluster: {cluster[0]}")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
