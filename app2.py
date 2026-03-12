# ----------------------------------
# IMPORT LIBRARIES
# ----------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
st.set_page_config(
    page_title="Telecom Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# HERO INTRODUCTION
# ----------------------------------
st.markdown("""
# 📊 Telecom Customer Churn Intelligence Dashboard

Welcome to the **AI-powered Customer Churn Prediction System**.

This platform helps telecom companies **identify customers likely to churn** and prioritize retention strategies.

### 🎯 Business Value
- Detect customers likely to leave
- Prioritize retention campaigns
- Reduce customer churn
- Understand drivers of churn using AI

---
""")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("churn_model.pkl", "rb"))
    return model

model = load_model()

# ----------------------------------
# DATASET REQUIREMENTS
# ----------------------------------
st.subheader("📂 Required Dataset Format")

st.write("""
Upload a **CSV dataset containing telecom customer information**.

Each row should represent **one customer**.

Typical columns include:

- tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- OnlineSecurity
- TechSupport
- PaymentMethod
- SeniorCitizen
- Partner
- Dependents
""")

# ----------------------------------
# EXAMPLE DATASET
# ----------------------------------
example_data = pd.DataFrame({
    "tenure":[12,24,3],
    "MonthlyCharges":[70.2,99.5,50.3],
    "TotalCharges":[840,2380,150],
    "Contract":["Month-to-month","Two year","Month-to-month"],
    "InternetService":["Fiber optic","DSL","Fiber optic"]
})

st.write("Example dataset format:")
st.dataframe(example_data)

# ----------------------------------
# DOWNLOAD SAMPLE CSV
# ----------------------------------
sample_csv = example_data.to_csv(index=False)

st.download_button(
    label="⬇ Download Sample CSV",
    data=sample_csv,
    file_name="sample_customer_data.csv",
    mime="text/csv"
)

# ----------------------------------
# WORKFLOW
# ----------------------------------
st.markdown("""
### 🔄 Prediction Workflow

1️⃣ Upload customer dataset  
2️⃣ AI model analyzes customer data  
3️⃣ System predicts churn probability  
4️⃣ Dashboard displays risk insights
""")

# ----------------------------------
# FILE UPLOADER
# ----------------------------------
st.subheader("📤 Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload telecom customer dataset (.csv)",
    type=["csv"]
)

# ----------------------------------
# PROCESS DATA
# ----------------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Customer Data Preview")

    st.dataframe(data.head(), use_container_width=True)

    # ----------------------------------
    # PREPROCESS DATA
    # ----------------------------------
    data_processed = pd.get_dummies(data)

    model_features = model.feature_names_in_

    data_processed = data_processed.reindex(columns=model_features, fill_value=0)

    data_processed = data_processed.astype(float)

    # ----------------------------------
    # PREDICT CHURN
    # ----------------------------------
    probabilities = model.predict_proba(data_processed)[:,1]

    data["Churn Probability"] = probabilities

    # ----------------------------------
    # RISK CLASSIFICATION
    # ----------------------------------
    def risk_level(prob):

        if prob <= 0.30:
            return "Low Risk"

        elif prob <= 0.60:
            return "Medium Risk"

        else:
            return "High Risk"

    data["Risk Level"] = data["Churn Probability"].apply(risk_level)

    # ----------------------------------
    # KPI DASHBOARD
    # ----------------------------------
    st.subheader("📈 Churn Risk Overview")

    total_customers = len(data)

    high_risk = (data["Risk Level"] == "High Risk").sum()
    medium_risk = (data["Risk Level"] == "Medium Risk").sum()
    low_risk = (data["Risk Level"] == "Low Risk").sum()

    avg_churn = data["Churn Probability"].mean()

    if total_customers > 0:
        high_pct = (high_risk / total_customers) * 100
        med_pct = (medium_risk / total_customers) * 100
        low_pct = (low_risk / total_customers) * 100
    else:
        high_pct = med_pct = low_pct = 0

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)

    col2.metric(
        "🔴 High Risk",
        high_risk,
        f"{high_pct:.1f}%"
    )

    col3.metric(
        "🟠 Medium Risk",
        medium_risk,
        f"{med_pct:.1f}%"
    )

    col4.metric(
        "🟢 Low Risk",
        low_risk,
        f"{low_pct:.1f}%"
    )

    st.info(f"Average churn probability: **{avg_churn:.2f}**")

    # ----------------------------------
    # RESULTS TABLE
    # ----------------------------------
    st.subheader("Prediction Results")

    st.dataframe(data, use_container_width=True)

    # ----------------------------------
    # RISK DISTRIBUTION CHART
    # ----------------------------------
    st.subheader("Customer Risk Distribution")

    risk_counts = data["Risk Level"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"],
        fill_value=0
    )

    fig, ax = plt.subplots()

    risk_counts.plot(
        kind="bar",
        ax=ax,
        color=["green", "orange", "red"]
    )

    ax.set_title("Customer Churn Risk Distribution", fontsize=14)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # ----------------------------------
    # SHAP EXPLANATION
    # ----------------------------------
    st.subheader("🔍 AI Model Explanation")

    # Use a sample for faster SHAP computation
    sample_data = data_processed.sample(min(100, len(data_processed)))

    explainer = shap.LinearExplainer(model, sample_data)

    shap_values = explainer.shap_values(sample_data)

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        sample_data.values,
        feature_names=sample_data.columns,
        show=False
    )

    st.pyplot(fig)

    st.write("""
Red features increase churn risk.  
Blue features decrease churn risk.

This helps telecom companies understand **why customers are leaving**.
""")

    # ----------------------------------
    # BUSINESS INSIGHTS
    # ----------------------------------
    st.subheader("📊 Business Insights")

    if high_pct > 30:

        st.error("⚠️ High churn risk detected. Immediate retention action recommended.")

    elif high_pct > 15:

        st.warning("⚠️ Moderate churn risk detected.")

    else:

        st.success("✅ Customer base appears relatively stable.")

    st.write("""
Recommended actions:

• Offer loyalty discounts to high-risk customers  
• Improve support for customers with technical issues  
• Promote long-term contracts  
• Analyze services linked to churn drivers
""")