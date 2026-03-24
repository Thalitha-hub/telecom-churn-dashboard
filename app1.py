# ============================================================
# TELECOM CHURN INTELLIGENCE PLATFORM
# Premium Streamlit App with Homepage + Methodology + Dashboard
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #081120 0%, #0f172a 45%, #111827 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: #07101d;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 34px 30px;
        border-radius: 28px;
        background:
            radial-gradient(circle at top right, rgba(59,130,246,0.30), transparent 28%),
            radial-gradient(circle at bottom left, rgba(168,85,247,0.22), transparent 25%),
            linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 18px 50px rgba(0,0,0,0.26);
        margin-bottom: 18px;
    }

    .glass {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        margin-bottom: 16px;
    }

    .metric-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        min-height: 118px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }

    .metric-title {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-sub {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 8px;
    }

    .pill {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .pill-blue {
        background: rgba(59,130,246,0.16);
        color: #93c5fd;
    }

    .pill-purple {
        background: rgba(168,85,247,0.16);
        color: #d8b4fe;
    }

    .pill-green {
        background: rgba(34,197,94,0.16);
        color: #86efac;
    }

    .pill-orange {
        background: rgba(245,158,11,0.16);
        color: #fcd34d;
    }

    .pill-red {
        background: rgba(239,68,68,0.16);
        color: #fca5a5;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
    }

    .muted {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .timeline-step {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 12px;
    }

    .timeline-step b {
        color: #ffffff;
    }

    .cta {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(37,99,235,0.16), rgba(139,92,246,0.16));
        border: 1px solid rgba(255,255,255,0.09);
    }

    .stDownloadButton button,
    .stButton button {
        border-radius: 12px;
        font-weight: 700;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_div(a, b):
    return (a / b) * 100 if b else 0

def money(x):
    return f"${x:,.2f}"

def risk_level(prob):
    if prob <= 0.30:
        return "Low Risk"
    elif prob <= 0.60:
        return "Medium Risk"
    return "High Risk"

def recommend_action(prob):
    if prob <= 0.30:
        return "Maintain regular engagement"
    elif prob <= 0.60:
        return "Target with loyalty messaging and service review"
    return "Immediate retention outreach with incentive and support"

def render_metric(title, value, sub):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def customer_id_column(df):
    possible = [
        "customerID", "CustomerID", "customer_id", "id", "ID",
        "AccountNumber", "PhoneNumber", "Customer"
    ]
    for col in possible:
        if col in df.columns:
            return col
    return None

@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

def build_bar_chart(risk_counts):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    ax.bar(risk_counts.index, risk_counts.values, color=colors, width=0.6)
    ax.set_title("Customer Risk Distribution", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Risk Level", color="white")
    ax.set_ylabel("Number of Customers", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    return fig

def build_histogram(probabilities):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.hist(probabilities, bins=20, color="#3b82f6", edgecolor="white")
    ax.set_title("Churn Probability Distribution", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Probability", color="white")
    ax.set_ylabel("Customer Count", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    return fig

def try_shap(model, data_processed):
    try:
        sample = data_processed.sample(min(100, len(data_processed)), random_state=42)

        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, sample)
            shap_values = explainer.shap_values(sample)
            return sample, shap_values, None
        else:
            explainer = shap.Explainer(model, sample)
            shap_values = explainer(sample)
            return sample, shap_values, None
    except Exception as e:
        return None, None, str(e)

# ------------------------------------------------------------
# STATIC CONTENT FROM PROJECT STORY
# ------------------------------------------------------------
HOME_HIGHLIGHTS = [
    "Predict churn probability for telecom customers",
    "Segment customers into low, medium, and high risk",
    "Estimate revenue exposed to churn",
    "Support retention teams with customer-level actions",
    "Explain predictions using SHAP"
]

METHODOLOGY_STEPS = [
    ("Business Understanding", "Define churn as a major revenue risk and frame the project around proactive customer retention."),
    ("Initial Data Inspection", "Review customer demographics, service usage, payment behavior, and churn patterns."),
    ("Exploratory Data Analysis", "Study churn distribution, tenure behavior, monthly charges, and contract effects."),
    ("Data Preparation", "Handle missing values, encode categorical features, engineer business features, and standardize inputs."),
    ("Model Development", "Train and compare Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM."),
    ("Model Selection", "Choose the best model based on predictive performance and business interpretability."),
    ("Explainability", "Use SHAP to identify the strongest drivers behind churn risk."),
    ("Business Impact Simulation", "Estimate preventable churn, protected revenue, and retention campaign value."),
    ("Web Application Delivery", "Convert the notebook work into an executive-ready Streamlit decision platform.")
]

BUSINESS_QUESTIONS = [
    "Which customers are most likely to churn?",
    "How much revenue is currently exposed?",
    "Which service or pricing patterns are linked to churn?",
    "Which customers should retention teams contact first?",
    "What actions should be taken for each risk segment?"
]

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.title("Telecom Intelligence")
    st.caption("Executive AI Retention Platform")

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Prediction Studio",
            "Methodology",
            "Executive Summary"
        ]
    )

    st.markdown("---")
    st.subheader("Business Assumptions")

    annual_revenue = st.number_input(
        "Average annual revenue per customer ($)",
        min_value=0.0,
        value=600.0,
        step=50.0
    )

    preventable_rate = st.slider(
        "Estimated preventable churn among high-risk customers (%)",
        min_value=1,
        max_value=100,
        value=15
    )

    top_n = st.slider(
        "Top high-risk customers to show",
        min_value=5,
        max_value=100,
        value=20
    )

    st.markdown("---")
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload telecom customer dataset (.csv)",
        type=["csv"]
    )

    notebook_path = "Business Understanding and Data Exploration.ipynb"
    if os.path.exists(notebook_path):
        with open(notebook_path, "rb") as f:
            st.download_button(
                "⬇ Download Project Notebook",
                data=f,
                file_name="Business Understanding and Data Exploration.ipynb",
                mime="application/x-ipynb+json"
            )

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Model file could not be loaded: {e}")
    st.stop()

# ------------------------------------------------------------
# GLOBAL HERO
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:8px;">📊 Telecom Churn Intelligence Platform</h1>
    <p style="font-size:1.06rem; color:#dbeafe; margin-bottom:14px;">
        An executive-grade AI product for churn prediction, retention prioritization, explainable insights, and revenue protection.
    </p>
    <span class="pill pill-blue">AI Prediction</span>
    <span class="pill pill-purple">Explainable Intelligence</span>
    <span class="pill pill-green">Retention Strategy</span>
    <span class="pill pill-orange">Revenue Impact</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if page == "Home":
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Platform Overview</div>', unsafe_allow_html=True)
        st.write(
            """
            This platform helps telecom companies move from reactive churn management to proactive retention.
            Instead of waiting for customers to leave, the system predicts churn risk early, highlights the most
            exposed customer segments, explains the drivers behind churn, and guides business teams toward the
            right intervention strategy.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What This Product Does</div>', unsafe_allow_html=True)
        for item in HOME_HIGHLIGHTS:
            st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Why It Matters</div>', unsafe_allow_html=True)
        st.write(
            """
            Customer churn directly affects recurring revenue, customer lifetime value, and market stability.
            This platform turns churn data into clear decisions for operations, marketing, service, and leadership.
            """
        )
        st.markdown("""
        <span class="pill pill-red">Churn Risk Visibility</span>
        <span class="pill pill-blue">Better Campaign Targeting</span>
        <span class="pill pill-green">Revenue Protection</span>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cta">', unsafe_allow_html=True)
        st.markdown("### Ready to analyze churn?")
        st.write("Upload a telecom customer CSV from the sidebar and open Prediction Studio.")
        st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("Core Goal", "Reduce Churn", "Predict and intervene before customer loss")
    with c2:
        render_metric("Primary Output", "Risk Segments", "Low, medium, and high risk classifications")
    with c3:
        render_metric("Strategic Value", "Retention ROI", "Focus resources where they matter most")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How the Platform Works</div>', unsafe_allow_html=True)

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.info("1. Upload customer data")
    with w2:
        st.info("2. Predict churn probabilities")
    with w3:
        st.info("3. Explain risk drivers")
    with w4:
        st.info("4. Trigger retention action")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Business Questions This App Answers</div>', unsafe_allow_html=True)
    for q in BUSINESS_QUESTIONS:
        st.markdown(f"- {q}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# METHODOLOGY PAGE
# ------------------------------------------------------------
elif page == "Methodology":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project Methodology</div>', unsafe_allow_html=True)
    st.write(
        """
        This page turns your notebook process into a business-friendly story.
        It helps users understand that the dashboard is backed by a real analytical pipeline,
        not just a black-box prediction screen.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    for title, desc in METHODOLOGY_STEPS:
        st.markdown(
            f"""
            <div class="timeline-step">
                <b>{title}</b><br>
                <span class="muted">{desc}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What This Adds to the Product</div>', unsafe_allow_html=True)
    st.write(
        """
        Adding this methodology section makes the product more credible for lecturers, executives,
        recruiters, judges, investors, and business stakeholders. It shows that the dashboard sits on top
        of a complete workflow: business framing, exploration, preprocessing, modeling, explainability,
        and impact analysis.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# EXECUTIVE SUMMARY PAGE
# ------------------------------------------------------------
elif page == "Executive Summary":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.write(
        """
        The platform addresses a major telecom business problem: customer churn. It uses machine learning
        to identify at-risk customers before they leave, supports better retention targeting, and provides
        explainable insights that business teams can act on. The intended outcome is stronger loyalty,
        better use of retention resources, and measurable revenue protection.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("Problem", "Customer Churn", "A direct threat to recurring revenue")
    with c2:
        render_metric("Solution", "Predictive Analytics", "Identify risk before churn happens")
    with c3:
        render_metric("Outcome", "Retention Optimization", "Protect revenue and improve targeting")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Strategic Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    - Deploy predictive monitoring into business workflows  
    - Prioritize outreach to high-risk customers  
    - Improve service quality around churn drivers  
    - Build retention planning around data, not guesswork
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# PREDICTION STUDIO PAGE
# ------------------------------------------------------------
elif page == "Prediction Studio":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Studio</div>', unsafe_allow_html=True)
    st.write("Upload a telecom customer dataset from the sidebar to activate the prediction workflow.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sample template section
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Template</div>', unsafe_allow_html=True)

    sample_df = pd.DataFrame({
        "tenure": [12, 24, 3],
        "MonthlyCharges": [70.2, 99.5, 50.3],
        "TotalCharges": [840, 2380, 150],
        "Contract": ["Month-to-month", "Two year", "Month-to-month"],
        "InternetService": ["Fiber optic", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "TechSupport": ["No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer", "Mailed check"],
        "SeniorCitizen": [0, 0, 1],
        "Partner": ["Yes", "Yes", "No"],
        "Dependents": ["No", "Yes", "No"]
    })

    st.dataframe(sample_df, use_container_width=True)
    st.download_button(
        "⬇ Download Sample CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_customer_data.csv",
        mime="text/csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.info("No dataset uploaded yet. Use the sidebar to upload your CSV file.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()

        if data.empty:
            st.warning("The uploaded CSV file is empty.")
            st.stop()

        original_data = data.copy()

        # Basic cleaning
        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = data[col].fillna(data[col].median())

        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = data[col].fillna("Unknown")

        # preprocessing
        try:
            data_processed = pd.get_dummies(data)
            model_features = model.feature_names_in_
            data_processed = data_processed.reindex(columns=model_features, fill_value=0)
            data_processed = data_processed.astype(float)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        # predictions
        try:
            probabilities = model.predict_proba(data_processed)[:, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        data["Churn Probability"] = probabilities
        data["Risk Level"] = data["Churn Probability"].apply(risk_level)
        data["Recommended Action"] = data["Churn Probability"].apply(recommend_action)

        total_customers = len(data)
        high_risk = int((data["Risk Level"] == "High Risk").sum())
        med_risk = int((data["Risk Level"] == "Medium Risk").sum())
        low_risk = int((data["Risk Level"] == "Low Risk").sum())
        avg_prob = float(data["Churn Probability"].mean())

        high_pct = safe_div(high_risk, total_customers)
        med_pct = safe_div(med_risk, total_customers)
        low_pct = safe_div(low_risk, total_customers)

        preventable_customers = round(high_risk * (preventable_rate / 100))
        revenue_at_risk = high_risk * annual_revenue
        revenue_saved = preventable_customers * annual_revenue

        sorted_data = data.sort_values("Churn Probability", ascending=False).reset_index(drop=True)
        id_col = customer_id_column(sorted_data)

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric("Customers Analyzed", f"{total_customers:,}", "Uploaded records")
        with k2:
            render_metric("High Risk Customers", f"{high_risk:,}", f"{high_pct:.1f}% of all customers")
        with k3:
            render_metric("Average Churn Probability", f"{avg_prob:.2f}", "Portfolio churn exposure")
        with k4:
            render_metric("Estimated Revenue Saved", money(revenue_saved), "Based on current assumptions")

        k5, k6, k7, k8 = st.columns(4)
        with k5:
            render_metric("Low Risk", f"{low_risk:,}", f"{low_pct:.1f}%")
        with k6:
            render_metric("Medium Risk", f"{med_risk:,}", f"{med_pct:.1f}%")
        with k7:
            render_metric("Revenue At Risk", money(revenue_at_risk), "High-risk annualized revenue")
        with k8:
            render_metric("Preventable Customers", f"{preventable_customers:,}", f"{preventable_rate}% of high-risk group")

        if high_pct > 30:
            st.error("High churn exposure detected. Immediate retention action is recommended.")
        elif high_pct > 15:
            st.warning("Moderate churn exposure detected. Focused retention campaigns are recommended.")
        else:
            st.success("Customer base appears relatively stable under the current analysis.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Predictions",
            "Top High Risk",
            "Explainable AI",
            "Retention Strategy"
        ])

        with tab1:
            left, right = st.columns(2)

            with left:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                risk_counts = data["Risk Level"].value_counts().reindex(
                    ["Low Risk", "Medium Risk", "High Risk"],
                    fill_value=0
                )
                st.subheader("Risk Distribution")
                st.pyplot(build_bar_chart(risk_counts), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Probability Distribution")
                st.pyplot(build_histogram(data["Churn Probability"]), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Customer Data Preview")
            st.dataframe(original_data.head(20), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Prediction Results")

            filter_choice = st.selectbox(
                "Filter by risk level",
                ["All", "High Risk", "Medium Risk", "Low Risk"]
            )

            if filter_choice == "All":
                filtered = sorted_data.copy()
            else:
                filtered = sorted_data[sorted_data["Risk Level"] == filter_choice].copy()

            st.dataframe(filtered, use_container_width=True)

            st.download_button(
                "⬇ Download Predictions",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="telecom_churn_predictions.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Top High-Risk Customers")

            top_high = sorted_data[sorted_data["Risk Level"] == "High Risk"].head(top_n).copy()

            if top_high.empty:
                st.info("No customers fall in the high-risk category under the current input.")
            else:
                keep_cols = []
                if id_col:
                    keep_cols.append(id_col)

                for col in [
                    "tenure", "MonthlyCharges", "TotalCharges", "Contract",
                    "InternetService", "TechSupport", "PaymentMethod"
                ]:
                    if col in top_high.columns and col not in keep_cols:
                        keep_cols.append(col)

                keep_cols += ["Churn Probability", "Risk Level", "Recommended Action"]
                keep_cols = [c for c in keep_cols if c in top_high.columns]

                st.dataframe(top_high[keep_cols], use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Explainable AI")

            sample_data, shap_values, shap_error = try_shap(model, data_processed)

            if shap_error:
                st.warning(f"SHAP output could not be generated for this model setup. Details: {shap_error}")
            else:
                try:
                    plt.figure(figsize=(10, 6))
                    if hasattr(shap_values, "values"):
                        shap.summary_plot(
                            shap_values.values,
                            sample_data,
                            feature_names=sample_data.columns,
                            show=False
                        )
                    else:
                        shap.summary_plot(
                            shap_values,
                            sample_data.values,
                            feature_names=sample_data.columns,
                            show=False
                        )

                    fig = plt.gcf()
                    fig.patch.set_facecolor("#111827")
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP chart failed to render: {e}")

            st.write(
                """
                Red points generally indicate features increasing churn risk, while blue points indicate
                features reducing churn likelihood. This helps business teams understand the main drivers
                behind customer loss.
                """
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with tab5:
            left, right = st.columns(2)

            with left:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Recommended Retention Strategy")
                st.markdown(f"""
                **High-risk customers**
                - Immediate outreach to {high_risk:,} customers
                - Offer incentives, loyalty bundles, or service recovery
                - Escalate customers with technical support or payment friction

                **Medium-risk customers**
                - Re-engage {med_risk:,} customers with targeted messaging
                - Review pricing, bundles, and service satisfaction

                **Low-risk customers**
                - Maintain positive experience
                - Monitor movement over time and protect loyalty
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Financial Snapshot")
                impact_df = pd.DataFrame({
                    "Metric": [
                        "Customers analyzed",
                        "High-risk customers",
                        "Estimated preventable customers",
                        "Average annual revenue per customer",
                        "Revenue at risk",
                        "Estimated revenue saved"
                    ],
                    "Value": [
                        f"{total_customers:,}",
                        f"{high_risk:,}",
                        f"{preventable_customers:,}",
                        money(annual_revenue),
                        money(revenue_at_risk),
                        money(revenue_saved)
                    ]
                })
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Action Table")

            outreach_cols = []
            if id_col:
                outreach_cols.append(id_col)

            for col in [
                "Risk Level", "Churn Probability", "Recommended Action",
                "Contract", "InternetService", "TechSupport",
                "MonthlyCharges", "tenure"
            ]:
                if col in sorted_data.columns and col not in outreach_cols:
                    outreach_cols.append(col)

            st.dataframe(sorted_data[outreach_cols].head(50), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)