import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Advanced Placement Predictor", layout="centered")

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    placement_model = joblib.load("models/placement_model.pkl")
    salary_model = joblib.load("models/salary_model.pkl")
    features = joblib.load("models/features.pkl")
    return placement_model, salary_model, features

placement_model, salary_model, features = load_models()

# ---------------- TITLE ---------------- #
st.title("💼 Advanced Placement & Salary Predictor")
st.markdown("---")

# ---------------- INPUT SECTION ---------------- #
st.subheader("📋 Enter Student Profile")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("🎓 CGPA", 0.0, 10.0, 7.0)
    internships = st.slider("🏢 Internships", 0, 5, 1)
    dsa_score = st.slider("🧠 DSA Score (0-100)", 0, 100, 60)

with col2:
    projects = st.slider("💻 Projects", 0, 10, 2)
    certifications = st.slider("📜 Certifications", 0, 10, 1)
    aptitude_score = st.slider("📊 Aptitude Score (0-100)", 0, 100, 65)

branch = st.selectbox("🎓 Branch", ["CSE", "ECE", "EEE", "MECH", "CIVIL"])
college_tier = st.selectbox("🏫 College Tier", ["Tier 1", "Tier 2", "Tier 3"])
gender = st.radio("👤 Gender", ["Male", "Female"], horizontal=True)

st.markdown("---")

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict", use_container_width=True):

    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = 0

    # -------- Numeric -------- #
    numeric_inputs = {
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "certifications": certifications,
        "dsa_score": dsa_score,
        "aptitude_score": aptitude_score
    }

    for col in numeric_inputs:
        if col in input_df.columns:
            input_df[col] = numeric_inputs[col]

    # -------- Categorical -------- #
    categorical_inputs = [
        f"gender_{gender}",
        f"branch_{branch}",
        f"college_tier_{college_tier}"
    ]

    for col in categorical_inputs:
        if col in input_df.columns:
            input_df[col] = 1

    # -------- Prediction -------- #
    placement = placement_model.predict(input_df)[0]
    prob = placement_model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if placement == 1:
        st.success("🎉 Likely to be PLACED")

    # Predict salary
        salary_log = salary_model.predict(input_df)[0]

# Prevent overflow
        salary_log = np.clip(salary_log, 0, 5)
        salary = salary_model.predict(input_df)[0]

# safety
        if salary < 0:
            salary = 0

# Final safety
        if np.isinf(salary) or np.isnan(salary):
            salary = 0

        st.metric("💰 Expected Salary (LPA)", f"₹ {salary:.2f}")

    # Safety check (avoid negative salary)

    # Show range (more professional)
        lower = max(0, salary - 1)
        upper = salary + 1
        st.caption(f"Estimated Range: ₹ {lower:.2f} – ₹ {upper:.2f} LPA")

    # Debug (optional – remove later)
        st.write("Raw Salary Prediction:", salary)

    else:
        st.error("❌ Not likely to be placed")
        st.metric("📉 Placement Probability", f"{prob*100:.2f}%")

    # ---------------- GRAPH: IMPROVEMENT VS PROBABILITY ---------------- #
    st.markdown("---")
    st.subheader("📈 CGPA Improvement vs Placement Probability")

    cgpa_range = np.linspace(5, 10, 20)
    probs = []

    for val in cgpa_range:
        temp_df = input_df.copy()
        if "cgpa" in temp_df.columns:
            temp_df["cgpa"] = val
        p = placement_model.predict_proba(temp_df)[0][1]
        probs.append(p)

    fig, ax = plt.subplots()
    ax.plot(cgpa_range, probs)
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Placement Probability")
    ax.set_title("Impact of CGPA on Placement")
    st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ---------------- #
    st.markdown("---")
    st.subheader("📊 Feature Importance")

    if hasattr(placement_model, "feature_importances_"):
        importance = placement_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance_df["Feature"], importance_df["Importance"])
        ax2.invert_yaxis()
        ax2.set_title("Top 10 Important Features")
        st.pyplot(fig2)

    else:
        st.info("Feature importance not available for this model.")