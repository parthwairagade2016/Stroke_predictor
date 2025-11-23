import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Stroke Predictor", layout="wide")

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
df = pd.read_csv("Strokes.csv")

if "id" in df.columns:
    df = df.drop("id", axis=1)

# ---------------------------------------------
# ENCODERS
# ---------------------------------------------
gender_enc = LabelEncoder()
work_enc = LabelEncoder()
res_enc = LabelEncoder()
married_enc = LabelEncoder()
smoke_enc = LabelEncoder()

df["gender"] = gender_enc.fit_transform(df["gender"])
df["work_type"] = work_enc.fit_transform(df["work_type"])
df["Residence_type"] = res_enc.fit_transform(df["Residence_type"])
df["ever_married"] = married_enc.fit_transform(df["ever_married"])
df["smoking_status"] = smoke_enc.fit_transform(df["smoking_status"])

X = df.drop("stroke", axis=1)
y = df["stroke"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importance = model.feature_importances_
features = X.columns

# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("🧠 AI-Powered Stroke Prediction System")
st.markdown("### Predict stroke probability and visualize factor contributions interactively.")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("👤 Enter Patient Name", placeholder="e.g. John Doe")

    gender_input = st.selectbox("Gender", gender_enc.classes_)
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])

with col2:
    ever_married_input = st.selectbox("Ever Married", married_enc.classes_)
    work_type_input = st.selectbox("Work Type", work_enc.classes_)
    residence_input = st.selectbox("Residence Type", res_enc.classes_)
    smoking_input = st.selectbox("Smoking Status", smoke_enc.classes_)
    avg_glucose = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)

# Encode inputs
gender_val = gender_enc.transform([gender_input])[0]
married_val = married_enc.transform([ever_married_input])[0]
work_val = work_enc.transform([work_type_input])[0]
res_val = res_enc.transform([residence_input])[0]
smoke_val = smoke_enc.transform([smoking_input])[0]

hypertension_val = 1 if hypertension == "Yes" else 0
heart_val = 1 if heart_disease == "Yes" else 0

# ---------------------------------------------
# PREDICT
# ---------------------------------------------
st.markdown("### 🔍 Click Predict to Analyze")

if st.button("✨ Predict Now"):

    input_data = np.array([[
        gender_val, age, hypertension_val, heart_val,
        married_val, work_val, res_val, avg_glucose, bmi, smoke_val
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.markdown(f"<h2 style='color:#FF4B4B;'>⚠️ HIGH Stroke Risk for {patient_name}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:#32CD32;'>✅ LOW Stroke Risk for {patient_name}</h2>", unsafe_allow_html=True)

    st.metric(label="Probability of Stroke", value=f"{probability:.2f}%")

    # -------------------------
    # INTERACTIVE PIE CHART
    # -------------------------
    st.markdown(f"### 📊 Factor Impact for {patient_name}")

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig = px.pie(
        imp_df,
        values="Importance",
        names="Feature",
        title=f"Stroke Risk Factor Contribution for {patient_name}",
        hole=0.45,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.05] * len(imp_df),   # Animation-like pop-out effect
        hoverinfo="label+percent",
    )

    fig.update_layout(
        showlegend=True,
        template="plotly_dark",
        margin=dict(t=50, b=20),
        transition_duration=500  # Smooth animation
    )

    st.plotly_chart(fig, use_container_width=True)
