import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Stroke Predictor", layout="centered")

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
df = pd.read_csv("Strokes.csv")

# Remove ID column if exists
if "id" in df.columns:
    df = df.drop("id", axis=1)

# ---------------------------------------------
# ENCODE CATEGORICAL COLUMNS
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

# ---------------------------------------------
# TRAIN MODEL
# ---------------------------------------------
X = df.drop("stroke", axis=1)
y = df["stroke"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save importance
importance = model.feature_importances_
features = X.columns


# ---------------------------------------------
# STREAMLIT USER INTERFACE
# ---------------------------------------------
st.title("🧠 Stroke Prediction System")
st.write("Enter the patient's details to predict stroke probability.")

patient_name = st.text_input("👤 Patient Name", value="Patient")

# --- USER INPUTS ---
gender_input = st.selectbox("Gender", gender_enc.classes_)
age = st.number_input("Age", min_value=1, max_value=120, value=45)

hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])

ever_married_input = st.selectbox("Ever Married", married_enc.classes_)
work_type_input = st.selectbox("Work Type", work_enc.classes_)
residence_input = st.selectbox("Residence Type", res_enc.classes_)
smoking_input = st.selectbox("Smoking Status", smoke_enc.classes_)

avg_glucose = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)


# --- ENCODE USER INPUTS ---
gender_val = gender_enc.transform([gender_input])[0]
married_val = married_enc.transform([ever_married_input])[0]
work_val = work_enc.transform([work_type_input])[0]
res_val = res_enc.transform([residence_input])[0]
smoke_val = smoke_enc.transform([smoking_input])[0]

hypertension_val = 1 if hypertension == "Yes" else 0
heart_val = 1 if heart_disease == "Yes" else 0


# ---------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------
if st.button("🔍 Predict Stroke"):

    # Arrange input in correct order
    input_data = np.array([[
        gender_val, age, hypertension_val, heart_val,
        married_val, work_val, res_val, avg_glucose, bmi, smoke_val
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ HIGH Stroke Risk for {patient_name}")
    else:
        st.success(f"✅ LOW Stroke Risk for {patient_name}")

    st.subheader(f"Probability of Stroke: **{probability:.2f}%**")

    # ---------------------------------------------
    # PIE CHART — Personalized Contribution
    # ---------------------------------------------
    st.subheader(f"📊 Factors Affecting Stroke for {patient_name}")

    importance_series = pd.Series(importance, index=features)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        importance_series.values,
        labels=importance_series.index,
        autopct="%.1f%%",
        startangle=90
    )
    ax.set_title(f"Feature Contribution for {patient_name}")
    st.pyplot(fig)

    st.info("Chart shows how much each parameter contributes to stroke prediction (model importance).")
