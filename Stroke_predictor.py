import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------
# LOAD & PREPROCESS DATA
# ---------------------------------------------
df = pd.read_csv("../Datasets/Strokes.csv")

label = LabelEncoder()
df["gender"] = label.fit_transform(df["gender"])
df["work_type"] = label.fit_transform(df["work_type"])
df["Residence_type"] = label.fit_transform(df["Residence_type"])
df["ever_married"] = label.fit_transform(df["ever_married"])
df["smoking_status"] = label.fit_transform(df["smoking_status"])

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train model
model_rf = RandomForestClassifier()
model_rf.fit(X, y)

# Feature importance for pie chart
feature_importance = model_rf.feature_importances_
features = X.columns

# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------
st.title("🧠 Stroke Prediction System")
st.write("Enter patient details to calculate stroke probability.")

# Patient name
patient_name = st.text_input("👤 Patient Name", "")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=50)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
avg_glucose = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
cholesterol = st.number_input("Cholesterol Level", min_value=0.0, value=180.0)

# Additional categorical fields
gender = st.selectbox("Gender", df["gender"].unique())
work_type = st.selectbox("Work Type", df["work_type"].unique())
residence = st.selectbox("Residence Type", df["Residence_type"].unique())
married = st.selectbox("Ever Married", df["ever_married"].unique())
smoking = st.selectbox("Smoking Status", df["smoking_status"].unique())

# ---------------------------------------------
# PREDICTION
# ---------------------------------------------
if st.button("🔍 Predict Stroke"):
    # Create input array
    input_data = np.array([[ 
        gender, age, hypertension, heart_disease, married,
        work_type, residence, avg_glucose, bmi, smoking
    ]])

    # Predict stroke
    prediction = model_rf.predict(input_data)[0]
    probability = model_rf.predict_proba(input_data)[0][1] * 100

    # Response
    if prediction == 1:
        st.error(f"⚠️ HIGH Stroke Risk for {patient_name}")
    else:
        st.success(f"✅ LOW Stroke Risk for {patient_name}")

    st.subheader(f"Probability of Stroke: **{probability:.2f}%**")

    # ---------------------------------------------
    # PIE CHART BASED ON FEATURE IMPORTANCE
    # ---------------------------------------------
    st.subheader(f"📊 Factors Affecting Stroke for {patient_name}")

    importance_series = pd.Series(feature_importance, index=features)

    # Pie Chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        importance_series.values,
        labels=importance_series.index,
        autopct="%.1f%%",
        startangle=90
    )
    ax.set_title(f"Feature Importance for {patient_name}")
    st.pyplot(fig)

    st.info("Note: This pie chart shows how much each factor **generally** affects stroke prediction using the model’s feature importance.")

