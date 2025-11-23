import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
df = pd.read_csv("Strokes.csv")

# ---------------------------------------------
# LABEL ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------
gender_enc = LabelEncoder().fit(df["gender"])
work_enc = LabelEncoder().fit(df["work_type"])
residence_enc = LabelEncoder().fit(df["Residence_type"])
married_enc = LabelEncoder().fit(df["ever_married"])
smoke_enc = LabelEncoder().fit(df["smoking_status"])

# Encode full dataset
df["gender"] = gender_enc.transform(df["gender"])
df["work_type"] = work_enc.transform(df["work_type"])
df["Residence_type"] = residence_enc.transform(df["Residence_type"])
df["ever_married"] = married_enc.transform(df["ever_married"])
df["smoking_status"] = smoke_enc.transform(df["smoking_status"])

X = df.drop("stroke", axis=1)
y = df["stroke"]

# ---------------------------------------------
# TRAIN MODEL
# ---------------------------------------------
model = RandomForestClassifier()
model.fit(X, y)

# Feature importance
feature_importance = model.feature_importances_
features = X.columns

# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------
st.title("🧠 Stroke Prediction System")
st.write("Provide patient information to predict stroke risk.")

# Patient Name
patient_name = st.text_input("👤 Patient Name", "")

# User Inputs
gender = st.selectbox("Gender", gender_enc.classes_)
age = st.number_input("Age", min_value=1, max_value=120, value=45)

hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
ever_married = st.selectbox("Ever Married", married_enc.classes_)

work_type = st.selectbox("Work Type", work_enc.classes_)
residence = st.selectbox("Residence Type", residence_enc.classes_)
smoking = st.selectbox("Smoking Status", smoke_enc.classes_)

avg_glucose = st.number_input("Average Glucose Level", min_value=0.0, value=120.0)
bmi = st.number_input("BMI", min_value=0.0, value=28.0)

# Convert Yes/No to numbers
hypertension_val = 1 if hypertension == "Yes" else 0
heart_val = 1 if heart_disease == "Yes" else 0

# Encode categorical using original encoders
gender_val = gender_enc.transform([gender])[0]
married_val = married_enc.transform([ever_married])[0]
work_val = work_enc.transform([work_type])[0]
res_val = residence_enc.transform([residence])[0]
smoke_val = smoke_enc.transform([smoking])[0]

# ---------------------------------------------
# PREDICT
# ---------------------------------------------
if st.button("🔍 Predict Stroke"):

    # Create input vector in correct order
    input_data = np.array([[ 
        gender_val,
        age,
        hypertension_val,
        heart_val,
        married_val,
        work_val,
        res_val,
        avg_glucose,
        bmi,
        smoke_val
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    # Display result
    if prediction == 1:
        st.error(f"⚠️ HIGH Stroke Risk for {patient_name}")
    else:
        st.success(f"✅ LOW Stroke Risk for {patient_name}")

    st.subheader(f"Probability of Stroke: **{probability:.2f}%**")

    # ---------------------------------------------
    # PIE CHART (Feature Importance)
    # ---------------------------------------------
    st.subheader(f"📊 Factors Affecting Stroke for {patient_name}")

    importance_series = pd.Series(feature_importance, index=features)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(
        importance_series.values,
        labels=importance_series.index,
        autopct="%.1f%%",
        startangle=90
    )
    ax.set_title(f"Feature Importance for {patient_name}")
    st.pyplot(fig)

    st.info(
        "This pie chart shows how much each feature contributes to prediction "
        "based on the Random Forest model."
    )
