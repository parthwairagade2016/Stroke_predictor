import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("Strokes.csv")
df = df.drop(["id", "gender"], axis=1)
df = df.fillna(0)

# Manual encoders for readable UI
work_type_options = {"Private":0, "Self-employed":1, "Govt_job":2, "children":3, "Never_worked":4}
residence_options = {"Rural":0, "Urban":1}
ever_married_options = {"No":0, "Yes":1}
smoking_options = {"formerly smoked":0, "never smoked":1, "smokes":2, "Unknown":3}

# Encode dataset for training
label = LabelEncoder()
df.work_type = label.fit_transform(df.work_type)
df.Residence_type = label.fit_transform(df.Residence_type)
df.ever_married = label.fit_transform(df.ever_married)
df.smoking_status = label.fit_transform(df.smoking_status)

x = df.drop(["stroke"], axis=1).values
y = df['stroke'].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Models
rf_model = RandomForestClassifier().fit(xtrain, ytrain)

# Streamlit UI
st.title("🩺 Stroke Predictor App")
st.write("Fill patient details below to predict stroke risk.")

patient_name = st.text_input("Patient Name", key="name_input")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45, key="age_input")
    hypertension = st.radio("Hypertension", ["No", "Yes"], key="hyper_input")
    heart_disease = st.radio("Heart Disease", ["No", "Yes"], key="heart_input")
    ever_married = st.selectbox("Ever Married", list(ever_married_options.keys()), key="married_input")

with col2:
    avg_glucose = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0, key="glucose_input")
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, key="bmi_input")
    work_type = st.selectbox("Work Type", list(work_type_options.keys()), key="work_input")
    residence = st.selectbox("Residence Type", list(residence_options.keys()), key="res_input")
    smoking_status = st.selectbox("Smoking Status", list(smoking_options.keys()), key="smoke_input")

if st.button("Predict Stroke Risk"):

    user_data = np.array([[age,
                           1 if hypertension == "Yes" else 0,
                           1 if heart_disease == "Yes" else 0,
                           ever_married_options[ever_married],
                           avg_glucose,
                           bmi,
                           work_type_options[work_type],
                           residence_options[residence],
                           smoking_options[smoking_status]]])

    prob = rf_model.predict_proba(user_data)[0][1]
    percentage = round(prob * 100, 2)

    st.subheader(f"Stroke Risk for {patient_name}:")
    st.write(f"### 🔍 Predicted Stroke Probability: **{percentage}%**")

    if percentage >= 70:
        st.error("🚨 High Risk! Immediate medical attention recommended.")
    elif percentage >= 25:
        st.warning("⚠️ Moderate Risk. Health monitoring advised.")
    elif percentage >= 20:
        st.warning("⚠️ Slight Risk. Health monitoring advised.")
    else:
        st.success("🟢 Low Risk.")

    # Pie chart
    features = ["Age", "Hypertension", "Heart Disease", "Ever Married",
                "Avg Glucose", "BMI", "Work Type", "Residence", "Smoking Status"]
    values = user_data[0]

    fig, ax = plt.subplots()
    ax.pie(values, labels=features, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Feature Contribution for {patient_name}")

    st.pyplot(fig)
