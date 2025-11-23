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

# Encode with LabelEncoder for model but preserve mapping
label = LabelEncoder()
df.work_type = label.fit_transform(df.work_type)
df.Residence_type = label.fit_transform(df.Residence_type)
df.ever_married = label.fit_transform(df.ever_married)
df.smoking_status = label.fit_transform(df.smoking_status)

x = df.drop(["stroke"], axis=1).values
y = df['stroke'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Models
dt_model = DecisionTreeClassifier().fit(xtrain, ytrain)
rf_model = RandomForestClassifier().fit(xtrain, ytrain)

# Streamlit UI
st.title("🩺 Stroke Predictor App")
st.write("Fill patient details below to predict stroke risk.")

patient_name = st.text_input("Patient Name")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45, key="age_input")
    hypertension = st.radio("Hypertension", ["No", "Yes"], key="hyper_input")("Hypertension", ["No", "Yes"])
    heart_disease = st.radio("Heart Disease", ["No", "Yes"], key="heart_input")("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", list(ever_married_options.keys()), key="married_input")("Ever Married", list(ever_married_options.keys()))

with col2:
    avg_glucose = st.number_input("Average Glucose Level", 0.0, 300.0, 100.0, key="glucose_input")("Average Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0, key="bmi_input")("BMI", 0.0, 60.0, 25.0)
    work_type = st.selectbox("Work Type", list(work_type_options.keys()), key="work_input")("Work Type", list(work_type_options.keys()))
    residence = st.selectbox("Residence Type", list(residence_options.keys()), key="res_input")("Residence Type", list(residence_options.keys()))
    smoking_status = st.selectbox("Smoking Status", list(smoking_options.keys()), key="smoke_input")("Smoking Status", list(smoking_options.keys()))

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

    # Prepare pie chart
    features = ["Age", "Hypertension", "Heart Disease", "Ever Married",
                "Avg Glucose", "BMI", "Work Type", "Residence", "Smoking Status"]
    values = user_data[0]

    fig, ax = plt.subplots()
    ax.pie(values, labels=features, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Feature Contribution for {patient_name}")

    st.pyplot(fig)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("../Datasets/Strokes.csv")
df = df.drop(["id", "gender"], axis=1)
df = df.fillna(0)
label = LabelEncoder()
df.work_type = label.fit_transform(df.work_type)
df.Residence_type = label.fit_transform(df.Residence_type)
df.ever_married = label.fit_transform(df.ever_married)
df.smoking_status = label.fit_transform(df.smoking_status)

x = df.drop(["stroke"], axis=1).values
y = df['stroke'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Models
dt_model = DecisionTreeClassifier()
dt_model.fit(xtrain, ytrain)
rf_model = RandomForestClassifier()
rf_model.fit(xtrain, ytrain)

# Streamlit UI
st.title("Stroke Predictor App")
st.write("Predict stroke probability and analyse patient risk.")

patient_name = st.text_input("Enter Patient Name")

age = st.number_input("Age", 0, 120, 30)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
ever_married = st.selectbox("Ever Married (Yes = 1, No = 0)", [0, 1])
avg_glucose = st.number_input("Average Glucose Level", 0.0, 300.0, 90.0)
bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
work_type = st.selectbox("Work Type (Encoded)", [0,1,2,3,4])
residence = st.selectbox("Residence Type (0 = Rural, 1 = Urban)", [0,1])
smoking_status = st.selectbox("Smoking Status (Encoded)", [0,1,2,3])

if st.button("Predict Stroke Risk"):
    user_data = np.array([[age, hypertension, heart_disease, ever_married,
                           avg_glucose, bmi, work_type, residence, smoking_status]])
    prob = rf_model.predict_proba(user_data)[0][1]
    percentage = round(prob * 100, 2)

    st.subheader(f"Stroke Risk for {patient_name}:")
    st.write(f"**Predicted Stroke Probability:** {percentage}%")

    if percentage >= 70:
        st.error("High Risk! Immediate medical consultation recommended.")
    elif percentage >= 40:
        st.warning("Moderate Risk. Monitor health closely.")
    else:
        st.success("Low Risk.")

    # Pie chart preparation
    features = ["age", "hypertension", "heart_disease", "ever_married",
                "avg_glucose", "bmi", "work_type", "residence", "smoking_status"]
    values = user_data[0]

    fig, ax = plt.subplots()
    ax.pie(values, labels=features, autopct="%1.1f%%")
    ax.set_title(f"Feature Contribution Breakdown for {patient_name}")
    st.pyplot(fig)
