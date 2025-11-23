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
