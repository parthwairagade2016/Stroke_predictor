import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title="Stroke Predictor App", layout="wide")

st.title("🧠 Stroke Prediction App")
st.write("Provide patient details below to estimate stroke risk.")

# -------------------------------
# LOAD & PREPARE DATA
# -------------------------------
df = pd.read_csv("Strokes.csv")
df.drop(["id", "gender"], axis=1, inplace=True)
df = df.fillna(0)

label = LabelEncoder()
df["work_type"] = label.fit_transform(df["work_type"])
df["Residence_type"] = label.fit_transform(df["Residence_type"])
df["ever_married"] = label.fit_transform(df["ever_married"])
df["smoking_status"] = label.fit_transform(df["smoking_status"])

X = df.drop(["stroke"], axis=1).values
y = df["stroke"].values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# -------------------------------
# USER INPUT SECTION
# -------------------------------
st.header("📝 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input("Patient Name", "Unknown Patient", key="name")

    age = st.number_input("Age", min_value=0, max_value=120, value=45, key="age")
    hypertension = st.radio("Hypertension", ["No", "Yes"], key="hypertension")
    heart_disease = st.radio("Heart Disease", ["No", "Yes"], key="heart")

with col2:
    ever_married = st.radio("Ever Married", ["No", "Yes"], key="married")

    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"],
        key="work"
    )
    residence = st.selectbox(
        "Residence Type",
        ["Urban", "Rural"],
        key="residence"
    )

with col3:
    avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 105.0, key="glucose")
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key="bmi")
    smoking_status = st.selectbox(
        "Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
        key="smoke"
    )

# -------------------------------
# ENCODING USER INPUT
# -------------------------------
ht = 1 if hypertension == "Yes" else 0
hd = 1 if heart_disease == "Yes" else 0
em = 1 if ever_married == "Yes" else 0

work_map = {
    "Private": 3, "Self-employed": 4, "Govt_job": 0,
    "Children": 1, "Never_worked": 2
}
work_type_index = work_map[work_type]

res_map = {"Urban": 1, "Rural": 0}
res_index = res_map[residence]

smoke_map = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}
smoke_index = smoke_map[smoking_status]

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🔍 Predict Stroke Risk"):
    user_data = np.array([
        age, ht, hd, avg_glucose, bmi,
        work_type_index, res_index, smoke_index, em
    ]).reshape(1, -1)

    pred_prob = model.predict_proba(user_data)[0][1]
    percentage = round(pred_prob * 100, 2)

    st.subheader(f"🧠 Stroke Risk for **{patient_name}**: **{percentage}%**")

    # -------------------------------
    # RISK ALERTS
    # -------------------------------
    if percentage >= 70:
        st.error("🚨 **High Risk! Immediate medical attention recommended.**")
    elif percentage >= 25:
        st.warning("⚠️ **Moderate Risk. Regular monitoring advised.**")
    elif percentage >= 20:
        st.warning("⚠️ **Slight Risk. Health monitoring advised.**")
    else:
        st.success("🟢 **Low Risk.**")

    # -------------------------------
    # INTERACTIVE PIE CHART (PLOTLY)
    # -------------------------------
    st.subheader("📊 Feature Contribution Pie Chart")

    pie_data = {
        "Feature": [
            "Age", "Hypertension", "Heart Disease", "Avg Glucose",
            "BMI", "Work Type", "Residence Type", "Smoking Status",
            "Ever Married"
        ],
        "Value": [
            age, ht, hd, avg_glucose, bmi,
            work_type_index, res_index, smoke_index, em
        ]
    }

    fig = px.pie(
        pie_data,
        names="Feature",
        values="Value",
        hole=0.45,
        title=f"Impact Distribution for {patient_name}"
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")  
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
