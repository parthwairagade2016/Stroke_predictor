import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# -------------------------------
# LOAD & PREPARE DATA
# -------------------------------
df = pd.read_csv("Strokes.csv")   # <<== CORRECTED PATH
df = df.drop(["id", "gender"], axis=1)
df = df.fillna(0)

# Manual readable UI options
work_type_options = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
residence_options = {"Rural": 0, "Urban": 1}
ever_married_options = {"No": 0, "Yes": 1}
smoking_options = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}

# Label encoding for training
label = LabelEncoder()
df.work_type = label.fit_transform(df.work_type)
df.Residence_type = label.fit_transform(df.Residence_type)
df.ever_married = label.fit_transform(df.ever_married)
df.smoking_status = label.fit_transform(df.smoking_status)

x = df.drop(["stroke"], axis=1).values
y = df["stroke"].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# -------------------------------
# TRAIN MODELS
# -------------------------------
rf_model = RandomForestClassifier().fit(xtrain, ytrain)
knn_model = KNeighborsClassifier(n_neighbors=5).fit(xtrain, ytrain)

rf_acc = rf_model.score(xtest, ytest)
knn_acc = knn_model.score(xtest, ytest)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🩺 Stroke Risk Predictor (Random Forest + KNN)")
st.write("Fill in patient details to estimate stroke likelihood.")

# Sidebar accuracy
st.sidebar.title("📊 Model Accuracy")
st.sidebar.write(f"**Random Forest Accuracy:** {round(rf_acc * 100, 2)}%")
st.sidebar.write(f"**KNN Accuracy:** {round(knn_acc * 100, 2)}%")

patient_name = st.text_input("Patient Name")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 120, 45)
    hypertension = st.radio("Hypertension", ["No", "Yes"])
    heart_disease = st.radio("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", list(ever_married_options.keys()))

with col2:
    avg_glucose = st.number_input("Average Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    work_type = st.selectbox("Work Type", list(work_type_options.keys()))
    residence = st.selectbox("Residence Type", list(residence_options.keys()))
    smoking_status = st.selectbox("Smoking Status", list(smoking_options.keys()))

# -------------------------------
# PREDICTION
# -------------------------------
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

    st.subheader(f"🧬 Stroke Risk for **{patient_name}**")
    st.write(f"### 🔍 Predicted Probability: **{percentage}%**")

    # Risk messages
    if percentage >= 70:
        st.error("🚨 High Risk! Immediate medical attention recommended.")
    elif percentage >= 25:
        st.warning("⚠️ Moderate Risk. Health monitoring advised.")
    elif percentage >= 20:
        st.warning("⚠️ Slight Risk. Health monitoring advised.")
    else:
        st.success("🟢 Low Risk.")

    # -------------------------------
    # INTERACTIVE PIE CHART
    # -------------------------------
    features = [
        "Age", "Hypertension", "Heart Disease", "Ever Married",
        "Avg Glucose", "BMI", "Work Type", "Residence", "Smoking Status"
    ]
    values = user_data[0]

    pie_df = pd.DataFrame({
        "Feature": features,
        "Value": values
    })

    fig = px.pie(
        pie_df,
        names="Feature",
        values="Value",
        title=f"Feature Contribution Breakdown for {patient_name}",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)
