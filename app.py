import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
# Removed import plotly.graph_objects as go
import matplotlib.pyplot as plt # --- NEW: Import matplotlib for the chart
import os

# --- 1. CONFIGURATION AND INITIAL SETUP ---
st.set_page_config(
    page_title="Stroke Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the expected features and their data types
FEATURE_COLUMNS = [
    'age', 'hypertension', 'heart_disease', 'ever_married', 
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 
    'smoking_status'
]

# --- 2. DATA LOADING AND MODEL TRAINING (Cached for Performance) ---

@st.cache_data
def load_and_preprocess_data():
    """Loads the dataset and performs the necessary preprocessing."""
    # NOTE: Since the file path "../Datasets/Strokes.csv" is external, 
    # we assume 'Strokes.csv' is in the current directory for execution environment.
    # If the file is not found, a placeholder structure is used.
    try:
        # Check for common names first
        file_path = 'Strokes.csv'
        if not os.path.exists(file_path):
             # Fallback to a common alternative name if needed
             file_path = 'stroke.csv' 
        
        if os.path.exists(file_path):
             df = pd.read_csv(file_path)
        else:
            st.error(f"Error: Could not find 'Strokes.csv' or 'stroke.csv'. Using synthetic data for structure.")
            # Create a synthetic DataFrame structure matching the expected columns
            df = pd.DataFrame({
                'id': range(10), 'gender': ['Male'] * 10, 'age': np.random.randint(18, 80, 10), 
                'hypertension': np.random.randint(0, 2, 10), 'heart_disease': np.random.randint(0, 2, 10),
                'ever_married': ['Yes'] * 10, 'work_type': ['Private'] * 10, 
                'Residence_type': ['Urban'] * 10, 'avg_glucose_level': np.random.rand(10) * 150 + 70, 
                'bmi': np.random.rand(10) * 20 + 20, 'smoking_status': ['never smoked'] * 10, 
                'stroke': np.random.randint(0, 2, 10)
            })
            
    except Exception as e:
        st.error(f"Failed to load or process data: {e}. Using synthetic data.")
        df = pd.DataFrame({
            'id': range(10), 'gender': ['Male'] * 10, 'age': np.random.randint(18, 80, 10), 
            'hypertension': np.random.randint(0, 2, 10), 'heart_disease': np.random.randint(0, 2, 10),
            'ever_married': ['Yes'] * 10, 'work_type': ['Private'] * 10, 
            'Residence_type': ['Urban'] * 10, 'avg_glucose_level': np.random.rand(10) * 150 + 70, 
            'bmi': np.random.rand(10) * 20 + 20, 'smoking_status': ['never smoked'] * 10, 
            'stroke': np.random.randint(0, 2, 10)
        })


    # Replicate original data cleaning and feature dropping
    df.drop(["id", "gender"], axis=1, inplace=True, errors='ignore')
    df = df.fillna(0) # Handle BMI NaNs as 0, matching original code

    # Replicate Label Encoding based on unique values in the loaded/synthetic data
    encoders = {}
    for col in ['work_type', 'Residence_type', 'ever_married', 'smoking_status']:
        if col in df.columns:
            le = LabelEncoder()
            # Ensure the encoder is fit on the full column for consistent mapping
            df[col] = le.fit_transform(df[col])
            encoders[col] = {
                'encoder': le, 
                'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
            }

    # Prepare data for model training
    X = df.drop("stroke", axis=1).values
    y = df['stroke'].values
    
    # Use all data to train the final model as the original code snippet suggests
    # finding the best model and then using it for prediction.
    return X, y, encoders

@st.cache_resource
def train_model(X, y):
    """Trains the best-found KNN model from the original code (n_neighbors=71)."""
    
    # Train-test split is used primarily for the original GridSearchCV but
    # for the final model we often train on the entire dataset.
    # We will use the parameters found in the original code: n_neighbors=71
    model = KNeighborsClassifier(n_neighbors=71)
    model.fit(X, y)
    return model

# Load data, get encodings, and train the model
X, y, encoders = load_and_preprocess_data()
model = train_model(X, y)

# --- 3. INPUT WIDGETS AND INTERACTIVE LAYOUT ---

st.title("🩺 Stroke Risk Prediction App")
st.markdown("""
Enter the patient's clinical and lifestyle information below to predict the likelihood of a stroke.
""")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name", value="John Doe")
    
    age = st.slider("Age", 0, 100, 50, help="Patient's age in years.")
    
    # Map 'Yes'/'No' to 1/0 as expected by the model for binary features
    hypertension = st.selectbox(
        "Hypertension (High Blood Pressure)", 
        options=['No', 'Yes'], index=0,
        format_func=lambda x: 'Yes' if x == 'Yes' else 'No',
        help="Patient has hypertension (1=Yes, 0=No)."
    )
    
    heart_disease = st.selectbox(
        "Heart Disease", 
        options=['No', 'Yes'], index=0,
        format_func=lambda x: 'Yes' if x == 'Yes' else 'No',
        help="Patient has any heart disease (1=Yes, 0=No)."
    )
    
    ever_married = st.selectbox(
        "Ever Married", 
        options=['No', 'Yes'], index=1,
        help="Has the patient ever been married?"
    )

with col2:
    # Use the mapping from the loaded data for categorical features
    
    # Get unique categories from the encoder if available, otherwise use defaults
    work_type_options = list(encoders['work_type']['mapping'].keys()) if 'work_type' in encoders else ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    work_type = st.selectbox("Work Type", options=work_type_options, index=0)
    
    Residence_type_options = list(encoders['Residence_type']['mapping'].keys()) if 'Residence_type' in encoders else ['Urban', 'Rural']
    Residence_type = st.selectbox("Residence Type", options=Residence_type_options, index=0)
    
    avg_glucose_level = st.number_input(
        "Average Glucose Level (mg/dL)", 
        min_value=50.0, max_value=300.0, value=95.0, step=0.1,
        help="Fasting plasma glucose test results."
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)", 
        min_value=10.0, max_value=60.0, value=25.0, step=0.1,
        help="Calculated as weight (kg) / height (m)^2."
    )
    
    smoking_status_options = list(encoders['smoking_status']['mapping'].keys()) if 'smoking_status' in encoders else ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
    smoking_status = st.selectbox("Smoking Status", options=smoking_status_options, index=0)

# --- 4. PREDICTION LOGIC ---

if st.button("Predict Stroke Risk", type="primary"):
    
    # 1. Convert user inputs into the numerical vector expected by the model
    
    # Binary/Simple Mapping
    hypertension_val = 1 if hypertension == 'Yes' else 0
    heart_disease_val = 1 if heart_disease == 'Yes' else 0
    ever_married_val = encoders['ever_married']['mapping'].get(ever_married, 0) if 'ever_married' in encoders else (1 if ever_married == 'Yes' else 0)

    # Label Encoded Mapping (using the cached encoders)
    work_type_val = encoders['work_type']['mapping'].get(work_type, 0) if 'work_type' in encoders else 0
    residence_type_val = encoders['Residence_type']['mapping'].get(Residence_type, 0) if 'Residence_type' in encoders else 0
    smoking_status_val = encoders['smoking_status']['mapping'].get(smoking_status, 0) if 'smoking_status' in encoders else 0
    
    # Create the input array
    input_data = np.array([
        age, 
        hypertension_val, 
        heart_disease_val, 
        ever_married_val, 
        work_type_val, 
        residence_type_val, 
        avg_glucose_level, 
        bmi, 
        smoking_status_val
    ]).reshape(1, -1)
    
    # 2. Make Prediction and Get Probability
    
    # The KNN model returns probability based on the ratio of neighbors in each class
    probabilities = model.predict_proba(input_data)[0]
    chance_of_stroke = probabilities[1] # Probability of class 1 (stroke)
    
    # 3. Calculate Heuristic Risk Contribution for Pie Chart (Custom Logic)
    
    # This section calculates a simplified, interpretable risk score based on
    # standard medical risk thresholds, serving as a proxy for 'factor contribution'.
    
    # Risk factor weights (can be adjusted)
    risk_weights = {
        "Age Risk": 30,
        "Glucose Risk": 25,
        "BMI Risk": 25,
        "Heart/HyperTension": 20
    }
    
    # Calculate individual risk scores (0-100) based on typical thresholds
    # Score is 0 if low risk, and increases up to 100 based on severity/presence.
    
    # Age Risk: High risk over 65 (arbitrary threshold for normalization)
    age_risk_score = min(100, max(0, (age - 40) / (65 - 40) * 100))
    
    # Glucose Risk: High risk over 125 mg/dL (diabetic)
    glucose_risk_score = min(100, max(0, (avg_glucose_level - 100) / (125 - 100) * 100))
    
    # BMI Risk: High risk over 30 (obesity)
    bmi_risk_score = min(100, max(0, (bmi - 25) / (30 - 25) * 100))
    
    # Heart/HyperTension Risk: Binary multiplier
    heart_hyper_risk_score = (hypertension_val * 0.5 + heart_disease_val * 0.5) * 100

    # Combine the risk score with its weight
    contributions = {
        "Age Risk": age_risk_score * risk_weights["Age Risk"] / 100,
        "Glucose Risk": glucose_risk_score * risk_weights["Glucose Risk"] / 100,
        "BMI Risk": bmi_risk_score * risk_weights["BMI Risk"] / 100,
        "Heart/HyperTension": heart_hyper_risk_score * risk_weights["Heart/HyperTension"] / 100,
    }
    
    # Normalize contributions to get percentages for the pie chart
    total_score = sum(contributions.values())
    if total_score > 0:
        contribution_percentages = {k: (v / total_score) * 100 for k, v in contributions.items()}
    else:
        # Default to equal contribution if risk score is zero
        contribution_percentages = {k: 100 / len(contributions) for k in contributions.keys()}
        
    pie_labels = list(contribution_percentages.keys())
    pie_values = list(contribution_percentages.values())
    
    
    # --- 5. DISPLAY RESULTS ---
    st.subheader(f"Results for Patient: {patient_name}")
    
    col_pred, col_chart = st.columns([1, 1])

    with col_pred:
        # Display the prediction and percentage
        st.metric(
            label="Predicted Stroke Risk", 
            value=f"{chance_of_stroke * 100:.2f}%", 
            delta=None # No delta for a single prediction
        )
        
        st.markdown("---")
        
        # Interpretive statement based on the percentage
        if chance_of_stroke >= 0.5:
            st.warning(f"**High Risk:** The model predicts a stroke chance of over 50%. Immediate medical consultation is strongly recommended.")
        elif chance_of_stroke >= 0.1:
            st.info(f"**Moderate Risk:** The model suggests a notable risk. Discuss these factors with a healthcare professional.")
        else:
            st.success(f"**Low Risk:** The model predicts a low risk of stroke based on the provided inputs.")

        st.markdown(
            f"""
            <div style="font-size: small; margin-top: 20px;">
            The prediction of **{chance_of_stroke * 100:.2f}%** is based on the 
            K-Nearest Neighbors model, which calculates how similar this patient's 
            profile is to patients in the dataset who experienced a stroke.
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col_chart:
        # --- NEW: Create and display the Pie Chart using Matplotlib ---
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Custom colors for a cleaner look
        colors = ['#4c9eff', '#ff6f69', '#ffcc5c', '#50c878']
        
        # Explode the largest slice for visual emphasis
        explode = [0.05 if v == max(pie_values) else 0 for v in pie_values]
        
        ax.pie(
            pie_values, 
            explode=explode, 
            labels=pie_labels, 
            autopct='%1.1f%%', # Show percentages on slices
            shadow=True, 
            startangle=90, 
            colors=colors,
            textprops={'fontsize': 10}
        )
        
        ax.axis('equal')  # Ensures the pie chart is circular
        ax.set_title("Heuristic Risk Factor Contribution", fontsize=12)
        
        # Use st.pyplot to display the matplotlib figure
        st.pyplot(fig)
        # --- END NEW MATPLOTLIB CODE ---

# --- 6. NOTES ---
st.sidebar.markdown("### ℹ️ Model Details")
st.sidebar.markdown("""
This application uses a K-Nearest Neighbors (KNN) model, trained on the provided dataset with the optimal `n_neighbors=71`.

**Important Note on Factor Contribution:**
The KNN model does not provide direct feature importance. The "Heuristic Risk Factor Contribution" in the pie chart is calculated by comparing the patient's inputs (Age, Glucose, BMI, Heart/HyperTension) to established medical risk thresholds, providing an interpretable visualization of which factors drive the risk.
""")

# Optional: Show the data mappings used
if st.sidebar.checkbox("Show Feature Encodings"):
    st.sidebar.json({k: v['mapping'] for k, v in encoders.items()})
