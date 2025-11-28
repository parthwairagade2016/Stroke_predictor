import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import plotly.express as px # Used for interactive pie chart
import os

# --- 1. CONFIGURATION AND INITIAL SETUP ---
st.set_page_config(
    page_title="🩺 Stroke Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Define the expected features and their data types
FEATURE_COLUMNS = [
    'age', 'hypertension', 'heart_disease', 'ever_married', 
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 
    'smoking_status'
]

# --- 2. DATA LOADING AND PREPROCESSING (Cached for Performance) ---

@st.cache_data(show_spinner="Loading and Training Model...")
def load_and_preprocess_data():
    """Loads the dataset, imputes missing data, performs scaling, encoding, and training."""
    try:
        # Check for common file names
        file_path = 'Strokes.csv'
        if not os.path.exists(file_path):
             file_path = 'stroke.csv' 
        
        if os.path.exists(file_path):
             df = pd.read_csv(file_path)
        else:
            # Fallback to synthetic data if file not found
            df = pd.DataFrame({
                'id': range(100), 'gender': ['Male'] * 100, 'age': np.random.randint(18, 80, 100), 
                'hypertension': np.random.randint(0, 2, 100), 'heart_disease': np.random.randint(0, 2, 100),
                'ever_married': ['Yes'] * 100, 'work_type': ['Private'] * 100, 
                'Residence_type': ['Urban'] * 100, 'avg_glucose_level': np.random.rand(100) * 150 + 70, 
                'bmi': np.random.rand(100) * 20 + 20, 'smoking_status': ['never smoked'] * 100, 
                'stroke': np.random.randint(0, 2, 100)
            })
            st.warning("Warning: Data file not found. Using synthetic dataset for structure.")
            
    except Exception as e:
        st.error(f"Error loading data: {e}. Using synthetic data.")
        df = pd.DataFrame({
            'id': range(100), 'gender': ['Male'] * 100, 'age': np.random.randint(18, 80, 100), 
            'hypertension': np.random.randint(0, 2, 100), 'heart_disease': np.random.randint(0, 2, 100),
            'ever_married': ['Yes'] * 100, 'work_type': ['Private'] * 100, 
            'Residence_type': ['Urban'] * 100, 'avg_glucose_level': np.random.rand(100) * 150 + 70, 
            'bmi': np.random.rand(100) * 20 + 20, 'smoking_status': ['never smoked'] * 100, 
            'stroke': np.random.randint(0, 2, 100)
        })

    # Data Cleaning and Imputation
    df.drop(["id", "gender"], axis=1, inplace=True, errors='ignore')
    
    # Impute missing BMI with the median for robust performance
    median_bmi = df['bmi'].median() if df['bmi'].notna().any() else 25.0
    df['bmi'].fillna(median_bmi, inplace=True)

    # Label Encoding for categorical features
    encoders = {}
    for col in ['work_type', 'Residence_type', 'ever_married', 'smoking_status']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = {
                'encoder': le, 
                'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
            }

    # Feature Scaling (Crucial for KNN)
    X = df.drop("stroke", axis=1).copy()
    y = df['stroke'].values
    
    scaler = StandardScaler()
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # --- NON-LINEAR TRANSFORMATION (Accuracy Enhancement for high-risk values) ---
    
    # Non-linear penalty for high Glucose: increases magnitude for values > 0.5 standard deviation (approx 120 mg/dL)
    X['avg_glucose_level'] = X['avg_glucose_level'].apply(lambda x: x * 1.0 + (max(0, x - 0.5)**2) if x > 0.5 else x)

    # Non-linear penalty for high BMI: increases magnitude for values > 1.5 standard deviations
    X['bmi'] = X['bmi'].apply(lambda x: x * 1.0 + (max(0, x - 1.5)**2) if x > 1.5 else x)
    
    # Custom Weighting for Binary/Categorical High-Risk Features
    # HIGHLY INCREASED WEIGHT for primary medical conditions
    X['hypertension'] = X['hypertension'].astype(float) * 4.0 # Increased from 3.0 to 4.0
    X['heart_disease'] = X['heart_disease'].astype(float) * 4.0 # Increased from 3.0 to 4.0
    X['smoking_status'] = X['smoking_status'].astype(float) * 2.5 
    
    # Weighting for Ever Married 
    X['ever_married'] = X['ever_married'].astype(float) * 1.5 
    
    # Weighting for Residence Type 
    X['Residence_type'] = X['Residence_type'].astype(float) * 1.5

    # --- MODEL TRAINING ---
    
    # Define hyperparameter grid 
    tuned_parameters = {
        'n_neighbors': [5, 11, 21, 41, 71, 101],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=tuned_parameters,
        cv=6, 
        scoring='accuracy', 
        n_jobs=-1
    )

    grid_search.fit(X.values, y)
    model = grid_search.best_estimator_
    
    return X.values, y, encoders, scaler, model 

# Load data, get encodings, scaler, and train the optimized model
X, y, encoders, scaler, model = load_and_preprocess_data()


# --- 3. INPUT WIDGETS AND INTERACTIVE LAYOUT ---

st.title("🩺 Stroke Risk Predictor")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name", value="John Doe")
    
    age = st.slider("Age", 0, 100, 50, help="Patient's age in years.")
    
    hypertension = st.selectbox(
        "Hypertension (High Blood Pressure)", 
        options=['No', 'Yes'], index=0,
        format_func=lambda x: 'Yes' if x == 'Yes' else 'No',
        help="Patient has hypertension (1=Yes, 0=No). CRITICAL RISK FACTOR."
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
        help="Has the patient ever been married? (Higher risk if 'Yes')"
    )

with col2:
    # Use the mapping from the loaded data for categorical features
    work_type_options = list(encoders['work_type']['mapping'].keys()) if 'work_type' in encoders else ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    work_type = st.selectbox("Work Type", options=work_type_options, index=0)
    
    Residence_type_options = list(encoders['Residence_type']['mapping'].keys()) if 'Residence_type' in encoders else ['Urban', 'Rural']
    Residence_type = st.selectbox("Residence Type", options=Residence_type_options, index=0, help="Urban residence is often associated with specific lifestyle factors.")
    
    avg_glucose_level = st.number_input(
        "Average Glucose Level (mg/dL)", 
        min_value=50.0, max_value=300.0, value=95.0, step=0.1,
        help="Fasting plasma glucose test results. High values drastically increase risk."
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)", 
        min_value=10.0, max_value=60.0, value=25.0, step=0.1,
        help="Calculated as weight (kg) / height (m)^2. Obesity is a major risk factor."
    )
    
    smoking_status_options = list(encoders['smoking_status']['mapping'].keys()) if 'smoking_status' in encoders else ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
    smoking_status = st.selectbox("Smoking Status", options=smoking_status_options, index=0)

# --- 4. PREDICTION LOGIC ---

if st.button("Predict Stroke Risk", type="primary"):
    
    # 1. Convert user inputs into the numerical vector expected by the model
    hypertension_val = 1 if hypertension == 'Yes' else 0
    heart_disease_val = 1 if heart_disease == 'Yes' else 0
    ever_married_val = encoders['ever_married']['mapping'].get(ever_married, 0) if 'ever_married' in encoders else (1 if ever_married == 'Yes' else 0)
    residence_type_val = encoders['Residence_type']['mapping'].get(Residence_type, 0) if 'Residence_type' in encoders else 0
    work_type_val = encoders['work_type']['mapping'].get(work_type, 0) if 'work_type' in encoders else 0
    smoking_status_val = encoders['smoking_status']['mapping'].get(smoking_status, 0) if 'smoking_status' in encoders else 0
    
    raw_input_data = np.array([
        age, hypertension_val, heart_disease_val, ever_married_val, 
        work_type_val, residence_type_val, avg_glucose_level, bmi, 
        smoking_status_val
    ]).reshape(1, -1)

    # Apply scaling and custom weighting to input data
    input_df = pd.DataFrame(raw_input_data, columns=FEATURE_COLUMNS)
    
    # Scale numerical features using the trained scaler
    input_df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(input_df[['age', 'avg_glucose_level', 'bmi']])
    
    # --- APPLY NON-LINEAR AND CUSTOM WEIGHTING TO INPUT DATA (MUST MATCH TRAINING) ---
    
    # Non-linear penalty for high Glucose (RISK STARTS EARLIER)
    input_df['avg_glucose_level'] = input_df['avg_glucose_level'].apply(lambda x: x * 1.0 + (max(0, x - 0.5)**2) if x > 0.5 else x)
    
    # Non-linear penalty for high BMI
    input_df['bmi'] = input_df['bmi'].apply(lambda x: x * 1.0 + (max(0, x - 1.5)**2) if x > 1.5 else x)
    
    # Custom weighting for binary/categorical features (MATCHING TRAINING WEIGHTS)
    input_df['hypertension'] = input_df['hypertension'] * 4.0 # CRITICAL WEIGHT APPLIED
    input_df['heart_disease'] = input_df['heart_disease'] * 4.0 # CRITICAL WEIGHT APPLIED
    input_df['smoking_status'] = input_df['smoking_status'] * 2.5
    
    # WEIGHTS FOR EVER MARRIED AND RESIDENCE TYPE
    input_df['ever_married'] = input_df['ever_married'] * 1.5
    input_df['Residence_type'] = input_df['Residence_type'] * 1.5
    
    input_data = input_df.values
    
    # 2. Make Prediction and Get Probability
    probabilities = model.predict_proba(input_data)[0]
    chance_of_stroke = probabilities[1] # Probability of class 1 (stroke)
    
    # 3. Calculate Heuristic Risk Contribution for Pie Chart (Custom Logic)
    risk_weights = {
        "Age Risk": 10,  
        "Glucose Risk": 30, 
        "BMI Risk": 20, 
        "Heart/HyperTension": 30, # High base weight
        "Lifestyle/Smoking": 10 
    }
    
    # Calculate individual risk scores (0-100) based on critical clinical thresholds
    age_risk_score = min(100, max(0, (age - 45) / (70 - 45) * 100)) 
    glucose_risk_score = min(100, max(0, (avg_glucose_level - 120) / (223 - 120) * 100)) 
    bmi_risk_score = min(100, max(0, (bmi - 25) / (35 - 25) * 100)) 
    # Emphasize hypertension (0.7) over heart disease (0.3) in the composite score
    heart_hyper_risk_score = (hypertension_val * 0.7 + heart_disease_val * 0.3) * 100 
    smoking_risk_map = {0: 0, 1: 75, 2: 100, 3: 50}
    smoking_risk_score = smoking_risk_map.get(smoking_status_val, 0)
    
    contributions = {
        "Age Risk": age_risk_score * risk_weights["Age Risk"] / 100,
        "Glucose Risk": glucose_risk_score * risk_weights["Glucose Risk"] / 100,
        "BMI Risk": bmi_risk_score * risk_weights["BMI Risk"] / 100,
        "Heart/HyperTension": heart_hyper_risk_score * risk_weights["Heart/HyperTension"] / 100,
        "Lifestyle/Smoking": smoking_risk_score * risk_weights["Lifestyle/Smoking"] / 100
    }
    
    total_score = sum(contributions.values())
    if total_score > 0:
        contribution_percentages = {k: (v / total_score) * 100 for k, v in contributions.items()}
    else:
        contribution_percentages = {k: 100 / len(contributions) for k in contributions.keys()}
        
    pie_data = pd.DataFrame(list(contribution_percentages.items()), columns=['Factor', 'Contribution'])
    
    
    # --- 5. DISPLAY RESULTS ---
    st.subheader(f"Results for Patient: {patient_name}")
    
    col_pred, col_chart = st.columns([1, 1])

    with col_pred:
        # Display the prediction and percentage
        st.metric(
            label="Predicted Stroke Risk", 
            value=f"{chance_of_stroke * 100:.2f}%", 
            delta=None 
        )
        
        st.markdown("---")
        
        # --- CUSTOMIZED RISK INTERPRETATION LOGIC ---
        risk_percent = chance_of_stroke * 100
        
        if risk_percent >= 40.0:
            st.error(f"**HIGH RISK ({risk_percent:.2f}%):** Immediate consultation with a medical professional is strongly recommended.")
        elif risk_percent >= 35.0:
            st.warning(f"**RISKY ({risk_percent:.2f}%):** The potential risk is significant and requires urgent attention.")
        elif risk_percent >= 25.0:
            st.warning(f"**MILD RISKY ({risk_percent:.2f}%):** These chances are concerning; medical review is advised.")
        elif risk_percent >= 15.0:
            st.info(f"**MODERATE CHANCES ({risk_percent:.2f}%):** There is a moderate potential risk profile. Review your risk factors.")
        else:
            st.success(f"**SLIGHT CHANCES ({risk_percent:.2f}%):** The risk profile is currently low.")
        # --- END CUSTOMIZED RISK INTERPRETATION LOGIC ---

        st.markdown(
            f"""
            <div style="font-size: small; margin-top: 20px;">
            This prediction is based on the optimally tuned model's comparison 
            of the patient's data to similar stroke cases in the dataset.
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col_chart:
        # Using Plotly Express for interactive chart
        fig = px.pie(
            pie_data, 
            values='Contribution', 
            names='Factor', 
            title='Heuristic Risk Factor Contribution',
            height=400,
            color_discrete_sequence=px.colors.sequential.Agsunset,
        )
        # Customizing layout for better aesthetics
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        
        st.plotly_chart(fig, use_container_width=True)
