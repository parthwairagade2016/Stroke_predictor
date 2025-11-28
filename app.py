import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION AND INITIAL SETUP ---
st.set_page_config(
    page_title="Stroke Risk Predictor (Optimized)",
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

@st.cache_data(show_spinner="Loading and Preprocessing Data...")
def load_and_preprocess_data():
    """Loads the dataset, imputes missing data, performs scaling and encoding."""
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
    X = df.drop("stroke", axis=1)
    y = df['stroke'].values
    
    scaler = StandardScaler()
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # --- START OF CUSTOM FEATURE WEIGHTING FOR KNN ---
    # Apply a multiplicative factor to high-risk binary features to increase their impact 
    # on the KNN distance calculation, making the model more sensitive to them.
    high_risk_binary_features = ['hypertension', 'heart_disease', 'smoking_status']
    for feature in high_risk_binary_features:
        # Scale only if the feature exists in the DataFrame's columns
        if feature in X.columns:
            # We use a factor (e.g., 2.0) to increase the magnitude of these features 
            # after standard scaling (which has already been applied to the numericals).
            # We scale all features that are not already scaled (like the binary ones) 
            # and then multiply the high-risk ones by a factor.
            
            # Since hypertension and heart_disease are already 0/1, we just multiply
            # the columns in X.
            feature_index = df.columns.get_loc(feature) # Get the index from the original dataframe columns
            if feature in ['hypertension', 'heart_disease']:
                 # Scale binary features by 2.5 to give them more weight
                X.iloc[:, feature_index] = X.iloc[:, feature_index].astype(float) * 2.5
            elif feature == 'smoking_status':
                 # Smoking is label encoded (0, 1, 2, 3), so we scale it too.
                 X.iloc[:, feature_index] = X.iloc[:, feature_index].astype(float) * 2.0


    # --- END OF CUSTOM FEATURE WEIGHTING ---
    
    return X.values, y, encoders, scaler # Return the fitted scaler

@st.cache_resource(show_spinner="Optimizing and Training KNN Model...")
def get_optimized_model(X, y):
    """Performs GridSearchCV on the scaled data and returns the best-optimized KNN model."""
    
    # Define hyperparameter grid 
    tuned_parameters = {
        'n_neighbors': [5, 11, 21, 41, 71, 101],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    
    # Use GridSearchCV for optimal parameter selection
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=tuned_parameters,
        cv=6, 
        scoring='accuracy', 
        n_jobs=-1
    )

    # Fit GridSearch on the scaled training data
    grid_search.fit(X, y)
    
    # Return the best estimator found
    return grid_search.best_estimator_

# Load data, get encodings, scaler, and train the optimized model
X, y, encoders, scaler = load_and_preprocess_data()
model = get_optimized_model(X, y)


# --- 3. INPUT WIDGETS AND INTERACTIVE LAYOUT ---

st.title("🩺 Stroke Risk Prediction App (Optimized)")
st.markdown("""
<p style='color: green;'>The K-Nearest Neighbors (KNN) model has been optimized for the highest predictive accuracy and is now more sensitive to high-risk factors like **Hypertension**, **Heart Disease**, and **Smoking**.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name", value="John Doe")
    
    age = st.slider("Age", 0, 100, 50, help="Patient's age in years.")
    
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
    hypertension_val = 1 if hypertension == 'Yes' else 0
    heart_disease_val = 1 if heart_disease == 'Yes' else 0
    ever_married_val = encoders['ever_married']['mapping'].get(ever_married, 0) if 'ever_married' in encoders else (1 if ever_married == 'Yes' else 0)
    work_type_val = encoders['work_type']['mapping'].get(work_type, 0) if 'work_type' in encoders else 0
    residence_type_val = encoders['Residence_type']['mapping'].get(Residence_type, 0) if 'Residence_type' in encoders else 0
    smoking_status_val = encoders['smoking_status']['mapping'].get(smoking_status, 0) if 'smoking_status' in encoders else 0
    
    raw_input_data = np.array([
        age, hypertension_val, heart_disease_val, ever_married_val, 
        work_type_val, residence_type_val, avg_glucose_level, bmi, 
        smoking_status_val
    ]).reshape(1, -1)

    # Apply scaling and custom weighting to input data using the fitted scaler
    input_df = pd.DataFrame(raw_input_data, columns=FEATURE_COLUMNS)
    
    # Scale numerical features
    input_df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(input_df[['age', 'avg_glucose_level', 'bmi']])
    
    # Apply the same manual weighting used during training to high-risk binary features
    input_df['hypertension'] = input_df['hypertension'] * 2.5
    input_df['heart_disease'] = input_df['heart_disease'] * 2.5
    input_df['smoking_status'] = input_df['smoking_status'] * 2.0
    
    input_data = input_df.values
    
    # 2. Make Prediction and Get Probability
    probabilities = model.predict_proba(input_data)[0]
    chance_of_stroke = probabilities[1] # Probability of class 1 (stroke)
    
    # 3. Calculate Heuristic Risk Contribution for Pie Chart (Custom Logic)
    # UPDATED WEIGHTS: Shifted weight from Age to Glucose and BMI, and slightly increased Smoking
    risk_weights = {
        "Age Risk": 15, 
        "Glucose Risk": 25, 
        "BMI Risk": 25, 
        "Heart/HyperTension": 25,
        "Lifestyle/Smoking": 10 # Slightly increased to 10
    }
    
    # Calculate individual risk scores (0-100) based on typical thresholds
    age_risk_score = min(100, max(0, (age - 40) / (65 - 40) * 100))
    glucose_risk_score = min(100, max(0, (avg_glucose_level - 100) / (150 - 100) * 100))
    bmi_risk_score = min(100, max(0, (bmi - 25) / (35 - 25) * 100))
    heart_hyper_risk_score = (hypertension_val * 0.5 + heart_disease_val * 0.5) * 100
    smoking_risk_map = {0: 0, 1: 75, 2: 100, 3: 50}
    smoking_risk_score = smoking_risk_map.get(smoking_status_val, 0)
    
    contributions = {
        "Age Risk": age_risk_score * risk_weights["Age Risk"] / 100,
        "Glucose Risk": glucose_risk_score * risk_weights["Glucose Risk"] / 100,
        "BMI Risk": bmi_risk_score * risk_weights["BMI Risk"] / 100,
        "Heart/HyperTension": heart_hyper_risk_score * risk_weights["Heart/HyperTension"] / 100,
        "Lifestyle/Smoking": smoking_risk_score * risk_weights["Lifestyle/Smoking"] / 100
    }
    
    # Normalize contributions to get percentages for the pie chart
    total_score = sum(contributions.values())
    if total_score > 0:
        contribution_percentages = {k: (v / total_score) * 100 for k, v in contributions.items()}
    else:
        # Default distribution if all risks are zero
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
            delta=None 
        )
        
        st.markdown("---")
        
        # --- UPDATED RISK INTERPRETATION LOGIC ---
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
        # --- END UPDATED RISK INTERPRETATION LOGIC ---

        st.markdown(
            f"""
            <div style="font-size: small; margin-top: 20px;">
            The prediction of **{chance_of_stroke * 100:.2f}%** is based on the 
            optimally tuned K-Nearest Neighbors model.
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col_chart:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        colors = ['#4c9eff', '#ff6f69', '#ffcc5c', '#50c878', '#9933cc'] 
        # Highlight the largest contribution
        explode = [0.05 if v == max(pie_values) else 0 for v in pie_values]
        
        ax.pie(
            pie_values, 
            explode=explode, 
            labels=pie_labels, 
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=90, 
            colors=colors[:len(pie_values)], 
            textprops={'fontsize': 10}
        )
        
        ax.axis('equal')  
        ax.set_title("Heuristic Risk Factor Contribution", fontsize=12)
        
        st.pyplot(fig)
