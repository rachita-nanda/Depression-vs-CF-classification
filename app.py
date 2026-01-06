import numpy as np
import pandas as pd
import pickle
import streamlit as st


log_model = pickle.load(open("logistic_model.pkl", "rb"))
dt_model  = pickle.load(open("decision_tree_model.pkl", "rb"))
rf_model  = pickle.load(open("random_forest_model.pkl", "rb"))

# Load feature order used during training
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.set_page_config(page_title="ME/CFS & Depression Prediction", layout="wide")

st.title(" ME/CFS & Depression Prediction System")
st.write("Fill in the details below to predict the diagnosis.")


model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"]
)


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sq_index = st.number_input("Sleep Quality Index", 1.0, 10.0, 5.0)
    bf_level = st.number_input("Brain Fog Level", 0.0, 10.0, 5.0)
    pps_score = st.number_input("Physical Pain Score", 0.0, 10.0, 5.0)
    stress_level = st.number_input("Stress Level", 0.0, 10.0, 5.0)
    dep_phq9 = st.number_input("PHQ-9 Depression Score", 0, 27, 10)

with col2:
    fs_scale = st.number_input("Fatigue Severity Scale", 0.0, 10.0, 6.0)
    pem_dur = st.number_input("PEM Duration (hours)", 0, 47, 12)
    sleep_hrs = st.number_input("Hours of Sleep", 3.0, 10.0, 7.0)
    pem_present = st.selectbox("PEM Present", ["Yes", "No"])
    med = st.selectbox("Meditation / Mindfulness", ["Yes", "No"])
    work_status = st.selectbox("Work Status", ["Partially working", "Working", "Not working"])
    social_level = st.selectbox(
        "Social Activity Level",
        ["Very low", "Low", "Medium", "High", "Very high"]
    )
    ex_freq = st.selectbox(
        "Exercise Frequency",
        ["Never", "Rarely", "Sometimes", "Often"]
    )


gen_m = 1 if gender == "Male" else 0
pem_y = 1 if pem_present == "Yes" else 0
med_y = 1 if med == "Yes" else 0

ws_pw = 1 if work_status == "Partially working" else 0
ws_w  = 1 if work_status == "Working" else 0

sl_l  = 1 if social_level == "Low" else 0
sl_m  = 1 if social_level == "Medium" else 0
sl_h  = 1 if social_level == "High" else 0
sl_vh = 1 if social_level == "Very high" else 0
sl_vl = 1 if social_level == "Very low" else 0

ex_n  = 1 if ex_freq == "Never" else 0
ex_r  = 1 if ex_freq == "Rarely" else 0
ex_s  = 1 if ex_freq == "Sometimes" else 0
ex_of = 1 if ex_freq == "Often" else 0


input_data = {
    "age": age,
    "gender_Male": gen_m,
    "sleep_quality_index": sq_index,
    "brain_fog_level": bf_level,
    "physical_pain_score": pps_score,
    "stress_level": stress_level,
    "depression_phq9_score": dep_phq9,
    "fatigue_severity_scale_score": fs_scale,
    "pem_duration_hours": pem_dur,
    "hours_of_sleep_per_night": sleep_hrs,
    "pem_present": pem_y,
    "meditation_or_mindfulness": med_y,
    "work_status_Partially working": ws_pw,
    "work_status_Working": ws_w,
    "social_activity_level_Low": sl_l,
    "social_activity_level_Medium": sl_m,
    "social_activity_level_High": sl_h,
    "social_activity_level_Very high": sl_vh,
    "social_activity_level_Very low": sl_vl,
    "exercise_frequency_Never": ex_n,
    "exercise_frequency_Rarely": ex_r,
    "exercise_frequency_Sometimes": ex_s,
    "exercise_frequency_Often": ex_of
}


input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

if model_choice == "Logistic Regression":
    model = log_model
elif model_choice == "Decision Tree":
    model = dt_model
else:
    model = rf_model


if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success(" **Prediction: Depression**")
    elif prediction == 1:
        st.success(" **Prediction: ME/CFS**")
    else:
        st.success(" **Prediction: Both Conditions**")
