import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("mental_health_model.pkl", "rb"))

# App Title and Layout
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

st.markdown("""
    <h2 style='text-align: center; color: #4CAF50;'>🧠 Student Mental Health Risk Predictor</h2>
    <p style='text-align: center;'>Predict your mental health risk based on lifestyle and academic habits.</p>
    <hr style='border: 1px solid #ddd;' />
""", unsafe_allow_html=True)

# Input fields with columns
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours (per day)", 0.0, 12.0, 6.5, 0.5)
    screen = st.slider("📱 Screen Time (hours/day)", 0.0, 12.0, 5.0, 0.5)
    stress = st.slider("💢 Stress Level", 1, 5, 3)

with col2:
    social = st.slider("🎯 Social Activity (events/week)", 0, 10, 3)
    grade = st.selectbox("🎓 Your Grade", ["A", "B", "C", "D", "F"])

# Mapping grade
grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
grade_encoded = grade_map[grade]

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Predict Mental Health Risk"):
    input_data = np.array([[sleep, screen, stress, social, grade_encoded]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown("""
            <div style='
                background-color:#ffe6e6;
                padding:20px;
                border-radius:10px;
                border-left:5px solid red;'>
                <h4 style='color:#cc0000;'>⚠️ High Mental Health Risk Detected!</h4>
                <p>Please consider reaching out to a counselor or mental health professional.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='
                background-color:#e6ffe6;
                padding:20px;
                border-radius:10px;
                border-left:5px solid green;'>
                <h4 style='color:#006600;'>✅ Low Mental Health Risk</h4>
                <p>You're doing well! Keep maintaining a healthy balance of habits.</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <br><hr>
    <p style='text-align: center; font-size: 13px; color: #888;'>Made By ❤️ Narasimha Manam</p>
""", unsafe_allow_html=True)
