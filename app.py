import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("🩺 Diabetes Prediction & Analytics System")
st.write("Enter your health details to check diabetes risk.")

# Input form
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 100)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prediction
if st.button("Predict"):
    user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    prob = model.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes ({prob:.2f} probability)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({prob:.2f} probability)")

# =====================
# Analytics Section
# =====================
st.subheader("📊 Dataset Insights")

# Load dataset again
df = pd.read_csv("diabetes.csv")

fig1 = px.histogram(df, x="Age", color="Outcome", nbins=30, title="Age vs Outcome")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(df, x="Outcome", y="Glucose", title="Glucose Levels by Outcome")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x="BMI", y="Glucose", color="Outcome", title="BMI vs Glucose by Outcome")
st.plotly_chart(fig3, use_container_width=True)
