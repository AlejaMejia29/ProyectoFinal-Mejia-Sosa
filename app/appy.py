import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Cargar modelo, scaler y columnas
# ==============================
model = joblib.load("../models/final_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
saved_columns = joblib.load("../models/columns.pkl")

# ==============================
# TÍTULO Y EXPLICACIÓN EJECUTIVA
# ==============================
st.title("Predicción de Promoción de Empleados – HR Analytics")

st.markdown("""
Esta aplicación permite estimar la probabilidad de que un empleado sea promovido, 
utilizando un modelo de Machine Learning entrenado con información histórica de HR.

El objetivo es apoyar decisiones de talento respecto a:
- Identificación de empleados con alto potencial
- Optimización de programas de desarrollo
- Reducción de rotación en áreas críticas
- Uso de criterios objetivos basados en datos

Las variables más influyentes en el modelo fueron:
- previous_year_rating
- avg_training_score
- KPIs_met >80%
- awards_won?
- length_of_service

Ingrese los datos del empleado para obtener la predicción.
""")


# ==============================
# Inputs del usuario
# ==============================
age = st.number_input("Edad", min_value=18, max_value=60, value=30)
no_of_trainings = st.number_input("Número de entrenamientos", min_value=0, max_value=10, value=1)
previous_year_rating = st.selectbox("Calificación año previo", [1,2,3,4,5])
length_of_service = st.number_input("Años de servicio", min_value=1, max_value=40, value=5)
avg_training_score = st.slider("Puntaje promedio de entrenamiento", 0, 100, 60)

department = st.selectbox(
    "Departamento", 
    [
        "Finance", 
        "HR",
        "Legal",
        "Operations",
        "Procurement", 
        "R&D",
        "Sales & Marketing",
        "Technology"
    ]
)

region = st.selectbox(
    "Región", 
    [f"region_{i}" for i in range(2, 35)]
)

education = st.selectbox(
    "Nivel educativo",
    ["Bachelor", "Below Secondary", "Master's & above"]
)

gender = st.radio("Género", ["m", "f"])

recruitment_channel = st.selectbox(
    "Canal de reclutamiento", 
    ["other", "referred", "sourcing"]
)

KPIs_met = st.radio("KPI mayor al 80%", [0, 1])
awards_won = st.radio("¿Ganó un premio?", [0, 1])


# ==============================
# Construcción del DataFrame base
# ==============================
df_input = pd.DataFrame([{
    "age": age,
    "no_of_trainings": no_of_trainings,
    "previous_year_rating": previous_year_rating,
    "length_of_service": length_of_service,
    "avg_training_score": avg_training_score,
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel,
    "KPIs_met >80%": KPIs_met,
    "awards_won?": awards_won
}])


# ==============================
# Aplicar One-Hot Encoding
# ==============================
df_input = pd.get_dummies(df_input)

# Asegurar columnas completas en el mismo orden
for col in saved_columns:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[saved_columns]


# ==============================
# Escalado de numéricas
# ==============================
num_cols = ["age", "no_of_trainings", "previous_year_rating",
            "length_of_service", "avg_training_score"]

df_input[num_cols] = scaler.transform(df_input[num_cols])


# ==============================
# PREDICCIÓN
# ==============================
if st.button("Predecir"):
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    st.subheader("Resultado de la Predicción")

    if prediction == 1:
        st.write("El empleado tiene ALTA probabilidad de ser promovido.")
    else:
        st.write("El empleado NO sería promovido según el modelo.")

    st.write(f"Probabilidad estimada: {prob:.2f}")
