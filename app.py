import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Definir los nombres de los features como en el modelo entrenado
nombres_features = ['edad', 'colesterol']

# Título de la aplicación
st.title("Asistente IA para cardiólogos")

# Introducción
st.write("""
    Esta aplicación permite predecir si una persona tiene o no un problema cardíaco basado en la edad y el nivel de colesterol.
    El modelo fue entrenado utilizando el algoritmo KNN (K-Nearest Neighbors) de Scikit-Learn y los datos fueron normalizados 
    utilizando MinMax Scaler. La predicción muestra si la persona tiene problemas cardíacos (1) o no (0).
""")

# Crear los tabs
tab_datos = st.radio("Selecciona una opción:", ("Ingresar datos", "Ver predicción"))

# Ingresar datos en el tab de datos
if tab_datos == "Ingresar datos":
    st.header("Por favor, ingresa los siguientes datos:")

    edad = st.number_input("Edad (18-80 años):", min_value=18, max_value=80)
    colesterol = st.number_input("Colesterol (50-600):", min_value=50, max_value=600)

    # Crear un diccionario con los datos de entrada
    datos_entrada = {'edad': edad, 'colesterol': colesterol}
    df_entrada = pd.DataFrame([datos_entrada], columns=nombres_features)

    # Guardar los datos para la predicción más tarde
    if st.button("Guardar datos"):
        st.session_state.datos_entrada = df_entrada
        st.success("Datos guardados correctamente. Ahora puedes ver la predicción.")
        

# Predicción en el tab de predicción
if tab_datos == "Ver predicción":
    if 'datos_entrada' in st.session_state:
        df_entrada = st.session_state.datos_entrada
        
        # Normalizar los datos
        entrada_normalizada = escalador.transform(df_entrada)

        # Realizar la predicción
        prediccion = modelo_knn.predict(entrada_normalizada)[0]

        # Mostrar el resultado de la predicción
        st.subheader("Resultado de la predicción:")

        if prediccion == 1:
            st.write("La persona **tiene problema cardíaco**.")
            st.image("https://www.clikisalud.net/wp-content/uploads/2018/09/problemas-cardiacos-jovenes.jpg", caption="Problema Cardíaco")
        else:
            st.write("La persona **no tiene problema cardíaco**.")
            st.image("https://hospitalcmq.com/wp-content/uploads/2022/12/corazon_saludable.jpg", caption="Corazón Saludable")
    else:
        st.warning("Primero guarda los datos ingresando la edad y el colesterol en la sección 'Ingresar datos'.")
