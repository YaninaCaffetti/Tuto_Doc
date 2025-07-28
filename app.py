# app.py 

import streamlit as st
import yaml
import sys
import os
import pandas as pd
import joblib
import torch
import traceback

# Añadir la carpeta src al path para poder importar nuestros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.emotion_classifier import EmotionClassifier
from src.cognitive_tutor import MoESystem
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Funciones de Carga (Cacheadas por Streamlit) ---
@st.cache_resource
def load_all_models_and_data(config_path='config.yaml'):
    """
    Carga todos los modelos y datos pre-entrenados desde el disco.
    Esta función se cachea para que solo se ejecute una vez por sesión.
    """
    with st.spinner("Cargando modelos y preparando el sistema... Esto puede tardar un momento."):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Cargar el clasificador de emociones
        emotion_model_path = config['model_paths']['emotion_classifier']
        if not os.path.exists(emotion_model_path):
            st.error(f"Error Crítico: No se encuentra el modelo de emociones en '{emotion_model_path}'. Por favor, ejecute 'train.py' primero.")
            st.stop()
        model_emo = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
        tokenizer_emo = AutoTokenizer.from_pretrained(emotion_model_path)
        emotion_classifier = EmotionClassifier(model_emo, tokenizer_emo)

        # Cargar el tutor cognitivo
        cognitive_model_path = config['model_paths']['cognitive_tutor']
        if not os.path.exists(cognitive_model_path):
            st.error(f"Error Crítico: No se encuentra el modelo de tutor en '{cognitive_model_path}'.")
            st.stop()
        cognitive_model = joblib.load(cognitive_model_path)
        
        # Cargar los perfiles de demo
        demo_profiles_path = config['data_paths']['demo_profiles']
        if not os.path.exists(demo_profiles_path):
            st.error(f"Error Crítico: No se encuentra el archivo de perfiles de demo en '{demo_profiles_path}'.")
            st.stop()
        df_profiles = pd.read_csv(demo_profiles_path, index_col='ID')
        
        feature_columns = [col for col in df_profiles.columns if '_memb' in col]
        cognitive_tutor_system = MoESystem(cognitive_model, feature_columns, config['affective_rules'])
        
        return emotion_classifier, cognitive_tutor_system, df_profiles

# --- Interfaz Principal de la Aplicación ---
def main():
    st.set_page_config(page_title="Tutor Cognitivo Adaptativo", layout="wide", initial_sidebar_state="expanded")

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.header("Sobre el Proyecto")
        st.markdown("""
        Esta aplicación es una demostración del prototipo desarrollado en la tesis doctoral **"Diseño de un proceso para la integración de un tutor cognitivo..."** de la **Mgter. Ing. Yanina A. Caffetti**.
        
        El sistema combina dos componentes de IA:
        1.  **Un Tutor Cognitivo** que asigna un arquetipo a un perfil de usuario.
        2.  **Un Clasificador de Emociones** que analiza el texto del usuario.
        
        Juntos, generan un plan de acción que se adapta tanto al perfil como al estado emocional del usuario.
        """)
        st.header("Instrucciones")
        st.markdown("""
        1.  **Seleccione un ID de usuario** de prueba. Cada ID representa un perfil diferente.
        2.  **Escriba una consulta** en el área de texto o elija una de los ejemplos.
        3.  Haga clic en **"Generar Respuesta"** para ver el análisis y el plan de acción adaptativo.
        """)

    # --- CONTENIDO PRINCIPAL ---
    st.title("🧠 Tutor Cognitivo Adaptativo con IA Afectiva 🤖")
    st.markdown("---")

    # Cargar modelos y datos
    emotion_classifier, cognitive_tutor_system, df_profiles = load_all_models_and_data()

    st.subheader("Simulador de Interacción")

    # Selección de perfil y entrada de texto en columnas para mejor layout
    col1, col2 = st.columns(2)
    with col1:
        user_ids = df_profiles.index.tolist()
        selected_id = st.selectbox("1. Seleccione un Perfil de Usuario de Prueba:", user_ids)
    
    with col2:
        example_queries = [
            "¡Es una vergüenza, llevo meses esperando y no me dan respuesta!",
            "No entiendo bien qué es la ley 22.431 pero gracias por la info, me da esperanza.",
            "Estoy muy ansioso por la entrevista de la próxima semana.",
            "El resultado fue decepcionante, no sé qué hacer ahora."
        ]
        selected_query = st.selectbox("2. (Opcional) Elija una consulta de ejemplo:", [""] + example_queries, index=0)

    # El área de texto se actualiza si se selecciona un ejemplo
    if selected_query:
        user_input = st.text_area("3. Escriba o modifique la consulta del usuario aquí:", value=selected_query, height=100)
    else:
        user_input = st.text_area("3. Escriba o modifique la consulta del usuario aquí:", value="", height=100, placeholder="Escriba aquí...")

    if st.button("Generar Respuesta del Tutor", type="primary"):
        if not user_input or not selected_id:
            st.warning("Por favor, asegúrese de que haya texto en la consulta y un ID de usuario seleccionado.")
        else:
            with st.spinner("Procesando... El tutor está analizando la consulta..."):
                try:
                    # Generar la respuesta completa
                    emotion_probs = emotion_classifier.predict_proba(user_input)
                    top_emotion = max(emotion_probs, key=emotion_probs.get)
                    user_profile = df_profiles.loc[selected_id]
                    cognitive_plan = cognitive_tutor_system.get_cognitive_plan(user_profile, emotion_probs)

                    # Construir el mensaje de introducción
                    if top_emotion in ["Ira", "Tristeza", "Miedo"]: 
                        intro_message = f"Percibo que puedes sentirte con un poco de **{top_emotion.lower()}**. Revisemos esto juntos para encontrar una solución. Aquí tienes un plan de acción:"
                    elif top_emotion in ["Anticipación", "Alegría", "Confianza"]: 
                        intro_message = f"¡Excelente! Percibo un estado de **{top_emotion.lower()}**. Para potenciar ese impulso, este es el plan que te sugiero:"
                    else: 
                        intro_message = "Entendido. En base a tu consulta, este es el plan de acción sugerido:"

                    # Mostrar los resultados
                    st.markdown("---")
                    st.subheader("Análisis y Plan de Acción Generado")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.info(f"**🧠 Emoción Dominante:** {top_emotion} ({emotion_probs[top_emotion]:.0%})")
                        st.write("**Espectro Emocional Completo:**")
                        df_probs = pd.DataFrame(emotion_probs.items(), columns=['Emoción', 'Confianza'])
                        df_probs = df_probs[df_probs['Confianza'] > 0.01].sort_values(by='Confianza', ascending=False)
                        st.dataframe(df_probs, use_container_width=True, hide_index=True)
                    
                    with res_col2:
                        st.markdown("##### ✨ **Respuesta Integrada y Afectiva:**")
                        st.markdown(intro_message)
                        st.markdown(cognitive_plan.replace('  - ', '* '))

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la respuesta: {e}")
                    st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
