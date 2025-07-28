# app.py (Versión 26.0 - Módulo de Servicio/Demo)

import streamlit as st
import yaml
import sys
import os
import pandas as pd
import joblib

# Añadir la carpeta src al path para poder importar nuestros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.emotion_classifier import EmotionClassifier
from src.cognitive_tutor import MoESystem
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_all_models_and_data(config):
    """Carga todos los modelos y datos pre-entrenados desde el disco."""
    
    # Cargar el clasificador de emociones
    emotion_model_path = config['model_paths']['emotion_classifier']
    if not os.path.exists(emotion_model_path):
        st.error(f"Error: No se encuentra el modelo de emociones en '{emotion_model_path}'. Por favor, ejecute 'main.py' primero para entrenar y guardar los modelos.")
        st.stop()
    model_emo = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
    tokenizer_emo = AutoTokenizer.from_pretrained(emotion_model_path)
    emotion_classifier = EmotionClassifier(model_emo, tokenizer_emo)

    # Cargar el tutor cognitivo
    cognitive_model_path = config['model_paths']['cognitive_tutor']
    if not os.path.exists(cognitive_model_path):
        st.error(f"Error: No se encuentra el modelo de tutor en '{cognitive_model_path}'. Por favor, ejecute 'main.py' primero para entrenar y guardar los modelos.")
        st.stop()
    cognitive_model = joblib.load(cognitive_model_path)
    
    # Cargar los perfiles de demo
    demo_profiles_path = config['data_paths']['demo_profiles']
    if not os.path.exists(demo_profiles_path):
        st.error(f"Error: No se encuentra el archivo de perfiles de demo en '{demo_profiles_path}'. Por favor, ejecute 'main.py' primero.")
        st.stop()
    df_profiles = pd.read_csv(demo_profiles_path, index_col='ID')
    
    feature_columns = [col for col in df_profiles.columns if '_memb' in col]
    cognitive_tutor_system = MoESystem(cognitive_model, feature_columns, config['affective_rules'])
    
    return emotion_classifier, cognitive_tutor_system, df_profiles

def main():
    st.set_page_config(page_title="Tutor Cognitivo Adaptativo", layout="wide")

    st.title("🧠 Tutor Cognitivo Adaptativo con IA Afectiva 🤖")
    st.markdown("*Proyecto de Tesis Doctoral de Mgter. Ing. Yanina A. Caffetti*")
    st.markdown("---")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    emotion_classifier, cognitive_tutor_system, df_profiles = load_all_models_and_data(config)
    st.success("✅ ¡Sistema y modelos cargados exitosamente!")

    st.subheader("Simulador de Interacción con el Tutor")
    user_ids = df_profiles.index.tolist()
    selected_id = st.selectbox("Seleccione un ID de Usuario para la demostración:", user_ids)
    user_input = st.text_area("Escriba la consulta del usuario aquí:", "No entiendo bien qué es la ley 22.431 pero gracias por la info, me da esperanza.", height=100)

    if st.button("Generar Respuesta del Tutor"):
        if user_input and selected_id:
            with st.spinner("Procesando..."):
                emotion_probs = emotion_classifier.predict_proba(user_input)
                top_emotion = max(emotion_probs, key=emotion_probs.get)
                user_profile = df_profiles.loc[selected_id]
                cognitive_plan = cognitive_tutor_system.get_cognitive_plan(user_profile, emotion_probs)

                if top_emotion in ["Ira", "Tristeza", "Miedo"]: intro_message = f"Percibo que puedes sentirte con un poco de {top_emotion.lower()}. Revisemos esto juntos para encontrar una solución. Aquí tienes un plan de acción:"
                elif top_emotion in ["Anticipación", "Alegría", "Confianza"]: intro_message = f"¡Excelente! Percibo un estado de {top_emotion.lower()}. Para potenciar ese impulso, este es el plan que te sugiero:"
                else: intro_message = "Entendido. En base a tu consulta, este es el plan de acción sugerido:"

                st.markdown("---")
                st.subheader("Análisis y Plan de Acción Generado")
                st.info(f"**🧠 Emoción Dominante Detectada:** {top_emotion} (Confianza: {emotion_probs[top_emotion]:.0%})")
                
                st.markdown("---")
                st.markdown("##### ✨ **Respuesta Integrada y Afectiva:**")
                st.markdown(intro_message)
                st.markdown(cognitive_plan.replace('  - ', '* '))
        else:
            st.warning("Por favor, ingrese un texto y seleccione un ID de usuario.")

if __name__ == '__main__':
    main()
