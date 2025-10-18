import streamlit as st
import yaml
import sys
import os
import pandas as pd
import joblib
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE MÓDulos ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.emotion_classifier import EmotionClassifier
    from src.cognitive_tutor import MoESystem
except ImportError as e:
    st.error(f"Error al importar los módulos: {e}")
    st.stop()

# --- 2. FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---
@st.cache_resource
def load_all_models_and_data(config_path: str = 'config.yaml') -> tuple:
    """Carga y prepara todos los artefactos necesarios para la aplicación."""
    with st.spinner("Cargando modelos y preparando el sistema..."):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not config:
                    st.error(f"Error Crítico: El archivo '{config_path}' está vacío o mal formado.")
                    st.stop()
        except FileNotFoundError:
            st.error(f"Error Crítico: No se encuentra '{config_path}'.")
            st.stop()
        
        paths = config.get('model_paths', {})
        for key, path in paths.items():
            if not os.path.exists(path):
                st.error(f"Error Crítico: No se encuentra la ruta para '{key}': '{path}'. Ejecute 'train.py' primero.")
                st.stop()

        emotion_classifier = EmotionClassifier(
            AutoModelForSequenceClassification.from_pretrained(paths['emotion_classifier']),
            AutoTokenizer.from_pretrained(paths['emotion_classifier'])
        )
        cognitive_model = joblib.load(paths['cognitive_tutor'])
        df_profiles = pd.read_csv(config['data_paths']['demo_profiles'], index_col='ID')
        feature_columns = [col for col in df_profiles.columns if '_memb' in col]
        
        thresholds = config.get('system_thresholds', {})
        cognitive_tutor_system = MoESystem(
            cognitive_model, 
            feature_columns, 
            config['affective_rules'],
            thresholds 
        )
        
        return emotion_classifier, cognitive_tutor_system, df_profiles, config

# --- 3. FUNCIONES DE LA INTERFAZ DE USUARIO ---

def get_initial_metrics() -> Dict:
    """Genera la estructura de diccionario para inicializar o reiniciar las métricas."""
    return {
        "total_interactions": 0, "negative_emotion_count": 0, "emotion_counts": {},
        "emotion_confidence_sum": {}, "profile_emotion_counts": {}
    }

def initialize_session_state(df_profiles: pd.DataFrame, config: Dict):
    """Inicializa las variables clave en el `session_state` de Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = get_initial_metrics()
    if 'selected_profile_id' not in st.session_state:
        st.session_state.selected_profile_id = df_profiles.index.tolist()[0]
    if 'config' not in st.session_state:
        st.session_state.config = config

def update_metrics(analysis: Dict):
    """Actualiza las métricas de la sesión con los datos de la última interacción."""
    metrics = st.session_state.metrics
    profile_id = st.session_state.selected_profile_id
    top_emotion = analysis['top_emotion']
    
    constants = st.session_state.config.get('constants', {})
    negative_emotions = constants.get('negative_emotions', [])
    
    metrics["total_interactions"] += 1
    if top_emotion in negative_emotions:
        metrics["negative_emotion_count"] += 1

    metrics["emotion_counts"][top_emotion] = metrics["emotion_counts"].get(top_emotion, 0) + 1
    metrics["emotion_confidence_sum"][top_emotion] = metrics["emotion_confidence_sum"].get(top_emotion, 0) + analysis['top_emotion_prob']

    if profile_id not in metrics["profile_emotion_counts"]:
        metrics["profile_emotion_counts"][profile_id] = {}
    profile_metrics = metrics["profile_emotion_counts"][profile_id]
    profile_metrics[top_emotion] = profile_metrics.get(top_emotion, 0) + 1

def render_sidebar(df_profiles: pd.DataFrame):
    """Crea y muestra todo el contenido de la barra lateral de la aplicación."""
    with st.sidebar:
        st.header("Sobre el Proyecto")
        st.markdown("Demostración del prototipo de tesis de la **Mgter. Ing. Yanina A. Caffetti**.")
        
        st.header("Configuración de la Simulación")
        
        def clear_chat_on_profile_change():
            st.session_state.messages = []
            st.session_state.metrics = get_initial_metrics()

        st.selectbox(
            "Seleccione un Perfil de Usuario:",
            df_profiles.index.tolist(),
            key='selected_profile_id',
            help="Cambiar de perfil reiniciará la conversación.",
            on_change=clear_chat_on_profile_change
        )
        st.info(f"Perfil activo: **{st.session_state.selected_profile_id}**")
        
        if st.button("Reiniciar Conversación Actual"):
            clear_chat_on_profile_change()
            st.rerun()

        st.header("Métricas de la Sesión")
        metrics = st.session_state.metrics
        if metrics["total_interactions"] > 0:
            neg_rate = (metrics["negative_emotion_count"] / metrics["total_interactions"]) * 100
            st.metric(label="Tasa de Emociones Negativas", value=f"{neg_rate:.1f}%")

            st.subheader("Distribución General de Emociones")
            if metrics["emotion_counts"]:
                df_emotion_dist = pd.DataFrame(metrics["emotion_counts"].items(), columns=['Emoción', 'Frecuencia'])
                st.bar_chart(df_emotion_dist.set_index('Emoción'))
        else:
            st.info("Inicie una conversación para ver las métricas.")

def render_chat_interface(emotion_classifier: EmotionClassifier, cognitive_tutor_system: MoESystem, df_profiles: pd.DataFrame):
    """Renderiza la interfaz de chat principal y maneja la lógica de la conversación."""
    st.title("🧠 Tutor Cognitivo Adaptativo con IA Afectiva 🤖")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "analysis" in message:
                with st.expander("Ver Análisis Detallado de esta Respuesta"):
                    analysis = message["analysis"]
                    st.info(f"**👤 Arquetipo Cognitivo:** {analysis['archetype']}")
                    st.info(f"**🧠 Emoción Dominante:** {analysis['top_emotion']} ({analysis['top_emotion_prob']:.0%})")
                    df_probs = pd.DataFrame(analysis['emotion_probs'].items(), columns=['Emoción', 'Confianza'])
                    st.bar_chart(df_probs[df_probs['Confianza'] > 0.01].set_index('Emoción'))

    if prompt := st.chat_input("Escriba su consulta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    user_profile = df_profiles.loc[st.session_state.selected_profile_id]
                    emotion_probs = emotion_classifier.predict_proba(prompt)[0]
                    top_emotion = max(emotion_probs, key=emotion_probs.get)
                    
                    cognitive_plan, predicted_archetype = cognitive_tutor_system.get_cognitive_plan(
                        user_profile, 
                        emotion_probs
                    )

                    # --- LÓGICA CONVERSACIONAL MEJORADA Y CENTRALIZADA ---
                    # Las listas de emociones ahora se leen desde la configuración
                    constants = st.session_state.config.get('constants', {})
                    negative_emotions = constants.get('negative_emotions', [])
                    positive_emotions = constants.get('positive_emotions', []) 

                    if top_emotion in negative_emotions:
                        intro_message = f"Percibo que puedes sentirte con un poco de **{top_emotion.lower()}**. Analicemos esto juntos con el siguiente plan:"
                    elif top_emotion in positive_emotions:
                        intro_message = f"¡Excelente! Percibo un estado de **{top_emotion.lower()}**. Para potenciar ese impulso, este es el plan:"
                    elif top_emotion == "Sorpresa":
                        intro_message = f"¡Vaya! Parece que esto te ha generado **{top_emotion.lower()}**. Veamos la situación con más detalle:"
                    else: 
                        intro_message = "Entendido. Este es el plan de acción sugerido:"
                    
                    full_response = f"{intro_message}\n\n{cognitive_plan}"
                    st.markdown(full_response)
                    
                    analysis_data = {
                        "archetype": predicted_archetype,
                        "top_emotion": top_emotion,
                        "top_emotion_prob": emotion_probs[top_emotion],
                        "emotion_probs": emotion_probs
                    }
                    st.session_state.messages.append({"role": "assistant", "content": full_response, "analysis": analysis_data})
                    
                    update_metrics(analysis_data)
                    st.rerun()

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la respuesta: {e}")
                    st.code(traceback.format_exc())

# --- 4. PUNTO DE ENTRADA PRINCIPAL ---
def main():
    """Función principal que orquesta la ejecución de la aplicación Streamlit."""
    st.set_page_config(page_title="Tutor Cognitivo Adaptativo", layout="centered", initial_sidebar_state="expanded")

    emotion_classifier, cognitive_tutor_system, df_profiles, config = load_all_models_and_data()
    
    initialize_session_state(df_profiles, config)
    render_sidebar(df_profiles)
    render_chat_interface(emotion_classifier, cognitive_tutor_system, df_profiles)

if __name__ == '__main__':
    main()

