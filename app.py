"""
Aplicación principal de Streamlit para el prototipo de Tesis Doctoral:
Tutor Cognitivo Adaptativo con IA Afectiva.

Esta aplicación sirve como la interfaz de demostración interactiva que integra
todos los componentes del sistema:
- El Clasificador de Emociones (entrenado en `emotion_classifier.py`)
- El Tutor Cognitivo (entrenado en `cognitive_model_trainer.py`)
- El Motor de Fusión de Expertos (definido en `cognitive_tutor.py`)
- La Configuración Central (leída desde `config.yaml`)
"""

import streamlit as st
import yaml
import sys
import os
import pandas as pd
import joblib
import traceback
from typing import Dict, List, Tuple, Any

# Importación segura de transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.error("Error Crítico: La librería `transformers` no está instalada. La aplicación no puede funcionar.")
    TRANSFORMERS_AVAILABLE = False
    # Definir clases dummy para que el script no falle al importar
    class DummyAutoClass: pass
    AutoModelForSequenceClassification = DummyAutoClass
    AutoTokenizer = DummyAutoClass
    st.stop()

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE MÓDulos ---
# Añadir la carpeta 'src' al path de Python para encontrar los módulos locales
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from emotion_classifier import EmotionClassifier
    from cognitive_tutor import MoESystem
except ImportError as e:
    st.error(f"Error Crítico al importar los módulos: {e}. "
             f"Asegúrate de que los archivos `emotion_classifier.py` y `cognitive_tutor.py` "
             f"estén presentes en la carpeta 'src'.")
    st.stop()
except Exception as e:
     st.error(f"Error inesperado al importar módulos locales: {e}")
     st.stop()

# --- 2. FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---
@st.cache_resource # Usar cache_resource para objetos pesados como modelos y datos
def load_all_models_and_data(config_path: str = 'config.yaml') -> Tuple[EmotionClassifier, MoESystem, pd.DataFrame, Dict]:
    """
    Carga y prepara todos los artefactos necesarios para la aplicación.

    Esta función se cachea para que los modelos pesados de IA solo se carguen
    una vez al iniciar la aplicación, mejorando drásticamente el rendimiento.
    Valida la existencia de todas las rutas de configuración y modelos.

    Args:
        config_path: Ruta al archivo de configuración principal.

    Returns:
        Una tupla con todos los componentes inicializados del sistema:
        (clasificador_emociones, sistema_tutor_cognitivo, perfiles_demo, config)
    
    Raises:
        st.stop(): Detiene la ejecución de la app si falta un archivo crítico.
    """
    with st.spinner("Cargando modelos y preparando el sistema..."):
        
        # Carga robusta de config.yaml
        config_abs_path = os.path.join(project_root, config_path)
        try:
            with open(config_abs_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config:
                    st.error(f"Error Crítico: El archivo '{config_abs_path}' está vacío o mal formado.")
                    st.stop()
        except FileNotFoundError:
            st.error(f"Error Crítico: No se encuentra '{config_abs_path}'.")
            st.stop()
        except yaml.YAMLError as e:
             st.error(f"Error al parsear '{config_abs_path}': {e}")
             st.stop()

        # Validación robusta de todas las rutas necesarias
        paths = config.get('model_paths', {})
        data_paths = config.get('data_paths', {})
        required_paths_keys = {
            'cognitive_tutor': paths.get('cognitive_tutor'),
            'emotion_classifier': paths.get('emotion_classifier'),
            'demo_profiles': data_paths.get('demo_profiles')
        }

        # Verificar y convertir rutas a absolutas
        verified_paths = {}
        for key, path in required_paths_keys.items():
            if not path:
                 st.error(f"Error Crítico: La ruta para '{key}' no está definida en '{config_path}'.")
                 st.stop()
            
            # Asumir que las rutas en Google Drive son absolutas, otras son relativas
            if not os.path.isabs(path) and "/content/drive/" not in path:
                 abs_path = os.path.join(project_root, path)
            else:
                 abs_path = path
            
            if not os.path.exists(abs_path):
                st.error(f"Error Crítico: No se encuentra la ruta para '{key}': '{abs_path}'. "
                         "Asegúrate de haber ejecutado el pipeline de entrenamiento ('python train.py --model all') "
                         "y que las rutas en 'config.yaml' sean correctas.")
                st.stop()
            verified_paths[key] = abs_path # Guardar la ruta absoluta verificada

        # Carga de modelos y datos
        try:
            emotion_model = AutoModelForSequenceClassification.from_pretrained(verified_paths['emotion_classifier'])
            emotion_tokenizer = AutoTokenizer.from_pretrained(verified_paths['emotion_classifier'])
            emotion_classifier = EmotionClassifier(emotion_model, emotion_tokenizer)
            
            cognitive_model = joblib.load(verified_paths['cognitive_tutor'])
            df_profiles = pd.read_csv(verified_paths['demo_profiles'], index_col='ID')
            
            feature_columns = [col for col in df_profiles.columns if '_memb' in col]
            if not feature_columns:
                 st.warning("No se encontraron columnas de características ('_memb') en demo_profiles.csv.")

            cognitive_tutor_system = MoESystem(
                cognitive_model,
                feature_columns,
                config.get('affective_rules', {}),
                config.get('system_thresholds', {})
            )

            return emotion_classifier, cognitive_tutor_system, df_profiles, config

        except Exception as e:
            st.error(f"Error durante la carga de modelos o datos: {e}")
            st.exception(e)
            st.stop()


# --- 3. FUNCIONES DE LA INTERFAZ DE USUARIO ---

def get_initial_session_data() -> Dict[str, Any]:
    """
    Genera la estructura de diccionario para inicializar o reiniciar las métricas
    y el contexto de la sesión.
    
    Returns:
        Un diccionario con la estructura de datos de la sesión.
    """
    return {
        "metrics": {
            "total_interactions": 0,
            "negative_emotion_count": 0,
            "positive_emotion_count": 0,
            "emotion_counts": {},
            "emotion_confidence_sum": {},
            "profile_emotion_counts": {}
        },
        "conversation_context": {
            "emotional_trajectory": [], # Historial de las últimas 5 emociones
            "activated_tutors": [],   # Historial de los últimos 5 arquetipos
        }
    }

def initialize_session_state(df_profiles: pd.DataFrame, config: Dict) -> None:
    """
    Inicializa las variables clave en el `session_state` de Streamlit
    si no existen.
    
    Args:
        df_profiles: DataFrame con los perfiles de demo para seleccionar el primero.
        config: El diccionario de configuración cargado.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_data" not in st.session_state:
        st.session_state.session_data = get_initial_session_data()
        
    profile_ids = df_profiles.index.tolist()
    if 'selected_profile_id' not in st.session_state:
        st.session_state.selected_profile_id = profile_ids[0] if profile_ids else None
    elif st.session_state.selected_profile_id not in profile_ids:
         st.session_state.selected_profile_id = profile_ids[0] if profile_ids else None

    if 'config' not in st.session_state:
        st.session_state.config = config

def update_session_data(analysis: Dict) -> None:
    """
    Actualiza tanto las métricas como el contexto de la conversación en
    `st.session_state` después de cada interacción.

    Args:
        analysis: Un diccionario que contiene los datos de la última predicción,
                  incluyendo 'top_emotion' (normalizada).
    """
    if "session_data" not in st.session_state: return

    session_data = st.session_state.session_data
    metrics = session_data["metrics"]
    context = session_data["conversation_context"]
    
    top_emotion = analysis.get('top_emotion', 'Desconocida')
    
    constants = st.session_state.config.get('constants', {})
    negative_emotions = constants.get('negative_emotions', [])
    positive_emotions = constants.get('positive_emotions', [])
    
    # Actualizar Métricas
    metrics["total_interactions"] += 1
    if top_emotion in negative_emotions:
        metrics["negative_emotion_count"] += 1
    elif top_emotion in positive_emotions:
        metrics["positive_emotion_count"] += 1

    metrics["emotion_counts"][top_emotion] = metrics["emotion_counts"].get(top_emotion, 0) + 1
    metrics["emotion_confidence_sum"][top_emotion] = metrics["emotion_confidence_sum"].get(top_emotion, 0) + analysis.get('top_emotion_prob', 0.0)

    profile_id = st.session_state.get('selected_profile_id')
    if profile_id:
        if profile_id not in metrics["profile_emotion_counts"]:
            metrics["profile_emotion_counts"][profile_id] = {}
        profile_metrics = metrics["profile_emotion_counts"][profile_id]
        profile_metrics[top_emotion] = profile_metrics.get(top_emotion, 0) + 1


    # Actualizar Contexto
    context["emotional_trajectory"].append(top_emotion)
    context["activated_tutors"].append(analysis.get('archetype', 'Desconocido'))
    context["emotional_trajectory"] = context["emotional_trajectory"][-5:] # Limitar historial
    context["activated_tutors"] = context["activated_tutors"][-5:]

def render_sidebar(df_profiles: pd.DataFrame) -> None:
    """
    Crea y muestra todo el contenido de la barra lateral de la aplicación,
    incluyendo la selección de perfil y las métricas de sesión.

    Args:
        df_profiles: DataFrame con los perfiles de demo para el selector.
    """
    with st.sidebar:
        st.header("Sobre el Proyecto")
        st.markdown("Demostración del prototipo de tesis de la **Mgter. Ing. Yanina A. Caffetti**.")
        
        st.header("Configuración de la Simulación")
        
        profile_ids = df_profiles.index.tolist()
        
        def clear_chat_on_profile_change():
            """Reinicia el chat y las métricas al cambiar de perfil."""
            st.session_state.messages = []
            st.session_state.session_data = get_initial_session_data()

        if not profile_ids:
            st.warning("No se encontraron perfiles de demostración.")
            st.session_state.selected_profile_id = None
            return

        try:
            current_index = profile_ids.index(st.session_state.get('selected_profile_id', profile_ids[0]))
        except ValueError:
             current_index = 0

        st.selectbox(
            "Seleccione un Perfil de Usuario:",
            profile_ids,
            index=current_index,
            key='selected_profile_id',
            help="Cambiar de perfil reiniciará la conversación.",
            on_change=clear_chat_on_profile_change
        )
        if st.session_state.selected_profile_id:
             st.info(f"Perfil activo: **{st.session_state.selected_profile_id}**")
        
        if st.button("Reiniciar Conversación Actual"):
            clear_chat_on_profile_change()
            st.rerun()

        st.header("Métricas de la Sesión")
        if "session_data" in st.session_state:
            metrics = st.session_state.session_data["metrics"]
            if metrics["total_interactions"] > 0:
                col1, col2 = st.columns(2)
                with col1:
                    neg_rate = (metrics["negative_emotion_count"] / metrics["total_interactions"]) * 100
                    st.metric(label="Tasa Negativa", value=f"{neg_rate:.1f}%")
                with col2:
                    pos_rate = (metrics["positive_emotion_count"] / metrics["total_interactions"]) * 100
                    st.metric(label="Tasa Positiva", value=f"{pos_rate:.1f}%")

                st.subheader("Distribución General de Emociones")
                if metrics["emotion_counts"]:
                    valid_counts = {k:v for k,v in metrics["emotion_counts"].items() if "DESCONOCIDA" not in k and k != "Desconocida"}
                    if valid_counts:
                        df_emotion_dist = pd.DataFrame(valid_counts.items(), columns=['Emoción', 'Frecuencia'])
                        st.bar_chart(df_emotion_dist.set_index('Emoción'))
            else:
                st.info("Inicie una conversación para ver las métricas.")
        
            st.header("Memoria Reciente")
            context = st.session_state.session_data["conversation_context"]
            if context["emotional_trajectory"]:
                st.write("**Historial Emocional:**")
                st.json(context["emotional_trajectory"])
            else:
                st.info("La memoria está vacía.")

def render_chat_interface(
    emotion_classifier: EmotionClassifier, 
    cognitive_tutor_system: MoESystem, 
    df_profiles: pd.DataFrame
) -> None:
    """
    Renderiza la interfaz de chat principal y maneja la lógica de la conversación.

    Args:
        emotion_classifier: La instancia del clasificador de emociones.
        cognitive_tutor_system: La instancia del sistema MoE.
        df_profiles: DataFrame con los perfiles de demo.
    """
    st.title("🧠 Tutor Cognitivo Adaptativo con IA Afectiva 🤖")

    # Mostrar mensajes existentes
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "analysis" in message and isinstance(message["analysis"], dict):
                with st.expander("Ver Análisis Detallado"):
                    analysis = message["analysis"]
                    st.info(f"**👤 Arquetipo Cognitivo:** {analysis.get('archetype', 'N/A')}")
                    st.info(f"**🧠 Emoción Dominante:** {analysis.get('top_emotion', 'N/A')} ({analysis.get('top_emotion_prob', 0.0):.0%})")
                    if 'emotion_probs' in analysis and isinstance(analysis['emotion_probs'], dict):
                        valid_probs = {k:v for k,v in analysis['emotion_probs'].items() if "DESCONOCIDA" not in k and v > 0.01}
                        if valid_probs:
                            df_probs = pd.DataFrame(valid_probs.items(), columns=['Emoción', 'Confianza'])
                            st.bar_chart(df_probs.set_index('Emoción'))
                        else:
                             st.write("No hay probabilidades de emoción significativas para mostrar.")

    # Entrada de usuario
    if prompt := st.chat_input("Escriba su consulta aquí..."):
        if not st.session_state.get('selected_profile_id'):
            st.warning("Por favor, seleccione un perfil de usuario en la barra lateral.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Pensando...")

            try:
                user_profile = df_profiles.loc[st.session_state.selected_profile_id]
                
                # --- LÓGICA DE PREDICCIÓN Y NORMALIZACIÓN ROBUSTA ---
                emotion_probs = emotion_classifier.predict_proba(prompt)[0]
                top_emotion_raw = max(emotion_probs, key=emotion_probs.get) if emotion_probs else "Desconocida"
                
                # Normalización: Elimina espacios, convierte a minúscula, luego capitaliza.
                top_emotion_normalized = top_emotion_raw.strip().lower().capitalize()
                
                # Opcional: Log de depuración si la normalización cambió algo
                if top_emotion_raw != top_emotion_normalized and "Etiqueta_" not in top_emotion_raw:
                    st.warning(f"DEBUG: Emoción normalizada. Original: '{top_emotion_raw}', Normalizada: '{top_emotion_normalized}'")

                # Llamada al MoESystem (pasando el prompt y el contexto)
                cognitive_plan, predicted_archetype = cognitive_tutor_system.get_cognitive_plan(
                    user_profile,
                    emotion_probs,
                    st.session_state.session_data["conversation_context"],
                    st.session_state.config,
                    prompt
                )

                # Diccionario de respuestas empáticas (claves deben estar normalizadas)
                empathetic_responses = {
                    "Alegria": "¡Qué buena noticia! Me alegra sentir tu optimismo. Para potenciar ese impulso, este es el plan:",
                    "Confianza": "¡Excelente! Percibo mucha seguridad en tus palabras. Usemos esa confianza como base para el siguiente plan de acción:",
                    "Anticipacion": "Noto tu expectativa. ¡Esa energía es muy valiosa! Enfoquémosla con el siguiente plan:",
                    "Tristeza": "Entiendo que puedas sentirte así. Analicemos la situación juntos. Te propongo el siguiente plan para que avancemos:",
                    "Miedo": "Comprendo que esta situación pueda generar incertidumbre. No te preocupes, estamos aquí para afrontarla. Este es el plan:",
                    "Ira": "Percibo tu frustración. Es una reacción válida. Vamos a canalizar esa energía de manera constructiva con este plan de acción:",
                    "Sorpresa": "¡Vaya! Parece que esto te ha tomado por sorpresa. Analicemos con calma la situación. Este es el plan:",
                    "Desconocida": "Entendido. Este es el plan de acción sugerido:"
                }

                # Búsqueda usando la clave normalizada
                intro_message = empathetic_responses.get(top_emotion_normalized, empathetic_responses["Desconocida"])
                
                full_response = f"{intro_message}\n\n{cognitive_plan}"
                message_placeholder.markdown(full_response)
                
                # Guardar datos de análisis consistentes (usando la versión normalizada)
                analysis_data = {
                    "archetype": predicted_archetype,
                    "top_emotion": top_emotion_normalized, # Guardar versión normalizada
                    "top_emotion_prob": emotion_probs.get(top_emotion_raw, 0.0),
                    "emotion_probs": emotion_probs
                }
                st.session_state.messages.append({"role": "assistant", "content": full_response, "analysis": analysis_data})
                
                update_session_data(analysis_data)
                # st.rerun() # No es necesario al usar el placeholder

            except Exception as e:
                error_message = f"Ocurrió un error al generar la respuesta: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_message}"})
                st.code(traceback.format_exc())

# --- 4. PUNTO DE ENTRADA PRINCIPAL ---
def main():
    """Función principal que orquesta la ejecución de la aplicación Streamlit."""
    st.set_page_config(page_title="Tutor Cognitivo Conversacional", layout="wide", initial_sidebar_state="expanded")

    try:
         emotion_classifier, cognitive_tutor_system, df_profiles, config = load_all_models_and_data()
    except Exception as load_error:
         st.error("Error fatal durante la carga inicial de modelos y datos.")
         st.exception(load_error)
         return

    try:
        initialize_session_state(df_profiles, config)
        render_sidebar(df_profiles)
        render_chat_interface(emotion_classifier, cognitive_tutor_system, df_profiles)
    except Exception as app_error:
        st.error("Ocurrió un error inesperado en la aplicación.")
        st.exception(app_error)


if __name__ == '__main__':
    main()

