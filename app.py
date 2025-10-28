"""
Aplicación principal de Streamlit para el prototipo de Tesis Doctoral:
Tutor Cognitivo Adaptativo con IA Afectiva.

- Guardado de perfiles de onboarding como JSON individuales (más robusto).
- Visualización de la lógica de fusión Afectivo-Cognitiva en la UI.
- Ruta de logs ahora es un directorio (config.yaml).


- Ruta de guardado de perfiles de onboarding centralizada en config.yaml.
- Se añade un ONBOARDING_ID único (timestamp) a cada perfil guardado.
- Normalización de etiquetas de emoción basada en config.yaml.
- Caption con perfil activo.


- Pipeline de inferencia en tiempo real con modo "Demo" y "Perfil Nuevo".
- Formulario de "onboarding".
- Integración con 'src.profile_inference'.

Componentes integrados:
- Clasificador de Emociones (`emotion_classifier.py`)
- Tutor Cognitivo (`cognitive_model_trainer.py`)
- Motor de Fusión de Expertos (`cognitive_tutor.py`)
- Motor de Inferencia de Perfil (`profile_inference.py`)
- Configuración Central (`config.yaml`)
"""

import streamlit as st
import yaml
import sys
import os
import pandas as pd
import joblib
import traceback
import datetime # Para el ID único
import json     # Para guardar en JSON
from typing import Dict, List, Tuple, Any

# Importación segura de transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.error("Error Crítico: La librería `transformers` no está instalada.")
    TRANSFORMERS_AVAILABLE = False
    class DummyAutoClass: pass
    AutoModelForSequenceClassification = DummyAutoClass
    AutoTokenizer = DummyAutoClass
    st.stop()

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE MÓDulos ---
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from emotion_classifier import EmotionClassifier
    from cognitive_tutor import MoESystem
    from profile_inference import infer_profile_features
except ImportError as e:
    st.error(f"Error Crítico al importar módulos: {e}. Verifique 'src'.")
    st.stop()
except Exception as e:
     st.error(f"Error inesperado al importar módulos: {e}")
     st.stop()

# --- 2. FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---
@st.cache_resource
def load_all_models_and_data(config_path: str = 'config.yaml') -> Tuple[EmotionClassifier, MoESystem, pd.DataFrame, Dict]:
    """
    Carga y prepara todos los artefactos necesarios. Cacheado para rendimiento.
    Valida rutas y dependencias. (MODIFICADO para log dir).
    """
    with st.spinner("Cargando modelos y preparando el sistema..."):
        # Carga robusta de config.yaml
        config_abs_path = os.path.join(project_root, config_path)
        try:
            with open(config_abs_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config:
                    st.error(f"Error Crítico: '{config_abs_path}' vacío o mal formado.")
                    st.stop()
        except FileNotFoundError:
            st.error(f"Error Crítico: No se encuentra '{config_abs_path}'.")
            st.stop()
        except yaml.YAMLError as e:
             st.error(f"Error al parsear '{config_abs_path}': {e}")
             st.stop()

        # Validación robusta de rutas
        paths = config.get('model_paths', {})
        data_paths = config.get('data_paths', {})
        required_paths_keys = {
            'cognitive_tutor': paths.get('cognitive_tutor'),
            'emotion_classifier': paths.get('emotion_classifier'),
            'demo_profiles': data_paths.get('demo_profiles'),
            # --- Ruta de directorio para logs ---
            'onboarding_log_dir': data_paths.get('onboarding_log_dir')
        }

        verified_paths = {}
        for key, path in required_paths_keys.items():
            if not path:
                 if key == 'onboarding_log_dir':
                     st.warning(f"Advertencia: Ruta para '{key}' no definida. No se guardarán perfiles.")
                     verified_paths[key] = None
                     continue
                 else:
                    st.error(f"Error Crítico: Ruta para '{key}' no definida en '{config_path}'.")
                    st.stop()

            # Resolver rutas relativas/absolutas
            if not os.path.isabs(path) and "/content/drive/" not in path:
                 base_dir = project_root
                 abs_path = os.path.join(base_dir, path)
            else:
                 abs_path = path

            # Crear directorio de logs si no existe
            if key == 'onboarding_log_dir':
                try:
                    os.makedirs(abs_path, exist_ok=True)
                    print(f"INFO: Directorio para logs asegurado: {abs_path}")
                except OSError as e:
                    st.error(f"Error Crítico: No se pudo crear directorio '{abs_path}'. Error: {e}")
                    st.stop()
                verified_paths[key] = abs_path
                continue

            # Verificar existencia de archivos de modelos/datos
            if not os.path.exists(abs_path):
                st.error(f"Error Crítico: No se encuentra '{key}': '{abs_path}'. Verifique config y pipelines.")
                st.stop()
            verified_paths[key] = abs_path

        # Carga de modelos y datos
        try:
            emotion_model = AutoModelForSequenceClassification.from_pretrained(verified_paths['emotion_classifier'])
            emotion_tokenizer = AutoTokenizer.from_pretrained(verified_paths['emotion_classifier'])
            emotion_classifier = EmotionClassifier(emotion_model, emotion_tokenizer)

            cognitive_model = joblib.load(verified_paths['cognitive_tutor'])
            df_profiles = pd.read_csv(verified_paths['demo_profiles'], index_col='ID')

            # Validación de features (crucial)
            if not hasattr(cognitive_model, 'feature_names_in_'):
                 st.error("Error Crítico: Modelo cognitivo sin 'feature_names_in_'. Re-entrenar con scikit-learn >= 1.0.")
                 st.stop()
            feature_columns = cognitive_model.feature_names_in_
            if not all(col in df_profiles.columns for col in feature_columns):
                 st.error("Error Crítico: Columnas en 'demo_profiles.csv' no coinciden con modelo. Regenerar ambos.")
                 missing_cols = set(feature_columns) - set(df_profiles.columns)
                 st.error(f"Columnas faltantes: {missing_cols}")
                 st.stop()

            cognitive_tutor_system = MoESystem(
                cognitive_model,
                feature_columns.tolist(),
                config.get('affective_rules', {}),
                config.get('system_thresholds', {})
            )

            # Añadir ruta de log verificada a config
            config['data_paths']['onboarding_log_dir'] = verified_paths.get('onboarding_log_dir')

            return emotion_classifier, cognitive_tutor_system, df_profiles, config

        except Exception as e:
            st.error(f"Error durante la carga de modelos/datos: {e}")
            st.exception(e)
            st.stop()


# --- 3. FUNCIONES DE LA INTERFAZ DE USUARIO ---

def get_initial_session_data() -> Dict[str, Any]:
    """Genera la estructura inicial para métricas y contexto de sesión."""
    return {
        "metrics": {
            "total_interactions": 0, "negative_emotion_count": 0, "positive_emotion_count": 0,
            "emotion_counts": {}, "emotion_confidence_sum": {}, "profile_emotion_counts": {}
        },
        "conversation_context": {"emotional_trajectory": [], "activated_tutors": []}
    }

def initialize_session_state(df_profiles: pd.DataFrame, config: Dict) -> None:
    """Inicializa variables clave en `session_state` si no existen."""
    if "messages" not in st.session_state: st.session_state.messages = []
    if "session_data" not in st.session_state: st.session_state.session_data = get_initial_session_data()

    profile_ids = df_profiles.index.tolist()
    # Lógica de perfil (Épica 2)
    if 'profile_mode' not in st.session_state: st.session_state.profile_mode = 'Demo'
    if 'selected_profile_id' not in st.session_state:
        st.session_state.selected_profile_id = profile_ids[0] if profile_ids else None
    elif st.session_state.selected_profile_id not in profile_ids:
         st.session_state.selected_profile_id = profile_ids[0] if profile_ids else None
    if 'current_user_profile' not in st.session_state: st.session_state.current_user_profile = None
    if 'current_archetype' not in st.session_state: st.session_state.current_archetype = "N/A"
    if 'config' not in st.session_state: st.session_state.config = config

def update_session_data(analysis: Dict) -> None:
    """Actualiza métricas y contexto tras cada interacción."""
    if "session_data" not in st.session_state: return
    session_data = st.session_state.session_data
    metrics, context = session_data["metrics"], session_data["conversation_context"]
    top_emotion = analysis.get('top_emotion', 'Desconocida')
    constants = st.session_state.config.get('constants', {})
    negative_emotions = constants.get('negative_emotions', [])
    positive_emotions = constants.get('positive_emotions', [])

    # Métricas
    metrics["total_interactions"] += 1
    if top_emotion in negative_emotions: metrics["negative_emotion_count"] += 1
    elif top_emotion in positive_emotions: metrics["positive_emotion_count"] += 1
    metrics["emotion_counts"][top_emotion] = metrics["emotion_counts"].get(top_emotion, 0) + 1
    metrics["emotion_confidence_sum"][top_emotion] = metrics["emotion_confidence_sum"].get(top_emotion, 0) + analysis.get('top_emotion_prob', 0.0)
    profile_id_key = 'current_archetype' if st.session_state.profile_mode == 'Perfil Nuevo' else 'selected_profile_id'
    profile_id = st.session_state.get(profile_id_key, 'N/A')
    if profile_id not in metrics["profile_emotion_counts"]: metrics["profile_emotion_counts"][profile_id] = {}
    profile_metrics = metrics["profile_emotion_counts"][profile_id]
    profile_metrics[top_emotion] = profile_metrics.get(top_emotion, 0) + 1

    # Contexto
    context["emotional_trajectory"] = (context["emotional_trajectory"] + [top_emotion])[-5:]
    context["activated_tutors"] = (context["activated_tutors"] + [analysis.get('archetype', 'Desconocido')])[-5:]

def render_sidebar(df_profiles: pd.DataFrame) -> None:
    """Crea y muestra la barra lateral (Selector de modo y métricas)."""
    with st.sidebar:
        st.header("Sobre el Proyecto")
        st.markdown("Demostración del prototipo de tesis de la **Mgter. Ing. Yanina A. Caffetti**.")
        st.header("Configuración de la Simulación")

        def clear_chat_and_profile():
            st.session_state.messages = []
            st.session_state.session_data = get_initial_session_data()
            st.session_state.current_user_profile = None
            st.session_state.current_archetype = "N/A"

        # Selector de Modo (Épica 2)
        st.radio("Seleccione Modo de Perfil:", ['Demo', 'Perfil Nuevo'], key='profile_mode',
                 on_change=clear_chat_and_profile, horizontal=True,
                 help="Elija 'Demo' o 'Perfil Nuevo'.")

        profile_ids = df_profiles.index.tolist()
        if st.session_state.profile_mode == 'Demo':
            if not profile_ids:
                st.warning("No se encontraron perfiles demo.")
                st.session_state.selected_profile_id = None
                return
            try:
                current_index = profile_ids.index(st.session_state.get('selected_profile_id', profile_ids[0]))
            except ValueError: current_index = 0
            st.selectbox("Seleccione un Perfil Demo:", profile_ids, index=current_index,
                         key='selected_profile_id', help="Cambiar reinicia la conversación.",
                         on_change=clear_chat_and_profile)
            if st.session_state.selected_profile_id:
                 st.info(f"Demo activo: **{st.session_state.selected_profile_id}**")
        else: # Modo 'Perfil Nuevo'
            st.info("Complete el formulario para iniciar.")

        if st.button("Reiniciar Conversación"):
            clear_chat_and_profile()
            st.rerun()

        # Métricas y Memoria
        st.header("Métricas de la Sesión")
        metrics = st.session_state.session_data["metrics"]
        if metrics["total_interactions"] > 0:
            col1, col2 = st.columns(2)
            neg_rate = (metrics["negative_emotion_count"] / metrics["total_interactions"]) * 100
            pos_rate = (metrics["positive_emotion_count"] / metrics["total_interactions"]) * 100
            col1.metric("Tasa Negativa", f"{neg_rate:.1f}%")
            col2.metric("Tasa Positiva", f"{pos_rate:.1f}%")
            st.subheader("Distribución Emocional")
            valid_counts = {k:v for k,v in metrics["emotion_counts"].items() if "DESCONOCIDA" not in k and k != "Desconocida"}
            if valid_counts:
                df_dist = pd.DataFrame(valid_counts.items(), columns=['Emoción', 'Frecuencia'])
                st.bar_chart(df_dist.set_index('Emoción'))
        else: st.info("Inicie una conversación.")

        st.header("Memoria Reciente")
        context = st.session_state.session_data["conversation_context"]
        if context["emotional_trajectory"]: st.json({"Emociones": context["emotional_trajectory"]})
        else: st.info("Vacía.")

# --- NUEVA FUNCIÓN (ÉPICA 2 / Refinada) ---
def render_onboarding_form(cognitive_tutor_system: MoESystem) -> None:
    """
    Muestra formulario de onboarding y guarda el perfil inferido como JSON individual.
    """
    st.header("📝 Formulario de Onboarding de Usuario")
    st.markdown("Complete estos datos para generar un perfil cognitivo adaptado.")

    # Mapeo de opciones UI a códigos de dataset
    EDAD_MAP = {"14-39": 3, "40-64": 4, "65+": 5}
    CAPITAL_MAP = {"Primario/Inferior": 1, "Secundario": 3, "Terciario Incompleto": 4, "Terciario Completo": 5}
    OCUP_MAP = {"Ocupado Formal": (1, 1), "Ocupado Informal": (1, 2), "Desocupado": (2, 9), "Inactivo": (3, 9)}
    DIFICULTAD_MAP = {"Sin dificultad": (0,0,0), "Motora": (1,1,1), "Visual": (1,2,1), "Auditiva": (1,3,1),
                     "Cognitiva/Mental": (1,4,1), "Autocuidado": (1,5,1), "Habla/Comunicación": (1,6,1),
                     "Múltiples": (1,7,2)}
    CUD_MAP = {"Sí": 1, "No": 2, "No sabe/NC": 9}

    with st.form(key="onboarding_form"):
        st.subheader("Información Demográfica y Educativa")
        edad_key = st.selectbox("Rango de edad:", list(EDAD_MAP.keys()))
        capital_key = st.selectbox("Máximo nivel educativo:", list(CAPITAL_MAP.keys()))
        st.subheader("Información Laboral y de Discapacidad")
        ocup_key = st.selectbox("Situación ocupacional:", list(OCUP_MAP.keys()))
        dificultad_key = st.selectbox("Condición principal:", list(DIFICULTAD_MAP.keys()))
        cud_key = st.selectbox("¿Posee CUD?", list(CUD_MAP.keys()))
        submitted = st.form_submit_button("Generar Perfil y Empezar")

    if submitted:
        with st.spinner("Analizando perfil..."):
            raw_data = {
                "edad_agrupada": EDAD_MAP[edad_key], "MNEA": CAPITAL_MAP[capital_key],
                "Estado_ocup": OCUP_MAP[ocup_key][0], "cat_ocup": OCUP_MAP[ocup_key][1],
                "dificultad_total": DIFICULTAD_MAP[dificultad_key][0],
                "tipo_dificultad": DIFICULTAD_MAP[dificultad_key][1],
                "dificultades": DIFICULTAD_MAP[dificultad_key][2],
                "certificado": CUD_MAP[cud_key],
                "PC08": 9, "pc03": 9, "tipo_hogar": 9 # Placeholders
            }
            full_profile_series = infer_profile_features(raw_data)
            if full_profile_series is None or full_profile_series.empty:
                st.error("Error al generar perfil.")
                return

            model = cognitive_tutor_system.cognitive_model
            try:
                profile_features_for_model = full_profile_series[model.feature_names_in_]
            except (KeyError, AttributeError) as e:
                st.error(f"Error: Incompatibilidad perfil/modelo. {e}")
                return
            predicted_archetype = model.predict(profile_features_for_model.values.reshape(1, -1))[0]

            # --- LÓGICA DE GUARDADO EN JSON INDIVIDUAL (MEJORADA) ---
            try:
                output_dir_path = st.session_state.config['data_paths'].get('onboarding_log_dir')
                if output_dir_path:
                    onboarding_id = f"Onboarding_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    profile_dict = full_profile_series.astype(str).to_dict() # Convertir a str para JSON
                    profile_dict['ONBOARDING_ID'] = onboarding_id
                    profile_dict['PREDICTED_ARCHETYPE'] = predicted_archetype

                    # Asegurar que la ruta sea absoluta y exista
                    if not os.path.isabs(output_dir_path):
                        output_dir_path = os.path.join(project_root, output_dir_path)
                    os.makedirs(output_dir_path, exist_ok=True) # Crea el directorio si no existe

                    file_path = os.path.join(output_dir_path, f"{onboarding_id}.json")

                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(profile_dict, f, ensure_ascii=False, indent=4)
                    print(f"INFO: Perfil onboarding guardado en {file_path}")
                else:
                    print("INFO: No se guardó perfil (ruta no configurada).")
            except Exception as save_e:
                st.warning(f"Advertencia: No se pudo guardar perfil. Error: {save_e}")
                print(f"ERROR guardando perfil: {save_e}")
                traceback.print_exc()
            # --- FIN LÓGICA DE GUARDADO ---

            st.session_state.current_user_profile = full_profile_series
            st.session_state.current_archetype = predicted_archetype
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"¡Perfil generado! Arquetipo inicial: **{predicted_archetype}**. Listo para tu consulta."
            })
            st.rerun()

# --- ¡NUEVA FUNCIÓN DE VISUALIZACIÓN! ---
def render_adaptive_logic_expander(analysis_data: Dict, config: Dict) -> None:
    """Muestra un expander con la lógica de fusión Afectivo-Cognitiva."""
    if not analysis_data or not isinstance(analysis_data, dict):
        return

    with st.expander("Ver Lógica Adaptativa de la Respuesta"):
        st.markdown("**1. Perfil Cognitivo Base:**")
        st.write(f"Arquetipo Predicho: **{analysis_data.get('archetype', 'N/A')}**")

        st.markdown("**2. Análisis Afectivo:**")
        emotion_probs = analysis_data.get('emotion_probs', {})
        top_emotion = analysis_data.get('top_emotion', 'Desconocida')
        top_prob = analysis_data.get('top_emotion_prob', 0.0)
        st.write(f"Emoción Dominante: **{top_emotion}** ({top_prob:.1%})")
        
        valid_probs = {k: v for k, v in emotion_probs.items() if "DESCONOCIDA" not in k and v > 0.05}
        if valid_probs:
            df_probs = pd.DataFrame(valid_probs.items(), columns=['Emoción', 'Confianza'])
            st.bar_chart(df_probs.set_index('Emoción'), height=200)

        st.markdown("**3. Modulación Afectiva Aplicada:**")
        affective_rules = config.get('affective_rules', {})
        applied_rules = []
        min_prob_threshold = config.get('system_thresholds', {}).get('affective_engine', {}).get('min_emotion_probability', 0.1)

        for emotion, prob in emotion_probs.items():
            emotion_norm = emotion.strip().capitalize() # Asegurar capitalización
            if prob >= min_prob_threshold and emotion_norm in affective_rules:
                rules = affective_rules[emotion_norm]
                for archetype, factor in rules.items():
                    applied_rules.append(f"- Emoción '{emotion_norm}' ({prob:.0%}) -> Modifica '{archetype}' (Factor: {factor})")

        if applied_rules:
            st.markdown("\n".join(applied_rules))
        else:
            st.write("No se aplicaron reglas de modulación significativas para esta emoción.")

        st.markdown("**4. Pesos Finales de Expertos (Contribución a la Respuesta):**")
        final_weights = analysis_data.get('final_weights')
        if final_weights and isinstance(final_weights, dict):
             # Filtrar pesos bajos y ordenar
             sorted_weights = sorted(
                 [(k, v) for k, v in final_weights.items() if v > 0.01],
                 key=lambda item: item[1],
                 reverse=True
             )
             if sorted_weights:
                 df_weights = pd.DataFrame(sorted_weights, columns=['Tutor Experto (Arquetipo)', 'Peso Final'])
                 st.dataframe(df_weights.set_index('Tutor Experto (Arquetipo)'))
             else:
                 st.write("Ningún experto tuvo un peso final significativo.")
        else:
            st.write("Pesos finales no disponibles (requiere modificación en MoESystem).")


def render_chat_interface(
    emotion_classifier: EmotionClassifier,
    cognitive_tutor_system: MoESystem
) -> None:
    """
    Renderiza la interfaz de chat principal. Incluye visualización de lógica adaptativa.
    """
    st.title("🧠 Tutor Cognitivo Adaptativo con IA Afectiva 🤖")
    st.caption(f"👤 Perfil activo: **{st.session_state.current_archetype}** ({st.session_state.profile_mode})")

    # --- ¡NUEVO EXPANDER DE VISUALIZACIÓN! ---
    # Intentar obtener los datos de análisis del último mensaje del asistente
    last_analysis_data = None
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_analysis_data = st.session_state.messages[-1].get("analysis")
    render_adaptive_logic_expander(last_analysis_data, st.session_state.config)
    # --- FIN EXPANDER ---

    # Mostrar mensajes existentes
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Quitar el expander de aquí, ya está arriba
            # if "analysis" in message and isinstance(message["analysis"], dict):
            #     with st.expander("Ver Análisis Detallado"): ...

    # Entrada de usuario
    if prompt := st.chat_input("Escriba su consulta aquí..."):
        if st.session_state.current_user_profile is None:
            st.warning("Error: Perfil no cargado. Seleccione 'Demo' o 'Perfil Nuevo'.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty(); message_placeholder.markdown("Pensando...")
            try:
                user_profile = st.session_state.current_user_profile
                emotion_probs = emotion_classifier.predict_proba(prompt)[0]

                # Normalización de Emoción (basada en YAML)
                top_emotion_raw = max(emotion_probs, key=emotion_probs.get) if emotion_probs else "Desconocida"
                official_labels = st.session_state.config.get('constants', {}).get('emotion_labels', [])
                label_map = {label.strip().lower(): label for label in official_labels}
                normalized_key = top_emotion_raw.strip().lower()
                top_emotion_normalized = label_map.get(normalized_key, top_emotion_raw.strip().capitalize())
                if "Etiqueta_" in top_emotion_normalized or not top_emotion_normalized:
                    top_emotion_normalized = "Desconocida"

                # --- ¡MODIFICACIÓN IMPORTANTE! ---
                # Asumir que get_cognitive_plan ahora devuelve los pesos finales
                plan_result = cognitive_tutor_system.get_cognitive_plan(
                    user_profile, emotion_probs,
                    st.session_state.session_data["conversation_context"],
                    st.session_state.config, prompt
                )
                # Desempaquetar el resultado esperado (ajustar si tu función devuelve otra cosa)
                if isinstance(plan_result, tuple) and len(plan_result) == 3:
                     cognitive_plan, predicted_archetype, final_weights = plan_result
                else:
                     # Fallback si MoESystem no fue modificado
                     st.warning("Advertencia: MoESystem.get_cognitive_plan no devolvió los pesos finales. La visualización estará incompleta.")
                     cognitive_plan, predicted_archetype = plan_result # Asumir el retorno antiguo
                     final_weights = None # Marcar como no disponible


                # Respuestas empáticas (basadas en emoción normalizada)
                empathetic_responses = { # ... (igual que antes) ...
                    "Alegria": "¡Qué buena noticia! Me alegra sentir tu optimismo. Para potenciar ese impulso, este es el plan:",
                    "Confianza": "¡Excelente! Percibo mucha seguridad en tus palabras. Usemos esa confianza como base para el siguiente plan de acción:",
                    "Anticipacion": "Noto tu expectativa. ¡Esa energía es muy valiosa! Enfoquémosla con el siguiente plan:",
                    "Tristeza": "Entiendo que puedas sentirte así. Analicemos la situación juntos. Te propongo el siguiente plan para que avancemos:",
                    "Miedo": "Comprendo que esta situación pueda generar incertidumbre. No te preocupes, estamos aquí para afrontarla. Este es el plan:",
                    "Ira": "Percibo tu frustración. Es una reacción válida. Vamos a canalizar esa energía de manera constructiva con este plan de acción:",
                    "Sorpresa": "¡Vaya! Parece que esto te ha tomado por sorpresa. Analicemos con calma la situación. Este es el plan:",
                    "Desconocida": "Entendido. Este es el plan de acción sugerido:"
                }
                for label in official_labels:
                    if label not in empathetic_responses: empathetic_responses[label] = f"Detecté '{label}'. Plan:"
                intro_message = empathetic_responses.get(top_emotion_normalized, empathetic_responses["Desconocida"])

                full_response = f"{intro_message}\n\n{cognitive_plan}"
                message_placeholder.markdown(full_response)

                # Guardar datos de análisis (incluyendo pesos finales)
                analysis_data = {
                    "archetype": predicted_archetype,
                    "top_emotion": top_emotion_normalized,
                    "top_emotion_prob": emotion_probs.get(top_emotion_raw, 0.0),
                    "emotion_probs": emotion_probs,
                    "final_weights": final_weights # <-- Añadir pesos
                }
                st.session_state.messages.append({"role": "assistant", "content": full_response, "analysis": analysis_data})
                st.session_state.current_archetype = predicted_archetype
                update_session_data(analysis_data)
                # No se necesita st.rerun() aquí, pero sí se necesita después de agregar el mensaje
                # para que el nuevo expander se muestre inmediatamente.
                st.rerun()


            except Exception as e:
                error_message = f"Error al generar respuesta: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_message}"})
                st.code(traceback.format_exc())

# --- 4. PUNTO DE ENTRADA PRINCIPAL ---
def main():
    """Orquesta la ejecución de la app Streamlit (Épica 2)."""
    st.set_page_config(page_title="Tutor Cognitivo Conversacional", layout="wide", initial_sidebar_state="expanded")
    try:
         emotion_classifier, cognitive_tutor_system, df_profiles, config = load_all_models_and_data()
    except Exception as load_error:
         st.error("Error fatal durante la carga inicial."); st.exception(load_error); return

    try:
        initialize_session_state(df_profiles, config)
        render_sidebar(df_profiles)

        # Cargar perfil demo si es necesario
        if st.session_state.profile_mode == 'Demo' and st.session_state.current_user_profile is None:
            if st.session_state.selected_profile_id:
                profile_series = df_profiles.loc[st.session_state.selected_profile_id]
                st.session_state.current_user_profile = profile_series
                model = cognitive_tutor_system.cognitive_model
                try:
                    profile_for_prediction = profile_series[model.feature_names_in_]
                    st.session_state.current_archetype = model.predict(profile_for_prediction.values.reshape(1, -1))[0]
                except KeyError as e: st.error(f"Error: Perfil demo incompatible. Faltan: {e}"); return
            else: st.error("Perfil demo no seleccionado."); return

        # Decidir qué pantalla mostrar
        if st.session_state.current_user_profile is not None:
            render_chat_interface(emotion_classifier, cognitive_tutor_system)
        elif st.session_state.profile_mode == 'Perfil Nuevo':
            render_onboarding_form(cognitive_tutor_system)
        else:
            st.info("Seleccione un perfil demo.")

    except Exception as app_error:
        st.error("Error inesperado en la aplicación."); st.exception(app_error)

if __name__ == '__main__':
    main()

