"""
Aplicación principal de Streamlit para el prototipo de Tesis Doctoral:
Tutor Cognitivo Adaptativo con IA Afectiva y Arquitectura Neuro-Simbólica.

Características:
- Interfaz de Chat con Gating Afectivo.
- Visualización de Auditoría XAI (Explainable AI) en tiempo real.
- Gestión de Perfiles de Usuario (Onboarding y Demo).
- Integración completa con el backend MoE refinado (Soporte dinámico de retornos).
"""

import streamlit as st
import yaml
import sys
import os
import pandas as pd
import joblib
import traceback
import datetime
import json
from typing import Dict, Tuple

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE MÓDULOS ---
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Configuración de página (Primera llamada obligatoria)
st.set_page_config(
    page_title="Tutor Cognitivo Neuro-Simbólico",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from src.emotion_classifier import EmotionClassifier
    from src.cognitive_tutor import MoESystem
    from src.profile_inference import infer_profile_features
except ImportError as e:
    st.error(f"Error Crítico al importar módulos del sistema: {e}. Verifique la carpeta 'src'.")
    st.stop()

# --- 2. FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---
@st.cache_resource
def load_all_models_and_data(config_path: str = 'config.yaml') -> Tuple[EmotionClassifier, MoESystem, pd.DataFrame, Dict]:
    """Carga modelos, configuraciones y perfiles demo. Cacheado para eficiencia."""
    with st.spinner("Inicializando Arquitectura Neuro-Simbólica..."):
        # 1. Cargar Configuración
        config_abs_path = os.path.join(project_root, config_path)
        try:
            with open(config_abs_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error cargando {config_path}: {e}")
            st.stop()

        # 2. Validar Rutas
        paths = config.get('model_paths', {})
        data_paths = config.get('data_paths', {})
        
        def resolve_path(path):
            if not path: return None
            return path if os.path.isabs(path) else os.path.join(project_root, path)

        model_cog_path = resolve_path(paths.get('cognitive_tutor'))
        model_emo_path = resolve_path(paths.get('emotion_classifier'))
        demo_path = resolve_path(data_paths.get('demo_profiles'))
        log_dir = resolve_path(data_paths.get('onboarding_log_dir'))

        if log_dir: os.makedirs(log_dir, exist_ok=True)

        # 3. Cargar Artefactos
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            emo_model = AutoModelForSequenceClassification.from_pretrained(model_emo_path)
            emo_tokenizer = AutoTokenizer.from_pretrained(model_emo_path)
            emotion_classifier = EmotionClassifier(emo_model, emo_tokenizer)

            if not os.path.exists(model_cog_path):
                st.error(f"Modelo cognitivo no encontrado en: {model_cog_path}")
                st.stop()
            
            cognitive_model = joblib.load(model_cog_path)
            
            if os.path.exists(demo_path):
                df_profiles = pd.read_csv(demo_path, index_col='ID')
            else:
                st.warning("Archivo de perfiles demo no encontrado. Se usará DataFrame vacío.")
                df_profiles = pd.DataFrame()

            feature_columns = getattr(cognitive_model, 'feature_names_in_', [])

            moe_system = MoESystem(
                cognitive_model,
                feature_columns,
                config.get('affective_rules', {}),
                config.get('system_thresholds', {})
            )
            
            config['data_paths']['onboarding_log_dir'] = log_dir
            return emotion_classifier, moe_system, df_profiles, config

        except Exception as e:
            st.error(f"Error fatal inicializando componentes: {e}")
            st.code(traceback.format_exc())
            st.stop()

# --- 3. GESTIÓN DE SESIÓN Y ESTADO ---

def get_initial_session_data() -> Dict:
    return {
        "metrics": {
            "total_interactions": 0,
            "negative_emotion_count": 0,
            "emotion_counts": {},
            "profile_emotion_counts": {}
        },
        "conversation_context": {
            "emotional_trajectory": [],
            "activated_tutors": []
        }
    }

def initialize_session_state(df_profiles: pd.DataFrame, config: Dict):
    if "messages" not in st.session_state: st.session_state.messages = []
    if "session_data" not in st.session_state: st.session_state.session_data = get_initial_session_data()
    if "config" not in st.session_state: st.session_state.config = config
    
    if "current_user_profile" not in st.session_state: st.session_state.current_user_profile = None
    if "current_archetype" not in st.session_state: st.session_state.current_archetype = "N/A"
    if "profile_mode" not in st.session_state: st.session_state.profile_mode = "Demo"
    
    profile_ids = df_profiles.index.tolist()
    if "selected_profile_id" not in st.session_state:
        st.session_state.selected_profile_id = profile_ids[0] if profile_ids else None

def update_metrics(analysis: Dict):
    metrics = st.session_state.session_data["metrics"]
    context = st.session_state.session_data["conversation_context"]
    top_emotion = analysis.get("top_emotion", "Neutral")
    metrics["total_interactions"] += 1
    metrics["emotion_counts"][top_emotion] = metrics["emotion_counts"].get(top_emotion, 0) + 1
    context["emotional_trajectory"] = (context["emotional_trajectory"] + [top_emotion])[-5:]

# --- 4. COMPONENTES UI (VISUALIZACIÓN XAI) ---

def render_adaptive_logic_expander(analysis_data: Dict, config: Dict):
    if not analysis_data: return

    with st.expander("🔍 Auditoría del Sistema (XAI & MoE)", expanded=False):
        col_emo, col_cog = st.columns(2)
        
        with col_emo:
            st.markdown("### 🧡 Estado Afectivo")
            emo = analysis_data.get('top_emotion', 'N/A')
            prob = analysis_data.get('top_emotion_prob', 0.0)
            st.info(f"Emoción: **{emo}** ({prob:.1%})")
            
            all_probs = analysis_data.get('emotion_probs', {})
            if all_probs:
                clean_probs = {k:v for k,v in all_probs.items() if v > 0.05}
                st.bar_chart(clean_probs, height=150)

        with col_cog:
            st.markdown("### 🧠 Decisión Cognitiva")
            xai = analysis_data.get('xai_metadata', {})
            metrics = xai.get('execution_metrics', {})
            
            raw_arch = metrics.get('raw_prediction', 'N/A')
            final_arch = metrics.get('selected_archetype', 'N/A')
            veto = metrics.get('veto_applied', False)
            
            if veto:
                st.error(f"🛡️ **GUARDRAIL ACTIVADO**")
                st.markdown(f"Neuronal: `{raw_arch}` 🚫 ➔ Simbólico: `{final_arch}` ✅")
                st.caption(f"Razón: {xai.get('guardrail_reason', {}).get('rule_id', 'Regla de Dominio')}")
            else:
                st.success(f"✅ **Predicción Validada**")
                st.markdown(f"Arquetipo: **{final_arch}**")

        st.markdown("---")
        st.markdown("### ⚙️ Motor de Inferencia")
        c1, c2, c3 = st.columns(3)
        c1.metric("Modo Búsqueda", str(metrics.get('expert_search_mode', 'N/A')))
        c2.metric("Masa Efectiva MoE", f"{metrics.get('moe_effective_mass', 0):.2f}")
        c3.metric("Congruencia", str(metrics.get('affective_congruence', 'N/A')))

        st.markdown("#### ⚖️ Pesos Finales")
        weights = analysis_data.get('final_weights', {})
        if weights:
            active_weights = {k: v for k, v in weights.items() if v > 0.01}
            st.dataframe(
                pd.DataFrame(active_weights.items(), columns=["Experto", "Peso"])
                .sort_values("Peso", ascending=False).style.format({"Peso": "{:.2%}"}),
                use_container_width=True, hide_index=True
            )

# --- 5. COMPONENTES UI (ONBOARDING & CHAT) ---

def render_sidebar(df_profiles):
    with st.sidebar:
        st.title("🎛️ Configuración")
        mode = st.radio("Modo", ["Demo", "Perfil Nuevo"], key="profile_mode")
        
        if mode == "Demo":
            if not df_profiles.empty:
                ids = df_profiles.index.tolist()
                selected = st.selectbox("Perfil Demo", ids, key="selected_profile_id")
                if st.button("Cargar Perfil Demo"):
                    st.session_state.current_user_profile = df_profiles.loc[selected]
                    st.session_state.current_archetype = "Cargando..." 
                    st.session_state.messages = []
                    st.rerun()
            else:
                st.warning("Sin perfiles demo.")
        
        st.divider()
        st.metric("Interacciones", st.session_state.session_data["metrics"]["total_interactions"])
        if st.button("🗑️ Reiniciar"):
            st.session_state.messages = []
            st.session_state.session_data = get_initial_session_data()
            st.rerun()

def render_onboarding_form(moe_system):
    st.header("📝 Nuevo Perfil")
    with st.form("onboarding"):
        c1, c2 = st.columns(2)
        edad = c1.selectbox("Rango Etario", ["14-39", "40-64", "65+"])
        educacion = c2.selectbox("Nivel Educativo", ["Primario", "Secundario", "Terciario/Univ. Incompleto", "Terciario/Univ. Completo"])
        c3, c4 = st.columns(2)
        ocupacion = c3.selectbox("Situación Laboral", ["Ocupado Formal", "Ocupado Informal", "Desocupado", "Inactivo"])
        discapacidad = c4.selectbox("Tipo Discapacidad", ["Motora", "Visual", "Auditiva", "Mental", "Habla", "Visceral"])
        cud = st.radio("¿Posee CUD?", ["Sí", "No", "En trámite"], horizontal=True)
        
        if st.form_submit_button("🚀 Crear e Iniciar"):
            edu_map = {"Terciario/Univ. Completo": 5, "Terciario/Univ. Incompleto": 4, "Secundario": 3, "Primario": 1}
            edad_map = {"14-39": 3, "40-64": 4, "65+": 5}
            raw_data = {
                "edad_agrupada": edad_map.get(edad, 3), "MNEA": edu_map.get(educacion, 3),
                "Estado_ocup": 2 if ocupacion == "Desocupado" else 1, "dificultad_total": 1,
                "tipo_dificultad": 6 if discapacidad == "Habla" else 1, "dificultades": 1,
                "certificado": 1 if cud == "Sí" else 2, "PC08": 9, "pc03": 9, "tipo_hogar": 9
            }
            with st.spinner("Infiriendo perfil..."):
                profile = infer_profile_features(raw_data)
                # Intento de guardado (opcional)
                try:
                    log_dir = st.session_state.config['data_paths']['onboarding_log_dir']
                    if log_dir:
                        with open(os.path.join(log_dir, f"user_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w') as f:
                            json.dump(profile.astype(str).to_dict(), f, indent=2)
                except: pass
                st.session_state.current_user_profile = profile
                st.session_state.current_archetype = "Perfil Generado"
                st.session_state.messages = []
                st.success("Perfil creado."); st.rerun()

def render_chat(emotion_clf, moe_system):
    st.subheader(f"💬 Chat ({st.session_state.current_archetype})")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "analysis" in msg:
                render_adaptive_logic_expander(msg["analysis"], st.session_state.config)

    if prompt := st.chat_input("Escribe tu consulta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    emo_probs = emotion_clf.predict_proba(prompt)[0]
                    top_emo = max(emo_probs, key=emo_probs.get)
                    
                    # Llamada al Backend
                    plan_result = moe_system.get_cognitive_plan(
                        user_profile=st.session_state.current_user_profile,
                        emotion_probs=emo_probs,
                        conversation_context=st.session_state.session_data["conversation_context"],
                        config=st.session_state.config,
                        user_prompt=prompt
                    )

                    # --- LÓGICA ROBUSTA DE DESEMPAQUETADO ---
                    xai_meta, final_weights = {}, {}
                    plan, archetype = "Error", "Desconocido"
                    
                    if isinstance(plan_result, tuple):
                        n = len(plan_result)
                        if n == 4:
                            plan, archetype, final_weights, xai_meta = plan_result
                        elif n == 3:
                            plan, archetype, final_weights = plan_result
                        elif n >= 2:
                            plan, archetype = plan_result[0], plan_result[1]
                    else:
                        plan = str(plan_result)

                    st.markdown(plan)
                    
                    analysis_data = {
                        "archetype": archetype, "top_emotion": top_emo,
                        "top_emotion_prob": emo_probs[top_emo], "emotion_probs": emo_probs,
                        "final_weights": final_weights, "xai_metadata": xai_meta
                    }
                    st.session_state.messages.append({"role": "assistant", "content": plan, "analysis": analysis_data})
                    st.session_state.current_archetype = archetype
                    update_metrics(analysis_data)
                    render_adaptive_logic_expander(analysis_data, st.session_state.config)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

# --- MAIN ---
def main():
    try:
        emo_clf, moe_sys, df_prof, config = load_all_models_and_data()
        initialize_session_state(df_prof, config)
        render_sidebar(df_prof)
        if st.session_state.current_user_profile is not None:
            render_chat(emo_clf, moe_sys)
        elif st.session_state.profile_mode == "Perfil Nuevo":
            render_onboarding_form(moe_sys)
        else:
            st.info("👈 Selecciona un perfil Demo para comenzar.")
    except Exception as e:
        st.error("Error fatal."); st.exception(e)

if __name__ == "__main__":
    main()
