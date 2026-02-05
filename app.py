"""
Aplicación principal de Streamlit para el prototipo de Tesis Doctoral:
Tutor Cognitivo Adaptativo con IA Afectiva y Arquitectura Neuro-Simbólica.

Características:
- Interfaz de Chat con Gating Afectivo.
- Visualización de Auditoría XAI (Explainable AI) en tiempo real.
- Gestión de Perfiles de Usuario (Onboarding y Demo).
- Integración robusta con backend MoE (4 valores).
- Branding Académico.
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

# Configuración de página (Debe ser lo primero)
st.set_page_config(
    page_title="Tutor Cognitivo - Tesis Doctoral",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para apariencia académica/profesional
st.markdown("""
<style>
    .header-academic { font-size: 14px; color: #666; margin-bottom: 20px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .success-box { padding: 10px; background-color: #e8f5e9; border-radius: 5px; border-left: 5px solid #4caf50; }
    .alert-box { padding: 10px; background-color: #ffebee; border-radius: 5px; border-left: 5px solid #f44336; }
</style>
""", unsafe_allow_html=True)

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
    """Carga modelos, configuraciones y perfiles demo."""
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
                st.warning("Archivo de perfiles demo no encontrado.")
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

# --- 3. GESTIÓN DE SESIÓN Y ESTADO (UX MEJORADO) ---

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

def clear_session_state():
    """Resetea la conversación y el perfil activo al cambiar de modo."""
    st.session_state.messages = []
    st.session_state.session_data = get_initial_session_data()
    st.session_state.current_user_profile = None
    st.session_state.current_archetype = "N/A"
    # No reseteamos 'profile_mode' ni 'selected_profile_id' aquí porque son widgets

def initialize_session_state(df_profiles: pd.DataFrame, config: Dict):
    if "messages" not in st.session_state: st.session_state.messages = []
    if "session_data" not in st.session_state: st.session_state.session_data = get_initial_session_data()
    if "config" not in st.session_state: st.session_state.config = config
    
    if "current_user_profile" not in st.session_state: st.session_state.current_user_profile = None
    if "current_archetype" not in st.session_state: st.session_state.current_archetype = "N/A"
    
    # Defaults de UI
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

    xai = analysis_data.get('xai_metadata', {})
    metrics = xai.get('execution_metrics', {})
    veto = metrics.get('veto_applied', False)
    
    label = "🔍 Auditoría del Sistema (XAI & MoE)"
    if veto: label += " 🛡️ GUARDRAIL ACTIVO"

    with st.expander(label, expanded=veto):
        c1, c2 = st.columns([1, 1], gap="medium")
        
        with c1:
            st.markdown("#### 🧡 Estado Afectivo")
            emo = analysis_data.get('top_emotion', 'N/A')
            prob = analysis_data.get('top_emotion_prob', 0.0)
            if emo in ["Ira", "Tristeza", "Miedo"]:
                st.warning(f"**{emo}** ({prob:.1%})")
            else:
                st.info(f"**{emo}** ({prob:.1%})")
            
            all_probs = analysis_data.get('emotion_probs', {})
            if all_probs:
                clean = {k:v for k,v in all_probs.items() if v > 0.05}
                st.bar_chart(clean, height=150, color="#FF4B4B" if emo in ["Ira", "Miedo"] else "#4B8BFF")

        with c2:
            st.markdown("#### 🧠 Decisión Cognitiva")
            raw_arch = metrics.get('raw_prediction', 'N/A')
            final_arch = metrics.get('selected_archetype', 'N/A')
            
            if veto:
                st.markdown('<div class="alert-box">🛡️ <b>Veto Simbólico Aplicado</b></div>', unsafe_allow_html=True)
                st.markdown(f"**Neuronal:** `{raw_arch}` ➔ **Final:** `{final_arch}`")
                st.caption(f"**Regla:** {xai.get('guardrail_reason', {}).get('rule_id', 'N/A')}")
            else:
                st.markdown('<div class="success-box">✅ <b>Predicción Validada</b></div>', unsafe_allow_html=True)
                st.markdown(f"**Arquetipo:** `{final_arch}`")

        st.divider()
        st.caption("Métricas Técnicas del Motor de Inferencia")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Búsqueda", str(metrics.get('expert_search_mode', 'N/A')))
        m2.metric("Masa Efectiva", f"{metrics.get('moe_effective_mass', 0):.2f}")
        m3.metric("Congruencia", str(metrics.get('affective_congruence', 'N/A')))
        m4.metric("Veto Mass", f"{metrics.get('moe_vetoed_mass', 0):.2f}")

        weights = analysis_data.get('final_weights', {})
        if weights:
            with st.expander("Ver Pesos de Expertos"):
                active = {k: v for k, v in weights.items() if v > 0.01}
                st.dataframe(pd.DataFrame(active.items(), columns=["Experto", "Peso"]).sort_values("Peso", ascending=False).style.format({"Peso": "{:.2%}"}), use_container_width=True, hide_index=True)

# --- 5. COMPONENTES UI (ONBOARDING & CHAT) ---

def render_sidebar(df_profiles):
    with st.sidebar:
        st.markdown("**Tesis Doctoral en Informática**")
        st.markdown("*Mgter. Ing. Yanina A. Caffetti*")
        st.divider()
        
        st.subheader("🎛️ Configuración")
        
        # Selector de Modo con Callback de Limpieza (FIX UX)
        mode = st.radio(
            "Modo de Operación", 
            ["Demo", "Perfil Nuevo"], 
            key="profile_mode",
            on_change=clear_session_state # <--- CLAVE PARA QUE CARGUE EL FORMULARIO
        )
        
        if mode == "Demo":
            if not df_profiles.empty:
                ids = df_profiles.index.tolist()
                st.selectbox("Perfil Demo", ids, key="selected_profile_id", on_change=clear_session_state)
                
                if st.button("🔄 Cargar/Reiniciar Demo", type="primary"):
                    st.session_state.current_user_profile = df_profiles.loc[st.session_state.selected_profile_id]
                    st.session_state.current_archetype = "Cargando..."
                    st.session_state.messages = []
                    st.rerun()
            else:
                st.warning("Sin datos demo.")
        
        st.divider()
        st.metric("Interacciones Sesión", st.session_state.session_data["metrics"]["total_interactions"])

def render_onboarding_form(moe_system):
    st.markdown("### 📝 Generación de Nuevo Perfil")
    st.info("Ingrese los datos sociodemográficos para simular un nuevo usuario y testear la inferencia del arquetipo.")
    
    with st.form("onboarding"):
        c1, c2 = st.columns(2)
        edad = c1.selectbox("Rango Etario", ["14-39", "40-64", "65+"])
        educacion = c2.selectbox("Nivel Educativo", ["Primario", "Secundario", "Terciario/Univ. Incompleto", "Terciario/Univ. Completo"])
        
        c3, c4 = st.columns(2)
        ocupacion = c3.selectbox("Situación Laboral", ["Ocupado Formal", "Ocupado Informal", "Desocupado", "Inactivo"])
        discapacidad = c4.selectbox("Tipo Discapacidad", ["Motora", "Visual", "Auditiva", "Mental", "Habla", "Visceral"])
        
        cud = st.radio("¿Posee Certificado Único de Discapacidad (CUD)?", ["Sí", "No", "En trámite"], horizontal=True)
        
        if st.form_submit_button("🚀 Generar Perfil e Iniciar", type="primary"):
            # Mapeo de UI a Datos (Coherente con modelo)
            edu_map = {"Terciario/Univ. Completo": 5, "Terciario/Univ. Incompleto": 4, "Secundario": 3, "Primario": 1}
            edad_map = {"14-39": 3, "40-64": 4, "65+": 5}
            
            raw_data = {
                "edad_agrupada": edad_map.get(edad, 3),
                "MNEA": edu_map.get(educacion, 3),
                "Estado_ocup": 2 if ocupacion == "Desocupado" else 1,
                "dificultad_total": 1,
                "tipo_dificultad": 6 if discapacidad == "Habla" else 1,
                "dificultades": 1,
                "certificado": 1 if cud == "Sí" else 2,
                "PC08": 9, "pc03": 9, "tipo_hogar": 9
            }
            
            with st.spinner("Ejecutando inferencia de perfil..."):
                profile = infer_profile_features(raw_data)
                
                # Guardar log si existe directorio
                try:
                    log_dir = st.session_state.config['data_paths']['onboarding_log_dir']
                    if log_dir:
                        fname = f"user_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(os.path.join(log_dir, fname), 'w') as f:
                            json.dump(profile.astype(str).to_dict(), f, indent=2)
                except: pass

                st.session_state.current_user_profile = profile
                st.session_state.current_archetype = "Perfil Generado"
                st.session_state.messages = []
                st.success("Perfil creado correctamente.")
                st.rerun()

def render_chat(emotion_clf, moe_system):
    st.markdown(f"### 💬 Chat con Tutor ({st.session_state.current_archetype})")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "analysis" in msg:
                render_adaptive_logic_expander(msg["analysis"], st.session_state.config)

    if prompt := st.chat_input("Consulte sobre derechos, trámites o empleo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Procesando semántica y afectividad..."):
                try:
                    emo_probs = emotion_clf.predict_proba(prompt)[0]
                    top_emo = max(emo_probs, key=emo_probs.get)
                    
                    # LLAMADA AL BACKEND (Soporta 4 valores)
                    plan_result = moe_system.get_cognitive_plan(
                        user_profile=st.session_state.current_user_profile,
                        emotion_probs=emo_probs,
                        conversation_context=st.session_state.session_data["conversation_context"],
                        config=st.session_state.config,
                        user_prompt=prompt
                    )

                    # Desempaquetado seguro
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
                    st.error("Ocurrió un error en el procesamiento.")
                    st.exception(e)

# --- MAIN ---
def main():
    try:
        emo_clf, moe_sys, df_prof, config = load_all_models_and_data()
        initialize_session_state(df_prof, config)
        render_sidebar(df_prof)
        
        st.markdown("<div class='header-academic'>Prototipo de Tesis Doctoral: Sistema Neuro-Simbólico de Tutoría Cognitiva</div>", unsafe_allow_html=True)

        if st.session_state.profile_mode == "Perfil Nuevo":
            # Si hay perfil cargado tras onboarding, mostrar chat, si no, formulario
            if st.session_state.current_user_profile is not None:
                render_chat(emo_clf, moe_sys)
            else:
                render_onboarding_form(moe_sys)
        
        elif st.session_state.profile_mode == "Demo":
            if st.session_state.current_user_profile is not None:
                render_chat(emo_clf, moe_sys)
            else:
                st.info("👈 Seleccione un **Perfil Demo** en el menú lateral y haga clic en **Cargar** para iniciar.")
                
    except Exception as e:
        st.error("Error fatal en la aplicación.")
        st.exception(e)

if __name__ == "__main__":
    main()
