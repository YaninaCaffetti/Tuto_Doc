# app.py

import streamlit as st
import yaml
import sys
import os


# Añadir la carpeta src al path para poder importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from main import load_models_and_data, get_full_response

def main():
    """
    Función principal que define la interfaz de usuario de la aplicación Streamlit.
    """
    st.set_page_config(page_title="Tutor Cognitivo Adaptativo", layout="wide")

    st.title("🧠 Tutor Cognitivo Adaptativo con IA Afectiva 🤖")
    st.markdown("*"Proyecto de Tesis Doctoral de Mgter. Ing. Yanina A. Caffetti*"")
    st.markdown("---")

    # Cargar la configuración y los modelos una sola vez usando el caché de Streamlit
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Esta función cargará los modelos desde el disco o los entrenará si no existen
        emotion_classifier, cognitive_tutor_system, df_fuzzified = load_models_and_data(config)
        st.success("✅ ¡Sistema y modelos cargados exitosamente!")

    except Exception as e:
        st.error(f"❌ Error crítico al inicializar el sistema: {e}")
        st.stop()


    # --- Interfaz de Usuario ---
    st.subheader("Simulador de Interacción con el Tutor")

    # Usar IDs de usuario de la demo como ejemplo
    user_ids = [35906, 77570]
    selected_id = st.selectbox("Seleccione un ID de Usuario para la demostración:", user_ids)

    user_input = st.text_area(
        "Escriba la consulta del usuario aquí:", 
        "No entiendo bien qué es la ley 22.431 pero gracias por la info, me da esperanza.",
        height=100
    )

    if st.button("Generar Respuesta del Tutor"):
        if user_input and selected_id:
            with st.spinner("Procesando... El tutor está analizando la respuesta..."):
                # Llamar a la función que genera la respuesta completa
                response_dict = get_full_response(
                    text=user_input,
                    user_id=selected_id,
                    emotion_classifier=emotion_classifier,
                    cognitive_tutor_system=cognitive_tutor_system,
                    df_fuzzified=df_fuzzified
                )
            
            st.markdown("---")
            st.subheader("Análisis y Plan de Acción Generado")

            # Mostrar los resultados de forma clara
            st.info(f"**🧠 Emoción Dominante Detectada:** {response_dict['top_emotion']} (Confianza: {response_dict['confidence']:.0%})")
            st.text(f"Espectro completo: {response_dict['emotion_spectrum']}")
            
            st.markdown("---")
            st.markdown("##### ✨ **Respuesta Integrada y Afectiva:**")
            st.markdown(response_dict['intro_message'])
            # Usamos st.markdown para que los saltos de línea y el formato se rendericen correctamente
            st.markdown(response_dict['cognitive_plan'].replace('  - ', '* '))

        else:
            st.warning("Por favor, ingrese un texto y seleccione un ID de usuario.")

if __name__ == '__main__':
    main()
