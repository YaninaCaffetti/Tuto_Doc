# main.py (Versión 26.0 - Refactorizado para Streamlit)

import pandas as pd
from collections import Counter
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import traceback
import yaml
import os
import joblib

# Importar nuestros módulos locales
from src.data_processing import run_feature_engineering, run_archetype_engineering, run_fuzzification
from src.emotion_classifier import train_and_evaluate_emotion_classifier, EmotionClassifier
from src.cognitive_tutor import MoESystem
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_models_and_data(config):
    """
    Carga los modelos y datos necesarios. Si los modelos no existen, los entrena y guarda.
    """
    # --- Carga o Entrenamiento del Clasificador de Emociones ---
    emotion_model_path = config['model_paths']['emotion_classifier']
    if os.path.exists(emotion_model_path):
        print(f"--- [PARTE I] Cargando clasificador de emociones desde: {emotion_model_path} ---")
        model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
        tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
        emotion_classifier = EmotionClassifier(model, tokenizer)
    else:
        print(f"--- [PARTE I] No se encontró modelo de emociones. Procediendo al entrenamiento... ---")
        emotion_classifier = train_and_evaluate_emotion_classifier(config)

    # --- Carga o Entrenamiento del Tutor Cognitivo ---
    cognitive_model_path = config['model_paths']['cognitive_tutor']
    
    # Cargar los datos de ENDIS siempre, son necesarios para la demo
    try:
        drive.mount('/content/drive', force_remount=True)
        df_raw = pd.read_csv(config['data_paths']['endis_raw'], delimiter=';', low_memory=False, index_col='ID')
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_archetyped)
        feature_columns = [col for col in df_fuzzified.columns if '_memb' in col]
    except Exception as e:
        raise Exception(f"❌ ERROR CRÍTICO AL CARGAR Y PROCESAR DATOS DE ENDIS: {e}")

    if os.path.exists(cognitive_model_path):
        print(f"\n--- [PARTE II] Cargando tutor cognitivo desde: {cognitive_model_path} ---")
        cognitive_model = joblib.load(cognitive_model_path)
    else:
        print(f"\n--- [PARTE II] No se encontró modelo de tutor. Procediendo al entrenamiento... ---")
        cfg_cog = config['model_params']['cognitive_tutor']
        
        pertenencia_cols = {col: col.replace('_v6', '').replace('_v3', '').replace('_v2', '').replace('_v1', '') for col in df_fuzzified.columns if 'Pertenencia_' in col}
        df_fuzzified.rename(columns=pertenencia_cols, inplace=True)
        columnas_arquetipos = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
        
        def determinar_arquetipo_predominante(row):
            pertenencias = row[columnas_arquetipos];
            if pertenencias.empty or len(pertenencias.dropna()) == 0 or pertenencias.max() < config['constants']['umbrales']['arquetipo']: return 'Arquetipo_No_Predominante'
            return pertenencias.idxmax().replace('Pertenencia_', '')
        
        df_fuzzified['Arquetipo_Predominante'] = df_fuzzified.apply(determinar_arquetipo_predominante, axis=1)
        df_entrenamiento = df_fuzzified[df_fuzzified['Arquetipo_Predominante'] != 'Arquetipo_No_Predominante'].copy()

        if len(df_entrenamiento) > 10:
            X_cognitive, y_cognitive = df_entrenamiento[feature_columns], df_entrenamiento['Arquetipo_Predominante']
            
            print("\n--- Aplicando SMOTE para balancear el conjunto de entrenamiento... ---")
            smote = SMOTE(random_state=cfg_cog['random_state'])
            X_train_sm, y_train_sm = smote.fit_resample(X_cognitive, y_cognitive)
            
            cognitive_model = RandomForestClassifier(
                n_estimators=cfg_cog['n_estimators'],
                max_depth=cfg_cog['max_depth'],
                random_state=cfg_cog['random_state']
            )
            cognitive_model.fit(X_train_sm, y_train_sm)
            print(f"--- ✅ Tutor Cognitivo REAL ({type(cognitive_model).__name__}) Entrenado. ---")
            
            print(f"\n  › Guardando el modelo de tutor en: {cognitive_model_path}")
            os.makedirs(os.path.dirname(cognitive_model_path), exist_ok=True)
            joblib.dump(cognitive_model, cognitive_model_path)
            print("  › Modelo guardado exitosamente.")
        else:
            raise ValueError("No hay suficientes datos para entrenar el tutor cognitivo.")

    cognitive_tutor_system = MoESystem(cognitive_model, feature_columns, config['affective_rules'])
    
    return emotion_classifier, cognitive_tutor_system, df_fuzzified


def get_full_response(text, user_id, emotion_classifier, cognitive_tutor_system, df_fuzzified):
    """
    Genera una respuesta completa del tutor, incluyendo análisis de emoción y plan cognitivo.
    """
    emotion_probs = emotion_classifier.predict_proba(text)
    top_emotion = max(emotion_probs, key=emotion_probs.get)

    try:
        user_profile = df_fuzzified.loc[user_id]
        cognitive_plan = cognitive_tutor_system.get_cognitive_plan(user_profile, emotion_probs)
    except Exception as e:
        cognitive_plan = f"[Sistema]: Error al generar el plan cognitivo: {e}"

    if top_emotion in ["Ira", "Tristeza", "Miedo"]:
        intro_message = f"Percibo que puedes sentirte con un poco de {top_emotion.lower()}. Revisemos esto juntos para encontrar una solución. Aquí tienes un plan de acción:"
    elif top_emotion in ["Anticipación", "Alegría", "Confianza"]:
        intro_message = f"¡Excelente! Percibo un estado de {top_emotion.lower()}. Para potenciar ese impulso, este es el plan que te sugiero:"
    else:
        intro_message = "Entendido. En base a tu consulta, este es el plan de acción sugerido:"
        
    return {
        "top_emotion": top_emotion,
        "confidence": emotion_probs[top_emotion],
        "emotion_spectrum": [f'{e}: {p:.0%}' for e, p in sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True) if p > 0.05],
        "cognitive_plan": cognitive_plan,
        "intro_message": intro_message
    }


if __name__ == '__main__':
    # Este bloque solo se ejecuta si corres 'python3 main.py' directamente.
    # Es útil para una prueba rápida en consola.
    
    print("Ejecutando main.py como script independiente para una demostración en consola...")
    
    # 1. Cargar configuración
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Cargar modelos y datos
    try:
        emotion_classifier, cognitive_tutor_system, df_fuzzified = load_models_and_data(config)
        
        # 3. Demostración
        demo_user_id = 35906
        demo_text = "¡Es una vergüenza, llevo meses esperando y no me dan respuesta!"
        
        response = get_full_response(
            demo_text,
            demo_user_id,
            emotion_classifier,
            cognitive_tutor_system,
            df_fuzzified
        )
        
        print("\n" + "="*80)
        print(f"INPUT: '{demo_text}' (Usuario: {demo_user_id})")
        print(f"OUTPUT: \n{response['intro_message']}\n{response['cognitive_plan']}")
        print("="*80)

    except Exception as e:
        print(f"❌ Ocurrió un error en la ejecución principal: {e}")
        traceback.print_exc()
