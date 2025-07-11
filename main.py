# main.py 

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

def main(config):
    """
    Función principal que orquesta todo el pipeline, con carga/guardado de modelos.
    """
    print("\n--- 🚀 INICIANDO PIPELINE FINAL INTEGRADO Y EVALUACIÓN DOCTORAL ---")

    # --- Parte I: Carga o Entrenamiento del Clasificador de Emociones ---
    emotion_model_path = config['model_paths']['emotion_classifier']
    if os.path.exists(emotion_model_path):
        print(f"--- [PARTE I] Cargando clasificador de emociones desde: {emotion_model_path} ---")
        model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
        tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
        emotion_classifier = EmotionClassifier(model, tokenizer)
    else:
        print(f"--- [PARTE I] No se encontró modelo de emociones. Procediendo al entrenamiento... ---")
        emotion_classifier = train_and_evaluate_emotion_classifier(config)

    # --- Parte II: Carga o Entrenamiento del Tutor Cognitivo ---
    cognitive_model_path = config['model_paths']['cognitive_tutor']
    cognitive_tutor_ready = False
    
    # Cargar los datos siempre, ya que son necesarios para la demostración
    try:
        drive.mount('/content/drive', force_remount=True)
        df_raw = pd.read_csv(config['data_paths']['endis_raw'], delimiter=';', low_memory=False, index_col='ID')
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_archetyped)
        feature_columns = [col for col in df_fuzzified.columns if '_memb' in col]
    except Exception as e:
        print(f"❌ ERROR CRÍTICO AL CARGAR Y PROCESAR DATOS DE ENDIS: {e}")
        return # Terminar la ejecución si los datos base no se pueden cargar

    if os.path.exists(cognitive_model_path):
        print(f"\n--- [PARTE II] Cargando tutor cognitivo desde: {cognitive_model_path} ---")
        cognitive_model = joblib.load(cognitive_model_path)
        cognitive_tutor_ready = True
    else:
        print(f"\n--- [PARTE II] No se encontró modelo de tutor. Procediendo al entrenamiento... ---")
        try:
            # Re-procesar los arquetipos para el entrenamiento
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
                _, X_test_cog, _, y_test_cog = train_test_split(X_cognitive, y_cognitive, test_size=config['model_params']['cognitive_tutor']['test_size'], random_state=config['model_params']['cognitive_tutor']['random_state'], stratify=y_cognitive)
                
                print("\n--- Aplicando SMOTE para balancear el conjunto de entrenamiento... ---")
                smote = SMOTE(random_state=config['model_params']['cognitive_tutor']['random_state'])
                X_train_sm, y_train_sm = smote.fit_resample(X_cognitive, y_cognitive) # Usamos todos los datos para entrenar el modelo final
                
                cognitive_model = RandomForestClassifier(
                    n_estimators=config['model_params']['cognitive_tutor']['n_estimators'],
                    max_depth=config['model_params']['cognitive_tutor']['max_depth'],
                    random_state=config['model_params']['cognitive_tutor']['random_state']
                )
                cognitive_model.fit(X_train_sm, y_train_sm)
                print(f"--- ✅ Tutor Cognitivo REAL ({type(cognitive_model).__name__}) Entrenado. ---")
                
                print(f"\n  › Guardando el modelo de tutor en: {cognitive_model_path}")
                os.makedirs(os.path.dirname(cognitive_model_path), exist_ok=True)
                joblib.dump(cognitive_model, cognitive_model_path)
                print("  › Modelo guardado exitosamente.")
                cognitive_tutor_ready = True
            else:
                raise ValueError("No hay suficientes datos después del filtrado para entrenar.")
        except Exception as e:
            print(f"❌ ERROR AL ENTRENAR TUTOR COGNITIVO: {e}")
            traceback.print_exc()


    # --- Parte IV: Demostración del Sistema Integrado ---
    print("\n\n--- 🏁 [PARTE IV] Demostración del Sistema Integrado ---")
    if cognitive_tutor_ready:
        cognitive_tutor_system = MoESystem(cognitive_model, feature_columns, config['affective_rules'])
        
        def get_integrated_response(text: str, user_id: int):
            print("\n" + "="*80); print(f"INPUT DEL USUARIO (ID: {user_id}): '{text}'")
            emotion_probs = emotion_classifier.predict_proba(text)
            top_emotion = max(emotion_probs, key=emotion_probs.get)

            print(f"🧠 Emoción Dominante Detectada: {top_emotion} (Confianza: {emotion_probs[top_emotion]:.0%})")
            print(f"   (Espectro completo: {[f'{e}: {p:.0%}' for e, p in sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True) if p > 0.05]})")
            print("📖 Plan Cognitivo Generado:")
            
            try:
                user_profile = df_fuzzified.loc[user_id]
                cognitive_plan = cognitive_tutor_system.get_cognitive_plan(user_profile, emotion_probs)
            except Exception as e: 
                cognitive_plan = f"[Sistema]: Error al generar el plan cognitivo: {e}"
            
            print("\n✨ **Respuesta Integrada y Afectiva:**")
            if top_emotion in ["Ira", "Tristeza", "Miedo"]: print(f"Percibo que puedes sentirte con un poco de {top_emotion.lower()}. Revisemos esto juntos para encontrar una solución. Aquí tienes un plan de acción:")
            elif top_emotion in ["Anticipación", "Alegría", "Confianza"]: print("¡Excelente! Percibo un estado de {top_emotion.lower()}. Para potenciar ese impulso, este es el plan que te sugiero:")
            else: print("Entendido. En base a tu consulta, este es el plan de acción sugerido:")
            print(cognitive_plan); print("="*80)

        demo_user_id_1 = 35906 
        demo_user_id_2 = 77570
        get_integrated_response(text="¡Es una vergüenza, llevo meses esperando y no me dan respuesta!", user_id=demo_user_id_1)
        get_integrated_response(text="No entiendo bien qué es la ley 22.431 pero gracias por la info, me da esperanza.", user_id=demo_user_id_2)
    else:
        print("--- ⚠️ La demostración se omite porque el tutor cognitivo no se pudo entrenar o cargar. ---")


if __name__ == '__main__':
    # Cargar la configuración desde el archivo YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Importar PyYAML para leer la config
    !pip install PyYAML -q
    import yaml
    
    main(config)
