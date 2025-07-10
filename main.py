# main.py

import pandas as pd
from collections import Counter
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import traceback

# Importar nuestros módulos locales
from src.data_processing import run_feature_engineering, run_archetype_engineering, run_fuzzification
from src.emotion_classifier import train_and_evaluate_emotion_classifier
from src.cognitive_tutor import MoESystem

# --- CONFIGURACIÓN GLOBAL ---
RAW_DATA_PATH = '/content/drive/My Drive/Tesis Doctoral - CAFFETTI/base_estudio_discapacidad_2018.csv'
UMBRAL_MINIMO_ARQUETIPO=0.15; TEST_SIZE=0.3; RANDOM_STATE=42
EMOTION_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"; EMOTION_LABELS = ['Ira', 'Miedo', 'Tristeza', 'Alegría', 'Confianza', 'Anticipación', 'Sorpresa', 'Neutral']
label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}; id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
AFFECTIVE_MODULATION_RULES = {
    'Tristeza': { 'Potencial_Latente': 1.5, 'Cand_Nec_Sig': 1.2, },
    'Miedo': { 'Potencial_Latente': 1.4, 'Cand_Nec_Sig': 1.3, 'Prof_Subutil': 0.8, },
    'Ira': { 'Prof_Subutil': 1.4, 'GestorCUD': 1.5, 'Potencial_Latente': 0.7, },
    'Alegría': { 'Nav_Informal': 1.3, 'Joven_Transicion': 1.3, 'Prof_Subutil': 1.2, },
    'Confianza': { 'Prof_Subutil': 1.4, 'Nav_Informal': 1.2, 'Joven_Transicion': 1.2, }
}

def main():
    print("\n--- 🚀 INICIANDO PIPELINE FINAL INTEGRADO Y EVALUACIÓN DOCTORAL ---")

    # --- Parte I: Entrenamiento Avanzado del Clasificador de Emociones ---
    emotion_classifier = train_and_evaluate_emotion_classifier(EMOTION_LABELS, label2id, EMOTION_MODEL_NAME, RANDOM_STATE)

    # --- Parte II: Entrenamiento del Tutor Cognitivo ---
    print("\n--- [PARTE II] Entrenando el Tutor Cognitivo... ---")
    cognitive_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=10)
    print(f"Modelo cognitivo principal seleccionado: {type(cognitive_model).__name__}")
    df_featured, df_fuzzified, feature_columns = None, None, []
    cognitive_tutor_ready = False

    try:
        drive.mount('/content/drive', force_remount=True)
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', low_memory=False, index_col='ID')
        print(f"  › Dataset ENDIS cargado: {df_raw.shape[0]} filas.")
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_archetyped)
        pertenencia_cols = {col: col.replace('_v6', '').replace('_v3', '').replace('_v2', '').replace('_v1', '') for col in df_fuzzified.columns if 'Pertenencia_' in col}
        df_fuzzified.rename(columns=pertenencia_cols, inplace=True)
        columnas_arquetipos = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
        def determinar_arquetipo_predominante(row):
            pertenencias = row[columnas_arquetipos];
            if pertenencias.empty or len(pertenencias.dropna()) == 0 or pertenencias.max() < UMBRAL_MINIMO_ARQUETIPO: return 'Arquetipo_No_Predominante'
            return pertenencias.idxmax().replace('Pertenencia_', '')
        df_fuzzified['Arquetipo_Predominante'] = df_fuzzified.apply(determinar_arquetipo_predominante, axis=1)
        feature_columns = [col for col in df_fuzzified.columns if '_memb' in col]
        df_entrenamiento = df_fuzzified[df_fuzzified['Arquetipo_Predominante'] != 'Arquetipo_No_Predominante'].copy()

        if len(df_entrenamiento) > 10:
            X_cognitive, y_cognitive = df_entrenamiento[feature_columns], df_entrenamiento['Arquetipo_Predominante']
            X_train_cog, X_test_cog, y_train_cog, y_test_cog = train_test_split(X_cognitive, y_cognitive, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cognitive)
            
            print("\n--- Aplicando SMOTE para balancear el conjunto de entrenamiento... ---")
            print(f"Distribución de clases ANTES de SMOTE: {Counter(y_train_cog)}")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_sm, y_train_sm = smote.fit_resample(X_train_cog, y_train_cog)
            print(f"Distribución de clases DESPUÉS de SMOTE: {Counter(y_train_sm)}")
            
            cognitive_model.fit(X_train_sm, y_train_sm)
            print(f"--- ✅ Tutor Cognitivo REAL ({type(cognitive_model).__name__}) Entrenado con datos balanceados por SMOTE. ---")
            cognitive_tutor_ready = True 
        else:
            raise ValueError("No hay suficientes datos después del filtrado para entrenar el tutor cognitivo.")

    except Exception as e:
        print(f"❌ ERROR AL ENTRENAR TUTOR COGNITIVO: {e}")
        cognitive_model = None
        df_fuzzified = pd.DataFrame(index=[35906, 77570]); df_fuzzified['TIENE_CUD'] = ['No_Tiene_CUD', 'Si_Tiene_CUD']

    # --- Parte III: Evaluación del Modelo Cognitivo Final ---
    print("\n\n--- 📊 [PARTE III] Evaluación del Modelo Cognitivo Final ---")
    if cognitive_tutor_ready:
        print(f"\n--- 1. Evaluación del modelo final: {type(cognitive_model).__name__} (entrenado con SMOTE) ---")
        y_pred_final = cognitive_model.predict(X_test_cog)
        print(f"  › Reporte de Clasificación ({type(cognitive_model).__name__} con SMOTE):")
        print(classification_report(y_test_cog, y_pred_final, zero_division=0))
    else:
        print("--- ⚠️ Se omitió la evaluación del Tutor Cognitivo porque no se pudo entrenar. ---")

    # --- Parte IV: Demostración del Sistema Integrado ---
    print("\n\n--- 🏁 [PARTE IV] Demostración del Sistema Integrado ---")
    cognitive_tutor_system = MoESystem(cognitive_model, feature_columns) if cognitive_tutor_ready else None
    
    def get_integrated_response(text: str, user_id: int):
        print("\n" + "="*80); print(f"INPUT DEL USUARIO (ID: {user_id}): '{text}'")
        predicted_emotion = emotion_classifier.predict(text); print(f"🧠 Emoción Detectada: {predicted_emotion}"); print("📖 Plan Cognitivo Generado:")
        
        if not cognitive_tutor_system:
            cognitive_plan = "[Sistema]: Tutor cognitivo no disponible (maqueta)."
        else:
            try:
                user_profile = df_fuzzified.loc[user_id]
                cognitive_plan = cognitive_tutor_system.get_cognitive_plan(user_profile, predicted_emotion)
            except Exception as e: 
                cognitive_plan = f"[Sistema]: Error al generar el plan cognitivo: {e}"
        
        print("\n✨ **Respuesta Integrada y Afectiva:**")
        if predicted_emotion in ["Ira", "Tristeza", "Miedo"]: print(f"Entiendo que te sientas así ({predicted_emotion.lower()}). Revisemos esto juntos para encontrar una solución. Aquí tienes un plan de acción:")
        elif predicted_emotion in ["Anticipación", "Alegría", "Confianza"]: print("¡Excelente! Me alegra que lo veas de esa manera. Para potenciar ese impulso, este es el plan que te sugiero:")
        else: print("Entendido. En base a tu consulta, este es el plan de acción sugerido:")
        print(cognitive_plan); print("="*80)

    demo_user_id_1 = 35906 
    demo_user_id_2 = 77570
    get_integrated_response(text="¡Es una vergüenza, llevo meses esperando y no me dan respuesta!", user_id=demo_user_id_1)
    get_integrated_response(text="No entiendo bien qué es la ley 22.431 pero gracias por la info, me da esperanza.", user_id=demo_user_id_2)

# --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == '__main__':
    try:
        main()
        print("\n✅ --- Proceso finalizado exitosamente. ---")
    except Exception as e:
        print(f"\n❌ OCURRIÓ UN ERROR CRÍTICO EN EL PIPELINE: {e}")
        traceback.print_exc()
