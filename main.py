# main.py

import pandas as pd
from collections import Counter
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import traceback
import yaml
from mlxtend.evaluate import mcnemar_table, mcnemar


# Importar nuestros módulos locales
from src.data_processing import run_feature_engineering, run_archetype_engineering, run_fuzzification, IF_HUPM
from src.emotion_classifier import train_and_evaluate_emotion_classifier
from src.cognitive_tutor import MoESystem

def main(config):
    """
    Función principal que orquesta todo el pipeline, ahora usando un diccionario de configuración.
    """
    print("\n--- 🚀 INICIANDO PIPELINE FINAL INTEGRADO Y EVALUACIÓN DOCTORAL ---")

    # --- Parte I: Entrenamiento del Clasificador de Emociones ---
    emotion_classifier = train_and_evaluate_emotion_classifier(config)

    # --- Parte II: Entrenamiento del Tutor Cognitivo ---
    print("\n--- [PARTE II] Entrenando el Tutor Cognitivo... ---")
    
    cognitive_model_final = None
    df_featured, df_fuzzified, feature_columns = None, None, []
    cognitive_tutor_ready = False
    
    cfg_cog = config['model_params']['cognitive_tutor']

    try:
        drive.mount('/content/drive', force_remount=True)
        df_raw = pd.read_csv(config['data_paths']['endis_raw'], delimiter=';', low_memory=False, index_col='ID')
        print(f"  › Dataset ENDIS cargado: {df_raw.shape[0]} filas.")
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_archetyped)
        pertenencia_cols = {col: col.replace('_v6', '').replace('_v3', '').replace('_v2', '').replace('_v1', '') for col in df_fuzzified.columns if 'Pertenencia_' in col}
        df_fuzzified.rename(columns=pertenencia_cols, inplace=True)
        columnas_arquetipos = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
        
        def determinar_arquetipo_predominante(row):
            pertenencias = row[columnas_arquetipos];
            if pertenencias.empty or len(pertenencias.dropna()) == 0 or pertenencias.max() < config['constants']['umbrales']['arquetipo']: return 'Arquetipo_No_Predominante'
            return pertenencias.idxmax().replace('Pertenencia_', '')
        
        df_fuzzified['Arquetipo_Predominante'] = df_fuzzified.apply(determinar_arquetipo_predominante, axis=1)
        feature_columns = [col for col in df_fuzzified.columns if '_memb' in col]
        df_entrenamiento = df_fuzzified[df_fuzzified['Arquetipo_Predominante'] != 'Arquetipo_No_Predominante'].copy()

        if len(df_entrenamiento) > 10:
            X_cognitive, y_cognitive = df_entrenamiento[feature_columns], df_entrenamiento['Arquetipo_Predominante']
            X_train_cog, X_test_cog, y_train_cog, y_test_cog = train_test_split(
                X_cognitive, y_cognitive, 
                test_size=cfg_cog['test_size'], 
                random_state=cfg_cog['random_state'], 
                stratify=y_cognitive
            )
            
            print("\n--- Aplicando SMOTE para balancear el conjunto de entrenamiento... ---")
            print(f"Distribución de clases ANTES de SMOTE: {Counter(y_train_cog)}")
            smote = SMOTE(random_state=cfg_cog['random_state'])
            X_train_sm, y_train_sm = smote.fit_resample(X_train_cog, y_train_cog)
            print(f"Distribución de clases DESPUÉS de SMOTE: {Counter(y_train_sm)}")
            
            cognitive_tutor_ready = True 
        else:
            raise ValueError("No hay suficientes datos después del filtrado para entrenar el tutor cognitivo.")

    except Exception as e:
        print(f"❌ ERROR AL ENTRENAR TUTOR COGNITIVO: {e}")
        traceback.print_exc()
        df_fuzzified = pd.DataFrame(index=[35906, 77570]); df_fuzzified['TIENE_CUD'] = ['No_Tiene_CUD', 'Si_Tiene_CUD']


    # --- Parte III: Evaluación, Benchmarking y Análisis Estadístico ---
    print("\n\n--- 📊 [PARTE III] Evaluación, Benchmarking y Análisis Estadístico ---")
    if cognitive_tutor_ready:
        
        print("\n--- 1. Entrenando modelos con datos de SMOTE... ---")

        rf_model = RandomForestClassifier(
            n_estimators=cfg_cog['n_estimators'], 
            max_depth=cfg_cog['max_depth'], 
            random_state=cfg_cog['random_state']
        )
        rf_model.fit(X_train_sm, y_train_sm)
        print("  › RandomForestClassifier entrenado.")
        cognitive_model_final = rf_model # Modelo que usará la demo

        dt_model = DecisionTreeClassifier(
            max_depth=cfg_cog['max_depth'], 
            random_state=cfg_cog['random_state']
        )
        dt_model.fit(X_train_sm, y_train_sm)
        print("  › DecisionTreeClassifier entrenado.")

        if_hupm_model = IF_HUPM(max_depth=cfg_cog['max_depth'])
        if_hupm_model.fit(X_train_sm, y_train_sm)
        print("  › IF-HUPM entrenado.")
        
        print("\n--- 2. Generando predicciones en el conjunto de prueba... ---")
        y_pred_rf = rf_model.predict(X_test_cog)
        y_pred_dt = dt_model.predict(X_test_cog)
        y_pred_if_hupm_raw = if_hupm_model.predict(X_test_cog)
        y_pred_if_hupm = y_pred_if_hupm_raw.str.extract(r'([A-Za-z_]+)')[0].fillna('Desconocido')

        print("\n--- 3. Reportes de Clasificación Comparativos ---")
        print("\n  › Reporte de Clasificación (RandomForest - Modelo Final):")
        print(classification_report(y_test_cog, y_pred_rf, zero_division=0))

        print("\n  › Reporte de Clasificación (DecisionTree - Benchmark de Simplicidad):")
        print(classification_report(y_test_cog, y_pred_dt, zero_division=0))

        print("\n  › Reporte de Clasificación (IF-HUPM - Modelo Interpretable):")
        print(classification_report(y_test_cog, y_pred_if_hupm, zero_division=0))
        
        print("\n--- 4. Test de Significancia Estadística (McNemar) ---")
        
        print("\n  › Comparando RandomForest vs. DecisionTree...")
        tb1 = mcnemar_table(y_target=y_test_cog, y_model1=y_pred_rf, y_model2=y_pred_dt)
        chi2_1, p_1 = mcnemar(ary=tb1, corrected=True)
        print(f"    Tabla de Contingencia:\n{tb1}")
        print(f"    Chi-cuadrado: {chi2_1:.2f}, P-value: {p_1:.4f}")
        if p_1 < 0.05: print("    Conclusión: La diferencia en el rendimiento es ESTADÍSTICAMENTE SIGNIFICATIVA.")
        else: print("    Conclusión: La diferencia en el rendimiento NO es estadísticamente significativa.")

        print("\n  › Comparando RandomForest vs. IF-HUPM...")
        tb2 = mcnemar_table(y_target=y_test_cog, y_model1=y_pred_rf, y_model2=y_pred_if_hupm)
        chi2_2, p_2 = mcnemar(ary=tb2, corrected=True)
        print(f"    Tabla de Contingencia:\n{tb2}")
        print(f"    Chi-cuadrado: {chi2_2:.2f}, P-value: {p_2:.4f}")
        if p_2 < 0.05: print("    Conclusión: La diferencia en el rendimiento es ESTADÍSTICAMENTE SIGNIFICATIVA.")
        else: print("    Conclusión: La diferencia en el rendimiento NO es estadísticamente significativa.")
            
    else:
        print("--- ⚠️ Se omitió el benchmarking y análisis estadístico porque no se pudo entrenar el tutor cognitivo. ---")

    # --- Parte IV: Demostración del Sistema Integrado ---
    print("\n\n--- 🏁 [PARTE IV] Demostración del Sistema Integrado ---")
    cognitive_tutor_system = MoESystem(cognitive_model_final, feature_columns, config['affective_rules']) if cognitive_tutor_ready else None
    
    def get_integrated_response(text: str, user_id: int):
        print("\n" + "="*80); print(f"INPUT DEL USUARIO (ID: {user_id}): '{text}'")
        emotion_probs = emotion_classifier.predict_proba(text)
        top_emotion = max(emotion_probs, key=emotion_probs.get)

        print(f"🧠 Emoción Dominante Detectada: {top_emotion} (Confianza: {emotion_probs[top_emotion]:.0%})")
        print(f"   (Espectro completo: {[f'{e}: {p:.0%}' for e, p in sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True) if p > 0.05]})")
        print("📖 Plan Cognitivo Generado:")
        
        if not cognitive_tutor_system:
            cognitive_plan = "[Sistema]: Tutor cognitivo no disponible (maqueta)."
        else:
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

if __name__ == '__main__':
    # Cargar la configuración desde el archivo YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
