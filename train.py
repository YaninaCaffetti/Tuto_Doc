# train.py (Versión de Depuración)

import pandas as pd
from collections import Counter
# from google.colab import drive # Ya eliminado, correcto.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import traceback
import yaml
import os
import joblib

# Importar nuestros módulos locales
from src.data_processing import run_feature_engineering, run_archetype_engineering, run_fuzzification, IF_HUPM
from src.emotion_classifier import train_and_evaluate_emotion_classifier
from imblearn.over_sampling import SMOTE
from mlxtend.evaluate import mcnemar_table, mcnemar

def train_and_evaluate_all(config):
    print("\n--- 🚀 INICIANDO PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN ---")

    # --- Parte I: Clasificador de Emociones ---
    # ... (esta parte no se modifica)

    # --- Parte II: Tutor Cognitivo ---
    print("\n--- [PARTE II] Entrenando y Evaluando el Tutor Cognitivo... ---")
    
    # Hemos eliminado el try-except para que los errores sean visibles
    
    # 1. Cargar los datos. Si esto falla, el script se detendrá.
    print("  › Intentando cargar el dataset desde:", config['data_paths']['endis_raw'])
    df_raw = pd.read_csv(config['data_paths']['endis_raw'], delimiter=';', low_memory=False, index_col='ID')
    print("  › Dataset cargado exitosamente.")

    # 2. Procesamiento de datos
    df_featured = run_feature_engineering(df_raw)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)
    
    # ... (el resto de la lógica de preparación de datos)

    pertenencia_cols = {col: col.replace('_v6', '').replace('_v3', '').replace('_v2', '').replace('_v1', '') for col in df_fuzzified.columns if 'Pertenencia_' in col}
    df_fuzzified.rename(columns=pertenencia_cols, inplace=True)
    columnas_arquetipos = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
    
    def determinar_arquetipo_predominante(row):
        pertenencias = row[columnas_arquetipos]
        if pertenencias.empty or len(pertenencias.dropna()) == 0 or pertenencias.max() < config['constants']['umbrales']['arquetipo']: return 'Arquetipo_No_Predominante'
        return pertenencias.idxmax().replace('Pertenencia_', '')
    
    df_fuzzified['Arquetipo_Predominante'] = df_fuzzified.apply(determinar_arquetipo_predominante, axis=1)
    feature_columns = [col for col in df_fuzzified.columns if '_memb' in col]
    df_entrenamiento = df_fuzzified[df_fuzzified['Arquetipo_Predominante'] != 'Arquetipo_No_Predominante'].copy()

    if len(df_entrenamiento) <= 10:
        raise ValueError("No hay suficientes datos para entrenar el tutor cognitivo después del filtrado.")
    
    # 3. División de datos y SMOTE
    X_cognitive, y_cognitive = df_entrenamiento[feature_columns], df_entrenamiento['Arquetipo_Predominante']
    X_train_cog, X_test_cog, y_train_cog, y_test_cog = train_test_split(X_cognitive, y_cognitive, test_size=config['model_params']['cognitive_tutor']['test_size'], random_state=config['model_params']['cognitive_tutor']['random_state'], stratify=y_cognitive)
    
    print("\n--- Aplicando SMOTE para balancear el conjunto de entrenamiento... ---")
    smote = SMOTE(random_state=config['model_params']['cognitive_tutor']['random_state'])
    X_train_sm, y_train_sm = smote.fit_resample(X_train_cog, y_train_cog)

    # 4. Entrenamiento y guardado de modelos (ahora se ejecutará sí o sí)
    cfg_cog = config['model_params']['cognitive_tutor']
    print("\n--- Entrenando y evaluando modelos cognitivos... ---")

    rf_model = RandomForestClassifier(n_estimators=cfg_cog['n_estimators'], max_depth=cfg_cog['max_depth'], random_state=cfg_cog['random_state'])
    rf_model.fit(X_train_sm, y_train_sm)
    
    # Guardar el modelo final
    cognitive_model_path = config['model_paths']['cognitive_tutor']
    os.makedirs(os.path.dirname(cognitive_model_path), exist_ok=True)
    joblib.dump(rf_model, cognitive_model_path)
    print(f"  › Modelo final guardado en: {cognitive_model_path}")

    # Guardar perfiles de demo
    demo_ids = [35906, 77570]
    os.makedirs(os.path.dirname(config['data_paths']['demo_profiles']), exist_ok=True)
    df_fuzzified.loc[demo_ids].to_csv(config['data_paths']['demo_profiles'], index=True)
    print(f"  › Perfiles de demostración guardados en {config['data_paths']['demo_profiles']}")
    
    # ... (el resto del código de evaluación y benchmarks)

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_and_evaluate_all(config)
    print("\n✅ --- Proceso de entrenamiento, evaluación y guardado finalizado. ---")
