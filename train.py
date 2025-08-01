# train.py 

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import traceback
import yaml
import os
import joblib

# Importar  módulos locales
from src.data_processing import run_feature_engineering, run_archetype_engineering, run_fuzzification
from src.emotion_classifier import train_and_evaluate_emotion_classifier
from imblearn.over_sampling import SMOTE

def train_and_evaluate_all(config: dict):
    """
    Orquesta el pipeline completo de entrenamiento, evaluación y guardado de modelos.

    Este script ejecuta dos pipelines principales en secuencia:
    1.  **Clasificador de Emociones:** Entrena y evalúa un modelo BERT fine-tuned
        para la clasificación de emociones a partir de texto.
    2.  **Tutor Cognitivo:** Procesa los datos de la encuesta ENDIS, entrena un modelo
        RandomForest para clasificar perfiles de usuario en arquetipos y guarda
        los modelos y perfiles de demostración necesarios para la aplicación.

    Args:
        config (dict): El diccionario de configuración cargado desde config.yaml.
    """
    print("\n--- 🚀 INICIANDO PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN ---")

    # --- Parte I: Entrenamiento y Guardado del Clasificador de Emociones ---
    print(f"\n--- [PARTE I] Entrenando y Evaluando el Clasificador de Emociones... ---")
    try:
        train_and_evaluate_emotion_classifier(config)
        print("--- ✅ Parte I completada: Clasificador de Emociones entrenado y guardado. ---")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN LA PARTE I (Clasificador de Emociones): {e}")
        traceback.print_exc()
        return # Detener la ejecución si esta parte crucial falla

    # --- Parte II: Entrenamiento y Benchmarking del Tutor Cognitivo ---
    print("\n--- [PARTE II] Entrenando y Evaluando el Tutor Cognitivo... ---")
    
    try:
        # --- A. Carga y Procesamiento de Datos ---
        print("  › Cargando el dataset cognitivo...")
        df_raw = pd.read_csv(config['data_paths']['endis_raw'], delimiter=';', low_memory=False, index_col='ID')
        print("  › Dataset cargado exitosamente.")
        
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_featured)
        
        # --- B. Creación y Guardado de Perfiles para la Demostración ---
        demo_ids = config['data_paths'].get('demo_user_ids', []) # Usar IDs del config o una lista vacía
        valid_demo_ids = [uid for uid in demo_ids if uid in df_fuzzified.index]
        
        if valid_demo_ids:
            demo_profiles_df = df_fuzzified.loc[valid_demo_ids].copy()
            # Forzar escenario de CUD si el ID 14 está en la lista para asegurar la demo
            if 14 in demo_profiles_df.index:
                demo_profiles_df.loc[14, 'TIENE_CUD'] = 'Si_Tiene_CUD'
                print("  › Perfil de demo forzado: ID 14 tiene CUD.")
            
            os.makedirs(os.path.dirname(config['data_paths']['demo_profiles']), exist_ok=True)
            demo_profiles_df.to_csv(config['data_paths']['demo_profiles'], index=True)
            print(f"  › Perfiles de demostración guardados para los IDs: {valid_demo_ids}")
        else:
            print("  › Advertencia: Ninguno de los IDs de demo especificados en config.yaml se encontró en el dataset.")

        # --- C. Preparación Final de Datos para Entrenamiento ---
        pertenencia_cols_map = {col: col.replace('Pertenencia_', '') for col in df_fuzzified.columns if 'Pertenencia_' in col}
        df_fuzzified.rename(columns=pertenencia_cols_map, inplace=True)
        
        from src.cognitive_tutor import EXPERT_MAP
        columnas_arquetipos = [col for col in df_fuzzified.columns if col in EXPERT_MAP.keys()]
        
        def determinar_arquetipo_predominante(row: pd.Series) -> str:
            """
            Función robusta para asignar una etiqueta de clase final.

            Maneja correctamente los casos donde no hay puntajes de pertenencia (NaNs)
            para evitar el error 'argmax of an empty sequence'.
            """
            pertenencias = row[columnas_arquetipos].dropna() # Clave: .dropna()
            
            if pertenencias.empty or pertenencias.max() < config['constants']['umbrales']['arquetipo']:
                return 'Arquetipo_No_Predominante'
            
            return pertenencias.idxmax()
        
        df_fuzzified['Arquetipo_Predominante'] = df_fuzzified.apply(determinar_arquetipo_predominante, axis=1)
        
        # --- D. División y Balanceo de Datos (SMOTE) ---
        feature_columns = [col for col in df_fuzzified.columns if '_memb' in col]
        df_entrenamiento = df_fuzzified[df_fuzzified['Arquetipo_Predominante'] != 'Arquetipo_No_Predominante'].copy()

        print(f"  › {len(df_entrenamiento)} perfiles superaron el umbral de arquetipo y se usarán para el entrenamiento.")

        if len(df_entrenamiento) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                df_entrenamiento[feature_columns], 
                df_entrenamiento['Arquetipo_Predominante'], 
                test_size=config['model_params']['cognitive_tutor']['test_size'], 
                random_state=config['model_params']['cognitive_tutor']['random_state'], 
                stratify=df_entrenamiento['Arquetipo_Predominante']
            )
            
            print("\n--- Aplicando SMOTE para balancear el conjunto de entrenamiento... ---")
            smote = SMOTE(random_state=config['model_params']['cognitive_tutor']['random_state'])
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            # --- E. Entrenamiento y Guardado del Modelo Cognitivo ---
            cfg_cog = config['model_params']['cognitive_tutor']
            
            print("\n--- Entrenando el modelo cognitivo final (RandomForest)... ---")
            model = RandomForestClassifier(
                n_estimators=cfg_cog['n_estimators'], 
                max_depth=cfg_cog['max_depth'], 
                random_state=cfg_cog['random_state']
            ).fit(X_train_sm, y_train_sm)
            
            cognitive_model_path = config['model_paths']['cognitive_tutor']
            os.makedirs(os.path.dirname(cognitive_model_path), exist_ok=True)
            joblib.dump(model, cognitive_model_path)
            print(f"  › Modelo cognitivo guardado exitosamente en: {cognitive_model_path}")
            print("--- ✅ Parte II completada. ---")
            
        else:
            raise ValueError("No hay suficientes datos para entrenar el tutor cognitivo después del filtrado.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN LA PARTE II (Tutor Cognitivo): {e}")
        traceback.print_exc()

if __name__ == '__main__':
    """
    Punto de entrada del script.

    Carga la configuración desde 'config.yaml' y ejecuta el pipeline principal
    de entrenamiento y evaluación.
    """
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        train_and_evaluate_all(config)
        print("\n✅ --- Proceso de entrenamiento, evaluación y guardado finalizado. ---")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'config.yaml'. Asegúrate de que exista en el directorio raíz.")
    except Exception as e:
        print(f"❌ Un error inesperado ocurrió durante la ejecución: {e}")
        traceback.print_exc()
