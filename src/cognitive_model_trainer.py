# src/cognitive_model_trainer.py

import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_cognitive_tutor(config: dict):
    """
    Entrena y guarda el modelo del tutor cognitivo (árbol de decisión).

    Este pipeline carga los datos de perfiles, entrena un clasificador de árbol de
    decisión para predecir el arquetipo y guarda el modelo entrenado.

    Args:
        config (dict): El diccionario de configuración cargado desde config.yaml.
    """
    print("\n--- 🧠 Iniciando entrenamiento del Tutor Cognitivo... ---")
    
    # --- 1. Cargar Datos y Configuración ---
    cfg_tutor = config['model_params']['cognitive_tutor']
    data_path = config['data_paths'].get('cognitive_training_data') # Asume una nueva ruta en el config
    
    if not data_path or not os.path.exists(data_path):
        print(f"❌ Error: No se encontró el archivo de datos para el tutor cognitivo en '{data_path}'.")
        print("   Asegúrese de que la ruta 'cognitive_training_data' esté definida en config.yaml.")
        return

    print(f"  › Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)

    # Definir características (X) y objetivo (y)
    # Asume que la última columna es el arquetipo a predecir.
    target_column = df.columns[-1] 
    feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"  › Variable objetivo: '{target_column}'")
    print(f"  › {len(feature_columns)} características de entrada.")

    # --- 2. Dividir Datos y Entrenar Modelo ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=cfg_tutor['random_state'],
        stratify=y
    )

    print("  › Entrenando el modelo de Árbol de Decisión...")
    model = DecisionTreeClassifier(
        max_depth=cfg_tutor['max_depth'],
        min_samples_leaf=cfg_tutor['min_samples_leaf'],
        random_state=cfg_tutor['random_state']
    )
    model.fit(X_train, y_train)

    # --- 3. Evaluar el Modelo ---
    print("\n--- 📊 Evaluando el Tutor Cognitivo en el conjunto de prueba... ---")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  › **Accuracy Final en el conjunto de prueba:** {accuracy:.3f}")
    
    print("\n  › Reporte de Clasificación Final:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- 4. Guardar el Modelo ---
    model_save_path = config['model_paths']['cognitive_tutor']
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"\n  › Guardando el modelo final en: {model_save_path}")
    joblib.dump(model, model_save_path)
    
    print("\n--- ✅ Pipeline del Tutor Cognitivo Finalizado. ---")

