# src/cognitive_model_trainer.py

import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_cognitive_tutor(config: dict):
    """
    Entrena y evalúa el modelo del tutor cognitivo usando validación cruzada estratificada.
    Finalmente, re-entrena el modelo con todos los datos para producción.
    """
    # --- Configurar MLflow ---
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name="train_cognitive_model_cv"):
        
        # --- 1. Cargar Datos y Parámetros ---
        print("\n--- 🧠 Cargando datos y configurando el pipeline de Validación Cruzada... ---")
        cfg_tutor = config['model_params']['cognitive_tutor']
        data_path = config['data_paths'].get('cognitive_training_data')
        mlflow.log_params(cfg_tutor)
        
        if not data_path or not os.path.exists(data_path):
            print(f"❌ Error: No se encontró el archivo de datos en '{data_path}'.")
            return

        df = pd.read_csv(data_path)
        target_column = df.columns[-1] 
        feature_columns = [col for col in df.columns if col != target_column]
        X, y = df[feature_columns], df[target_column]
        
        # --- 2. Definir el Modelo y la Estrategia de Validación Cruzada ---
        model = RandomForestClassifier(
            n_estimators=cfg_tutor['n_estimators'],      
            max_depth=cfg_tutor['max_depth'],
            min_samples_leaf=cfg_tutor['min_samples_leaf'],
            random_state=cfg_tutor['random_state'],
            n_jobs=-1
        )

        # Usamos 5 pliegues estratificados. 'shuffle=True' es una buena práctica.
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg_tutor['random_state'])

        # --- 3. Ejecutar la Validación Cruzada ---
        print("\n--- 🔄 Ejecutando Validación Cruzada (5 folds)... ---")
        # Usamos 'f1_macro' como métrica principal, es más robusta que accuracy para clases desbalanceadas.
        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1_macro')

        # --- 4. Reportar y Registrar Resultados de la CV ---
        print("\n--- 📊 Resultados de la Validación Cruzada (F1-Score Macro)... ---")
        print(f"  › Scores de cada fold: {np.round(scores, 3)}")
        print(f"  › F1-Score Promedio (CV): {scores.mean():.3f} +/- {scores.std():.3f}")

        # Registramos las métricas clave de la validación cruzada en MLflow
        mlflow.log_metric("cv_f1_macro_mean", scores.mean())
        mlflow.log_metric("cv_f1_macro_std", scores.std())
        
        # Opcional: registrar el score de cada fold para un análisis más detallado
        for i, score in enumerate(scores):
            mlflow.log_metric("cv_f1_macro_fold", score, step=i+1)

    # --- 5. Re-entrenar el Modelo Final con TODOS los datos ---
    # Una vez validada la configuración del modelo, lo re-entrenamos con todos los datos
    # para que aprenda lo máximo posible. Este es el modelo que se usará en producción.
    print("\n--- 🚂 Re-entrenando modelo final con el 100% de los datos... ---")
    final_model = RandomForestClassifier(**cfg_tutor, n_jobs=-1).fit(X, y)
    
    # --- 6. Guardar Modelo Final y Registrar Artefacto ---
    model_save_path = config['model_paths']['cognitive_tutor']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(final_model, model_save_path)
    
    # Se registra el modelo final en el 'run' de MLflow
    # Esto asocia el modelo final con las métricas de CV que lo validaron.
    mlflow.sklearn.log_model(final_model, "cognitive_tutor_model_final")
    
    print(f"\n  › Modelo final guardado en: {model_save_path} y registrado en MLflow.")
    print("\n--- ✅ Pipeline del Tutor Cognitivo Finalizado. ---")
