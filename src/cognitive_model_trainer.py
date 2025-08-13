import pandas as pd
import joblib
import os
import mlflow 
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_cognitive_tutor(config: dict):
    """
    Entrena, evalúa y registra el modelo del tutor cognitivo usando MLflow.
    """
    print("\n--- 🧠 Iniciando entrenamiento del Tutor Cognitivo... ---")
    
    # --- Configurar y iniciar ejecución de MLflow ---
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name="train_cognitive_model"):
        
        # --- 1. Cargar Datos y Registrar Parámetros ---
        cfg_tutor = config['model_params']['cognitive_tutor']
        data_path = config['data_paths'].get('cognitive_training_data')
        
        # Registrar hiperparámetros en MLflow
        mlflow.log_params(cfg_tutor)
        
        if not data_path or not os.path.exists(data_path):
            print(f"❌ Error: No se encontró el archivo de datos en '{data_path}'.")
            return

        df = pd.read_csv(data_path)
        target_column = df.columns[-1] 
        feature_columns = [col for col in df.columns if col != target_column]
        X, y = df[feature_columns], df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=cfg_tutor['random_state'], stratify=y)

        # --- 2. Entrenar Modelo ---
        model = DecisionTreeClassifier(
            max_depth=cfg_tutor['max_depth'],
            min_samples_leaf=cfg_tutor['min_samples_leaf'],
            random_state=cfg_tutor['random_state']
        )
        model.fit(X_train, y_train)

        # --- 3. Evaluar y Registrar Métricas en MLflow ---
        print("\n--- 📊 Evaluando y registrando métricas en MLflow... ---")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Registrar métricas clave en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_f1_score", report_dict['macro avg']['f1-score'])
        mlflow.log_metric("macro_precision", report_dict['macro avg']['precision'])
        mlflow.log_metric("macro_recall", report_dict['macro avg']['recall'])
        
        print(f"  › Accuracy: {accuracy:.3f}")

        # --- 4. Guardar Modelo y Registrar Artefacto ---
        model_save_path = config['model_paths']['cognitive_tutor']
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        
        # Registrar el modelo en MLflow
        mlflow.sklearn.log_model(model, "cognitive_tutor_model")
        
        print(f"\n  › Modelo guardado en: {model_save_path} y registrado en MLflow.")
        
    print("\n--- ✅ Pipeline del Tutor Cognitivo Finalizado. ---")

