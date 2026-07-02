# src/cognitive_model_trainer.py

import pandas as pd
import joblib
import os
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

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
            class_weight='balanced', 
            n_jobs=-1
        )

       
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

        # --- 4.5 Matriz de Confusión (Out-of-Fold) ---
        # cross_val_score NO devuelve predicciones individuales, solo el score agregado.
        # Para poder auditar solapamientos entre arquetipos (ej. Prof_Subutil vs
        # Com_Desafiado) necesitamos las predicciones out-of-fold de cada instancia,
        # que se obtienen con cross_val_predict usando la misma estrategia de CV.
        print("\n--- 🔍 Calculando matriz de confusión (predicciones out-of-fold)... ---")
        y_pred_oof = cross_val_predict(model, X, y, cv=cv_strategy, n_jobs=-1)

        cm_labels = sorted(y.unique().tolist())
        cm = confusion_matrix(y, y_pred_oof, labels=cm_labels)

        report_txt = classification_report(y, y_pred_oof, labels=cm_labels, zero_division=0)
        print("\n  › Reporte de Clasificación (Out-of-Fold, 6 arquetipos):")
        print(report_txt)
        mlflow.log_text(report_txt, "reports/classification_report_archetypes_oof.txt")

        # Guardamos la matriz también como CSV, útil para pegar directo en la tesis (Cap. V)
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        os.makedirs("reports", exist_ok=True)
        cm_csv_path = os.path.join("reports", "confusion_matrix_archetypes.csv")
        cm_df.to_csv(cm_csv_path)
        mlflow.log_artifact(cm_csv_path, artifact_path="reports")

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(8, len(cm_labels) * 1.3), max(6, len(cm_labels) * 1.0)))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                xticklabels=cm_labels, yticklabels=cm_labels,
                title='Matriz de Confusión - Arquetipos Cognitivos (Out-of-Fold, 5-Fold CV)',
                ylabel='Arquetipo Real', xlabel='Arquetipo Predicho'
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            thresh = cm.max() / 1.5 if cm.max() > 0 else 0.1
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/confusion_matrix_archetypes_oof.png")
            plt.close(fig)
            print(f"  › Matriz de confusión guardada en '{cm_csv_path}' y registrada en MLflow.")
        except ImportError:
            warnings.warn("Matplotlib no disponible. Se omite el gráfico de la matriz de confusión.")

        # Chequeo automático de solapamientos: para cada arquetipo, ¿a qué otro arquetipo
        # se confunde con más frecuencia? Útil para responder directamente a la pregunta
        # del comité sobre transferencias de carga indebidas entre arquetipos.
        print("\n  › Principales confusiones por arquetipo (fuera de la diagonal):")
        for i, arch in enumerate(cm_labels):
            row = cm[i].copy()
            total = row.sum()
            row[i] = 0  # anulamos la diagonal para buscar el error dominante
            if total > 0 and row.max() > 0:
                j = int(np.argmax(row))
                pct = row[j] / total
                print(f"     - {arch}: {row[j]}/{total} ({pct:.1%}) confundido con {cm_labels[j]}")

        # --- 5. Re-entrenar el Modelo Final con TODOS los datos ---
        # Una vez validada la configuración del modelo, lo re-entrenamos con todos los datos
        # para que aprenda lo máximo posible. Este es el modelo que se usará en producción.
        # FIX: se mantiene dentro del mismo 'run' que las métricas de CV, para que el
        # modelo final quede trazado junto a la validación que lo respalda.
        print("\n--- 🚂 Re-entrenando modelo final con el 100% de los datos... ---")
        final_model = RandomForestClassifier(**cfg_tutor, class_weight='balanced', n_jobs=-1).fit(X, y)

        # --- 6. Guardar Modelo Final y Registrar Artefacto ---
        model_save_path = config['model_paths']['cognitive_tutor']
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)

        # Se registra el modelo final en el mismo 'run' de MLflow que sus métricas de CV.
        mlflow.sklearn.log_model(final_model, "cognitive_tutor_model_final")

        print(f"\n  › Modelo final guardado en: {model_save_path} y registrado en MLflow.")

    print("\n--- ✅ Pipeline del Tutor Cognitivo Finalizado. ---")
