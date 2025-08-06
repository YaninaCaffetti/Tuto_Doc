# src/emotion_classifier.py 

import pandas as pd
import torch
from torch import nn
import numpy as np
import os
import warnings
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# --- SKLearn Imports ---
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# --- Hugging Face Imports ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, Features, ClassLabel, Value

# --- 1. CLASE DE CLASIFICACIÓN (Para Inferencia) ---

class EmotionClassifier:
    """
    Clasificador para la inferencia de emociones a partir de texto.
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict_proba(self, text: str) -> dict:
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}

    def predict(self, text: str) -> str:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

# --- 2. LÓGICA DE PREPARACIÓN DE DATOS ---
# Sin cambios en la lógica de obtención de datos.
def get_custom_domain_data() -> pd.DataFrame:
    """Carga el corpus de emociones específico del dominio de la tesis."""
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
        ("Me siento desmotivado con la búsqueda.", "Tristeza"), ("¡Conseguí la entrevista!", "Alegría"),
        ("La situación es frustrante.", "Ira"), ("Tengo fe en que el plan funcionará.", "Confianza"),
        ("El procedimiento es claro.", "Neutral"), ("Tengo miedo de no estar a la altura.", "Miedo"),
        ("¡Wow, no puedo creer que esto sea posible!", "Sorpresa"), ("No puedo esperar a ver los resultados.", "Anticipación"),
        ("¡Qué buena noticia! Esto me da mucho ánimo.", "Alegría"), ("Recibí la documentación para el siguiente paso.", "Neutral")
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def download_and_prepare_dataset(config: dict, emotion_labels: list) -> pd.DataFrame:
    """Descarga, procesa y guarda un dataset unificado para el entrenamiento."""
    url = config['data_paths']['emotion_corpus_url']
    raw_local_path = config['data_paths']['emotion_corpus_local']
    processed_local_path = raw_local_path.replace('.csv', '_processed.csv')

    if os.path.exists(processed_local_path):
        print(f"  › Cargando dataset procesado desde '{processed_local_path}'.")
        return pd.read_csv(processed_local_path)

    if not os.path.exists(raw_local_path):
        print(f"  › Descargando dataset público de emociones desde {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            os.makedirs(os.path.dirname(raw_local_path), exist_ok=True)
            with open(raw_local_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except requests.exceptions.RequestException as e:
            warnings.warn(f"No se pudo descargar el dataset público. Usando solo corpus de dominio. Error: {e}")
            return get_custom_domain_data()

    df_public = pd.read_csv(raw_local_path, header=None, names=['text', 'emotion_raw'])
    label_map = {'anger': 'Ira', 'sadness': 'Tristeza', 'fear': 'Miedo', 'joy': 'Alegría', 'surprise': 'Sorpresa', 'love': 'Confianza', 'others': 'Neutral'}
    df_public['emotion'] = df_public['emotion_raw'].map(label_map)
    df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]
    
    df_domain = get_custom_domain_data()
    df_combined = pd.concat([df_public_clean, df_domain], ignore_index=True)
    df_final = df_combined[df_combined['emotion'].isin(emotion_labels)].drop_duplicates().reset_index(drop=True)
    
    print(f"  › Dataset combinado creado con {len(df_final)} ejemplos. Guardando en '{processed_local_path}'.")
    df_final.to_csv(processed_local_path, index=False)
    return df_final

# --- 3. PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN AVANZADO ---

def train_and_evaluate_emotion_classifier(config: dict, run_hyperparameter_search: bool = False):
    """
    Orquesta el pipeline experimental completo para el clasificador de emociones.

    Args:
        config (dict): El diccionario de configuración cargado desde config.yaml.
        run_hyperparameter_search (bool): Si es True, ejecuta una búsqueda de
            hiperparámetros con Optuna antes de la validación cruzada.
    """
    print("\n--- [PARTE I] Iniciando Pipeline Experimental del Clasificador de Emociones... ---")
    
    # --- A. Preparación de Datos y Parámetros ---
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    
    df_processed = download_and_prepare_dataset(config, EMOTION_LABELS)

    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
    df_processed['label'] = df_processed['emotion'].map(label2id)

    def model_init():
        """Función requerida por Optuna para cargar un modelo nuevo en cada trial."""
        return AutoModelForSequenceClassification.from_pretrained(
            cfg_emo['model_name'],
            num_labels=len(EMOTION_LABELS),
            id2label=id2label,
            label2id=label2id
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
    def tokenize_function(examples): 
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    best_params = {
        "learning_rate": float(cfg_emo['learning_rate']),
        "num_train_epochs": cfg_emo['epochs']
    }

    # --- B. (Opcional) Búsqueda de Hiperparámetros con Optuna ---
    if run_hyperparameter_search:
        print("\n--- 🔍 Iniciando Búsqueda de Hiperparámetros con Optuna... ---")
        
        # Dividir una pequeña porción para la búsqueda
        train_df_hp, eval_df_hp = train_test_split(df_processed, test_size=0.3, random_state=RANDOM_STATE, stratify=df_processed['label'])
        train_ds_hp = Dataset.from_pandas(train_df_hp).map(tokenize_function, batched=True)
        eval_ds_hp = Dataset.from_pandas(eval_df_hp).map(tokenize_function, batched=True)

        def objective(trial: optuna.Trial):
            training_args = TrainingArguments(
                output_dir="./hp_search_results",
                learning_rate=trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
                num_train_epochs=trial.suggest_int("num_train_epochs", 3, 8),
                per_device_train_batch_size=cfg_emo['train_batch_size'],
                evaluation_strategy="epoch",
                logging_steps=float('inf'), # Desactivar logs intermedios
                report_to="none"
            )
            trainer = Trainer(model_init=model_init, args=training_args, train_dataset=train_ds_hp, eval_dataset=eval_ds_hp)
            result = trainer.train()
            return result.training_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10) # 10 trials para una búsqueda razonable
        best_params = study.best_params
        print(f"--- ✅ Búsqueda finalizada. Mejores parámetros encontrados: {best_params} ---")

    # --- C. Validación Cruzada (Cross-Validation) ---
    print("\n--- 🔄 Iniciando Validación Cruzada (K-Fold)... ---")
    n_splits = 5 # 5 folds es un estándar robusto
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    all_metrics = []
    error_analysis_data = []

    for fold, (train_index, val_index) in enumerate(skf.split(df_processed['text'], df_processed['emotion'])):
        print(f"\n  --- Fold {fold + 1}/{n_splits} ---")
        train_df = df_processed.iloc[train_index]
        val_df = df_processed.iloc[val_index]

        train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
        val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=f"./cv_results/fold_{fold}",
            **best_params, # Usar los mejores parámetros encontrados
            per_device_train_batch_size=cfg_emo['train_batch_size'],
            evaluation_strategy="epoch",
            logging_steps=float('inf'),
            report_to="none"
        )
        
        model = model_init() # Cargar un modelo fresco para cada fold
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()
        
        # Evaluación y Análisis de Errores
        predictions = trainer.predict(val_ds)
        y_pred_labels = np.argmax(predictions.predictions, axis=1)
        y_pred = [id2label[i] for i in y_pred_labels]
        y_true = val_df['emotion']
        
        fold_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        all_metrics.append(fold_f1)
        print(f"  › F1-Score (Macro) para el Fold {fold + 1}: {fold_f1:.3f}")

        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                error_analysis_data.append({
                    "fold": fold + 1,
                    "text": val_df.iloc[i]['text'],
                    "true_emotion": true_label,
                    "predicted_emotion": pred_label
                })

    # --- D. Resultados Finales y Artefactos ---
    print("\n--- 📊 Resultados Finales de la Validación Cruzada ---")
    mean_f1 = np.mean(all_metrics)
    std_f1 = np.std(all_metrics)
    print(f"  › F1-Score (Macro) Promedio: {mean_f1:.3f} ± {std_f1:.3f}")

    # Guardar el análisis de errores
    if error_analysis_data:
        error_df = pd.DataFrame(error_analysis_data)
        error_df.to_csv("emotion_error_analysis.csv", index=False)
        print("  › Análisis de errores guardado en 'emotion_error_analysis.csv'")
    
    # --- E. Entrenamiento Final y Guardado del Modelo de Producción ---
    print("\n--- 🚂 Entrenando modelo final con TODOS los datos... ---")
    full_dataset = Dataset.from_pandas(df_processed).map(tokenize_function, batched=True)
    
    final_training_args = TrainingArguments(
        output_dir="./results_emotion_final",
        **best_params,
        per_device_train_batch_size=cfg_emo['train_batch_size'],
        report_to="none"
    )
    
    final_model = model_init()
    final_trainer = Trainer(model=final_model, args=final_training_args, train_dataset=full_dataset)
    final_trainer.train()
    
    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo final de producción en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    final_trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("  › Modelo de producción guardado exitosamente.")

    print("--- ✅ Pipeline Experimental del Clasificador de Emociones Finalizado. ---")
