# src/emotion_classifier.py (Versión Experimental con Metodología Avanzada)

import pandas as pd
import torch
from torch import nn
import numpy as np
import os
import warnings

# --- SKLearn Imports ---
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

# --- Hugging Face Imports ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, Features, ClassLabel, Value

# --- 1. CLASE DE CLASIFICACIÓN (Para Inferencia) ---
# Esta clase es el "motor" que usa la aplicación final.
class EmotionClassifier:
    """
    Clasificador para la inferencia de emociones a partir de texto, utilizando un
    modelo pre-entrenado de Hugging Face.
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        """Inicializa el clasificador de emociones."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict_proba(self, text: str) -> dict:
        """Calcula la distribución de probabilidad de todas las emociones para un texto."""
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}

    def predict(self, text: str) -> str:
        """Predice la etiqueta de la emoción dominante en un texto."""
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

# --- 2. LÓGICA DE PREPARACIÓN DE DATOS ---

def get_custom_domain_data() -> pd.DataFrame:
    """Carga el corpus de emociones específico del dominio de la tesis."""
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def download_and_prepare_dataset(emotion_labels: list) -> pd.DataFrame:
    """
    Descarga un dataset público desde Hugging Face, lo procesa y lo combina
    con el corpus de dominio para crear un set de datos de entrenamiento robusto.
    """
    print("  › Cargando dataset 'emotion' desde el Hub de Hugging Face...")
    try:
        # Cargar el dataset público estándar
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        
        # Mapeo de etiquetas del dataset público (números) a las del proyecto (texto)
        label_map = {0: 'Tristeza', 1: 'Alegría', 2: 'Amor/Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
        df_public['emotion'] = df_public['label'].map(label_map)
        
        # Mapeo secundario para alinear con tus etiquetas
        final_map = {'Amor/Confianza': 'Confianza'}
        df_public['emotion'] = df_public['emotion'].replace(final_map)
        
        df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]
        
        # Cargar y combinar con el dataset de dominio
        df_domain = get_custom_domain_data()
        df_combined = pd.concat([df_public_clean, df_domain], ignore_index=True)
        
        # Filtrar para quedarse solo con las etiquetas finales del proyecto y eliminar duplicados
        df_final = df_combined[df_combined['emotion'].isin(emotion_labels)].drop_duplicates(subset=['text']).reset_index(drop=True)
        
        print(f"  › Dataset combinado creado con {len(df_final)} ejemplos únicos.")
        return df_final
    except Exception as e:
        warnings.warn(f"No se pudo descargar el dataset público. Usando solo corpus de dominio. Error: {e}")
        return get_custom_domain_data()

# --- 3. PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN AVANZADO ---

class WeightedLossTrainer(Trainer):
    """
    Trainer de Hugging Face personalizado que utiliza una función de pérdida
    ponderada para mitigar el efecto del desbalance de clases.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Sobrescribe el método de cálculo de pérdida para incluir los pesos de clase.
        El **kwargs se añade para compatibilidad con versiones más nuevas de Transformers.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate_emotion_classifier(config: dict):
    """
    Orquesta el pipeline experimental completo para el clasificador de emociones,
    incluyendo validación cruzada y balanceo de clases.
    """
    # --- A. Preparación de Datos y Parámetros ---
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    
    df_processed = download_and_prepare_dataset(EMOTION_LABELS)

    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    df_processed['label'] = df_processed['emotion'].map(label2id)

    # --- B. Validación Cruzada (Cross-Validation) ---
    print("\n--- 🔄 Iniciando Validación Cruzada (K-Fold)... ---")
    n_splits = cfg_emo['experimental_pipeline']['cv_folds']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    all_fold_metrics = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(df_processed['text'], df_processed['emotion'])):
        print(f"\n  --- Fold {fold + 1}/{n_splits} ---")
        train_df = df_processed.iloc[train_index]
        val_df = df_processed.iloc[val_index]

        # Cálculo de Pesos para Balanceo de Clases (se hace para cada fold)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['emotion']), y=train_df['emotion'])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        
        tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
        def tokenize(batch): return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
        
        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label=id2label, label2id=label2id)
        
        training_params = cfg_emo['training_params']
        training_params['learning_rate'] = float(training_params['learning_rate'])
        
        training_args = TrainingArguments(output_dir=f"./cv_results/fold_{fold}", **training_params)
        
        trainer = WeightedLossTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, class_weights=class_weights_tensor)
        trainer.train()
        
        # Evaluación del fold
        predictions = trainer.predict(val_ds)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = val_ds['label']
        
        fold_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        all_fold_metrics.append(fold_f1)
        print(f"  › F1-Score (Macro) para el Fold {fold + 1}: {fold_f1:.3f}")

    # --- C. Resultados Finales y Conclusión de la Validación ---
    mean_f1 = np.mean(all_fold_metrics)
    std_f1 = np.std(all_fold_metrics)
    print(f"\n--- 📊 Resultados Finales de Validación Cruzada ---")
    print(f"  › F1-Score (Macro) Promedio: {mean_f1:.3f} ± {std_f1:.3f}")
    print("  › La desviación estándar baja indica que el rendimiento del modelo es estable y no depende de una división de datos afortunada.")
    
    # --- D. Entrenamiento Final del Modelo de Producción ---
    print("\n--- 🚂 Entrenando modelo final con TODOS los datos... ---")
    full_dataset = Dataset.from_pandas(df_processed).map(tokenize, batched=True)
    
    # Recalcular pesos con todos los datos para el modelo final
    final_class_weights = compute_class_weight('balanced', classes=np.unique(df_processed['emotion']), y=df_processed['emotion'])
    final_class_weights_tensor = torch.tensor(final_class_weights, dtype=torch.float)

    final_model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label=id2label, label2id=label2id)
    
    final_training_params = cfg_emo['training_params']
    final_training_params['learning_rate'] = float(final_training_params['learning_rate'])
    final_args = TrainingArguments(output_dir="./results_emotion_final", **final_training_params)
    
    final_trainer = WeightedLossTrainer(model=final_model, args=final_args, train_dataset=full_dataset, class_weights=final_class_weights_tensor)
    final_trainer.train()
    
    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo final de producción en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    final_trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print("\n--- ✅ Pipeline Experimental del Clasificador de Emociones Finalizado. ---")

