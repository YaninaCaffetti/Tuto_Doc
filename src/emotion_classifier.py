# src/emotion_classifier.py (Versión Optimizada con Parámetros Corregidos)

import pandas as pd
import torch
from torch import nn
import numpy as np
import os
import warnings

# --- SKLearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

# --- Hugging Face Imports ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset

# --- 1. CLASE DE CLASIFICACIÓN (Para Inferencia) ---
class EmotionClassifier:
    """
    Clasificador para la inferencia de emociones a partir de texto.

    Esta clase encapsula un modelo y tokenizador de Hugging Face para facilitar
    la predicción de emociones en nuevas muestras de texto.
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        """
        Inicializa el clasificador de emociones.

        Args:
            model (AutoModelForSequenceClassification): El modelo de transformers fine-tuned.
            tokenizer (AutoTokenizer): El tokenizador correspondiente al modelo.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict_proba(self, text: str) -> dict:
        """
        Calcula la distribución de probabilidad de todas las emociones para un texto.

        Args:
            text (str): El texto de entrada a ser clasificado.

        Returns:
            dict: Un diccionario que mapea cada etiqueta de emoción a su
                  probabilidad calculada.
        """
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}

    def predict(self, text: str) -> str:
        """
        Predice la etiqueta de la emoción dominante en un texto.

        Args:
            text (str): El texto de entrada a ser clasificado.

        Returns:
            str: La etiqueta de la emoción con la mayor probabilidad.
        """
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

# --- 2. LÓGICA DE PREPARACIÓN DE DATOS ---
def get_custom_domain_data() -> pd.DataFrame:
    """
    Carga el corpus de emociones específico del dominio de la tesis.

    Estos ejemplos son cruciales para adaptar el modelo de lenguaje al
    contexto particular del proyecto.

    Returns:
        pd.DataFrame: Un DataFrame con las columnas 'text' y 'emotion'.
    """
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def download_and_prepare_dataset(emotion_labels: list) -> pd.DataFrame:
    """
    Descarga un dataset público desde Hugging Face y lo combina con el corpus de dominio.

    Este proceso unifica un corpus general con datos específicos del proyecto
    para crear un conjunto de entrenamiento más robusto y contextualizado.

    Args:
        emotion_labels (list): La lista de etiquetas de emoción válidas para el proyecto.

    Returns:
        pd.DataFrame: Un DataFrame unificado y listo para el entrenamiento.
    """
    print("  › Cargando dataset 'emotion' desde el Hub de Hugging Face...")
    try:
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        label_map = {0: 'Tristeza', 1: 'Alegría', 2: 'Amor/Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
        df_public['emotion'] = df_public['label'].map(label_map)
        final_map = {'Amor/Confianza': 'Confianza'}
        df_public['emotion'] = df_public['emotion'].replace(final_map)
        df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]
        df_domain = get_custom_domain_data()
        df_combined = pd.concat([df_public_clean, df_domain], ignore_index=True)
        df_final = df_combined[df_combined['emotion'].isin(emotion_labels)].drop_duplicates(subset=['text']).reset_index(drop=True)
        print(f"  › Dataset combinado creado con {len(df_final)} ejemplos únicos.")
        return df_final
    except Exception as e:
        warnings.warn(f"No se pudo descargar el dataset público. Usando solo corpus de dominio. Error: {e}")
        return get_custom_domain_data()

# --- 3. PIPELINE DE ENTRENamiento Y EVALUACIÓN OPTIMIZADO ---
class WeightedLossTrainer(Trainer):
    """
    Trainer de Hugging Face personalizado que utiliza una función de pérdida
    ponderada para mitigar el efecto del desbalance de clases durante el
    entrenamiento del modelo.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        """
        Inicializa el Trainer personalizado.

        Args:
            class_weights (torch.Tensor, optional): Un tensor con los pesos
                calculados para cada clase.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Sobrescribe el método de cálculo de pérdida para incluir los pesos de clase.

        Esta modificación asegura que el modelo penalice más los errores en las
        clases minoritarias, mejorando su capacidad para identificarlas.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate_emotion_classifier(config: dict):
    """
    Orquesta el pipeline experimental completo para el clasificador de emociones.

    Este proceso incluye la preparación de datos, el cálculo de pesos para el
    balanceo de clases, el fine-tuning de un modelo BERT y la evaluación final
    de su rendimiento en un conjunto de prueba no visto.

    Args:
        config (dict): El diccionario de configuración cargado desde config.yaml.
    """
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    
    df_processed = download_and_prepare_dataset(EMOTION_LABELS)
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    df_processed['label'] = df_processed['emotion'].map(label2id)

    train_val_df, test_df = train_test_split(df_processed, test_size=0.20, random_state=RANDOM_STATE, stratify=df_processed['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val_df['label'])
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
    def tokenize(batch): return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
    
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label=id2label, label2id=label2id)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['emotion']), y=train_df['emotion'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    training_params = cfg_emo['training_params']
    training_params['learning_rate'] = float(training_params['learning_rate'])
    
    # --- ¡ESTA ES LA CORRECCIÓN! ---
    # Se han renombrado los parámetros para que coincidan con la API actual de Transformers.
    training_args = TrainingArguments(
        output_dir="./results_emotion_training",
        eval_strategy="epoch",          # ANTES: evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        **training_params
    )
    
    trainer = WeightedLossTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, class_weights=class_weights_tensor)
    
    print("\n--- 🚂 Entrenando modelo final... ---")
    trainer.train()

    print("\n--- 📊 Evaluando el modelo final en el conjunto de prueba no visto... ---")
    predictions = trainer.predict(test_ds)
    y_pred_labels = np.argmax(predictions.predictions, axis=1)
    y_true_labels = test_ds['label']
    y_pred = [id2label[i] for i in y_pred_labels]
    y_true = [id2label[i] for i in y_true_labels]

    print("\n  › Reporte de Clasificación Final:")
    print(classification_report(y_true, y_pred, labels=EMOTION_LABELS, zero_division=0))
    
    final_f1_score = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"  › **F1-Score (Macro) Final en el conjunto de prueba:** {final_f1_score:.3f}")

    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo final de producción en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print("\n--- ✅ Pipeline Experimental del Clasificador de Emociones Finalizado. ---")
