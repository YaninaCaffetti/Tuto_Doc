# src/emotion_classifier.py (con Back-Translation, Mejoras y MLflow)

import pandas as pd
import torch
from torch import nn
import numpy as np
import os
import warnings
from typing import Union, List, Dict

# --- SKLearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# --- Hugging Face Imports ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset, Dataset

# --- MLflow Import ---
import mlflow

# --- 1. CLASE DE CLASIFICACIÓN (Para Inferencia) ---
class EmotionClassifier:
    """Clasificador para la inferencia de emociones a partir de texto."""
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict_proba(self, text: Union[str, List[str]]) -> List[Dict[str, float]]:
        """Calcula la distribución de probabilidad de las emociones para un texto o lista de textos."""
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        results = []
        for prob_tensor in probabilities:
            prob_dict = {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(prob_tensor)}
            results.append(prob_dict)
        return results

    def predict(self, text: str, return_index: bool = False) -> Union[str, int]:
        """Predice la emoción dominante en un texto."""
        probs_list = self.predict_proba(text)[0]
        probabilities = list(probs_list.values())
        predicted_index = np.argmax(probabilities)
        if return_index:
            return predicted_index
        else:
            return self.model.config.id2label[predicted_index]

# --- 2. LÓGICA DE PREPARACIÓN DE DATOS Y AUMENTO ---
def get_custom_domain_data() -> pd.DataFrame:
    """Carga un corpus de emociones específico del dominio de la tesis."""
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def augment_with_back_translation(df: pd.DataFrame, lang_src: str = 'es', lang_tgt: str = 'en') -> pd.DataFrame:
    """Aumenta un DataFrame de texto usando la técnica de Back-Translation."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  › Realizando Back-Translation en el dispositivo: {device.upper()}")
        model_name_src_tgt = f'Helsinki-NLP/opus-mt-{lang_src}-{lang_tgt}'
        model_name_tgt_src = f'Helsinki-NLP/opus-mt-{lang_tgt}-{lang_src}'
        tokenizer_src_tgt = MarianTokenizer.from_pretrained(model_name_src_tgt)
        model_src_tgt = MarianMTModel.from_pretrained(model_name_src_tgt).to(device)
        tokenizer_tgt_src = MarianTokenizer.from_pretrained(model_name_tgt_src)
        model_tgt_src = MarianMTModel.from_pretrained(model_name_tgt_src).to(device)
        augmented_texts = []
        for text in df['text']:
            inputs = tokenizer_src_tgt(text, return_tensors="pt", padding=True).to(device)
            translated_ids = model_src_tgt.generate(**inputs)
            text_tgt = tokenizer_src_tgt.decode(translated_ids[0], skip_special_tokens=True)
            inputs_back = tokenizer_tgt_src(text_tgt, return_tensors="pt", padding=True).to(device)
            back_translated_ids = model_tgt_src.generate(**inputs_back)
            text_back = tokenizer_tgt_src.decode(back_translated_ids[0], skip_special_tokens=True)
            augmented_texts.append(text_back)
        return pd.DataFrame({'text': augmented_texts, 'emotion': df['emotion']})
    except Exception as e:
        warnings.warn(f"No se pudo realizar el Back-Translation. Se omitirá este paso. Error: {e}")
        return pd.DataFrame()

def download_and_prepare_dataset(emotion_labels: list, use_augmentation: bool = True) -> pd.DataFrame:
    """Descarga un dataset público, lo aumenta y lo combina con el corpus de dominio."""
    print("  › Cargando dataset 'emotion' desde el Hub de Hugging Face...")
    try:
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        label_map = {0: 'Tristeza', 1: 'Alegría', 2: 'Amor/Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
        df_public['emotion'] = df_public['label'].map(label_map)
        df_public['emotion'] = df_public['emotion'].replace({'Amor/Confianza': 'Confianza'})
        df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]
        df_domain = get_custom_domain_data()
        dfs_to_combine = [df_public_clean, df_domain]
        if use_augmentation:
            print("  › Aumentando corpus de dominio con Back-Translation...")
            df_domain_augmented = augment_with_back_translation(df_domain)
            if not df_domain_augmented.empty:
                dfs_to_combine.append(df_domain_augmented)
        df_combined = pd.concat(dfs_to_combine, ignore_index=True)
        df_final = df_combined[df_combined['emotion'].isin(emotion_labels)].drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        print(f"  › Dataset combinado creado con {len(df_final)} ejemplos únicos.")
        return df_final
    except Exception as e:
        warnings.warn(f"No se pudo descargar el dataset público. Usando solo corpus de dominio. Error: {e}")
        return get_custom_domain_data()

# --- 3. PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN ---
class WeightedLossTrainer(Trainer):
    """Trainer personalizado que utiliza una función de pérdida ponderada."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        if self.class_weights is not None:
            print(f"  › WeightedLossTrainer inicializado con pesos de clase: {np.round(self.class_weights.cpu().numpy(), 2)}")
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate_emotion_classifier(config: dict):
    """Orquesta el pipeline completo para el clasificador de emociones, registrando los resultados en MLflow."""
    print("\n--- 🎭 Iniciando entrenamiento del Clasificador de Emociones... ---")
    
    # --- Configurar MLflow para el Trainer de Hugging Face ---
    os.environ["MLFLOW_TRACKING_URI"] = config['mlflow']['tracking_uri']
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config['mlflow']['experiment_name']
    
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    
    df_processed = download_and_prepare_dataset(EMOTION_LABELS, use_augmentation=cfg_emo['data_augmentation']['use_back_translation'])
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    df_processed['label'] = df_processed['emotion'].map(label2id)

    train_val_df, test_df = train_test_split(df_processed, test_size=0.20, random_state=RANDOM_STATE, stratify=df_processed['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val_df['label'])
    
    train_ds, val_ds, test_ds = Dataset.from_pandas(train_df), Dataset.from_pandas(val_df), Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
    def tokenize(batch): return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
    
    train_ds, val_ds, test_ds = train_ds.map(tokenize, batched=True), val_ds.map(tokenize, batched=True), test_ds.map(tokenize, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label=id2label, label2id=label2id)
    
    class_weights = compute_class_weight('balanced', classes=EMOTION_LABELS, y=train_df['emotion'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    training_params = cfg_emo['training_params']
    training_params['learning_rate'] = float(training_params['learning_rate'])
    
    # --- Añadir 'report_to' en TrainingArguments ---
    training_args = TrainingArguments(
        output_dir="./results_emotion_training",
        report_to="mlflow", # Activar el logging a MLflow
        run_name="train_emotion_classifier", # Nombre de la ejecución en MLflow
        eval_strategy="epoch",
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
    
    final_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    final_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    final_f1_score = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"  › **Precision (Macro) Final:** {final_precision:.3f}")
    print(f"  › **Recall (Macro) Final:** {final_recall:.3f}")
    print(f"  › **F1-Score (Macro) Final:** {final_f1_score:.3f}")

    # --- Registrar métricas finales en la misma ejecución de MLflow ---
    mlflow.log_metric("final_test_precision_macro", final_precision)
    mlflow.log_metric("final_test_recall_macro", final_recall)
    mlflow.log_metric("final_test_f1_score_macro", final_f1_score)

    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo final de producción en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # El Trainer de Hugging Face gestiona el fin de la ejecución de MLflow.
    print("\n--- ✅ Pipeline del Clasificador de Emociones Finalizado. ---")
