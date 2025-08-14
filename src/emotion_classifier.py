# src/emotion_classifier.py (Integrado con mejoras: split+BT solo en train, seeds, MLflow, collator, metrics)
"""
Módulo de entrenamiento e inferencia para clasificación de emociones en español
con Transformers. Incluye:
- Preparación de datos + back-translation (solo en train)
- Entrenamiento con pérdida ponderada por clase
- Registro de métricas y artefactos en MLflow, incluyendo matriz de confusión
- Pipeline reproducible con fijación de semillas
"""

from __future__ import annotations

import os
import io
import json
import random
import warnings
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# Hugging Face
from datasets import Dataset,load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback, set_seed as hf_set_seed
)
from transformers import MarianMTModel, MarianTokenizer

# MLflow
import mlflow

# ==============================
# Utilidades
# ==============================

def set_seed(seed: int) -> None:
    """Fija semillas para reproducibilidad completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def ensure_list(text: Union[str, List[str]]) -> List[str]:
    """Garantiza que la entrada sea una lista de strings."""
    if isinstance(text, str):
        return [text]
    if isinstance(text, list) and all(isinstance(t, str) for t in text):
        return text
    raise TypeError("`text` debe ser str o List[str].")


# ==============================
# Inferencia
# ==============================

class EmotionClassifier:
    """Clasificador para inferencia de emociones a partir de texto."""

    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict_proba(self, text: Union[str, List[str]]) -> List[Dict[str, float]]:
        """Devuelve una lista de distribuciones de probabilidad por emoción."""
        texts = ensure_list(text)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        id2label = self.model.config.id2label
        return [
            {id2label[i]: float(p) for i, p in enumerate(row)}
            for row in probabilities
        ]

    def predict(self, text: str, return_index: bool = False) -> Union[str, int]:
        """Predice la emoción dominante en un texto; opcionalmente devuelve el índice."""
        probs_list = self.predict_proba(text)[0]
        labels = list(probs_list.keys())
        values = list(probs_list.values())
        idx = int(np.argmax(values))
        return idx if return_index else labels[idx]


# ==============================
# Datos y Aumento
# ==============================

def get_custom_domain_data() -> pd.DataFrame:
    """Pequeño corpus de dominio propio."""
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"),
        ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"),
        ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"),
        ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])


def augment_with_back_translation(df: pd.DataFrame, lang_src: str = 'es', lang_tgt: str = 'en') -> pd.DataFrame:
    """Back-Translation de textos (con MarianMT)."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  › Back-Translation en: {device.upper()}")

        model_name_src_tgt = f'Helsinki-NLP/opus-mt-{lang_src}-{lang_tgt}'
        model_name_tgt_src = f'Helsinki-NLP/opus-mt-{lang_tgt}-{lang_src}'

        tok_st = MarianTokenizer.from_pretrained(model_name_src_tgt)
        mt_st = MarianMTModel.from_pretrained(model_name_src_tgt).to(device)

        tok_ts = MarianTokenizer.from_pretrained(model_name_tgt_src)
        mt_ts = MarianMTModel.from_pretrained(model_name_tgt_src).to(device)

        augmented = []
        for text in df['text']:
            inputs = tok_st(text, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_ids = mt_st.generate(**inputs, max_new_tokens=128)
            text_tgt = tok_st.decode(translated_ids[0], skip_special_tokens=True)

            inputs_back = tok_ts(text_tgt, return_tensors="pt", padding=True, truncation=True).to(device)
            back_ids = mt_ts.generate(**inputs_back, max_new_tokens=128)
            text_back = tok_ts.decode(back_ids[0], skip_special_tokens=True)

            augmented.append(text_back)

        return pd.DataFrame({'text': augmented, 'emotion': df['emotion'].values})
    except Exception as e:
        warnings.warn(f"Back-Translation falló; se omite. Error: {e}")
        return pd.DataFrame(columns=['text', 'emotion'])


def load_base_dataset(emotion_labels: List[str]) -> pd.DataFrame:
    """
    Carga el dataset público y lo combina con el corpus de dominio,
    SIN aumentar ni partir. Devuelve columnas ['text','emotion'] filtradas y únicas.
    """
    print("  › Cargando dataset 'emotion' (HF) + dominio...")
    try:
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        label_map = {0: 'Tristeza', 1: 'Alegría', 2: 'Amor/Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
        df_public['emotion'] = df_public['label'].map(label_map).replace({'Amor/Confianza': 'Confianza'})
        df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]

        df_domain = get_custom_domain_data()

        df_base = pd.concat([df_public_clean, df_domain], ignore_index=True)
        df_base = (
            df_base[df_base['emotion'].isin(emotion_labels)]
            .drop_duplicates(subset=['text'], keep='first')
            .reset_index(drop=True)
        )
        print(f"  › Base combinada con {len(df_base)} ejemplos únicos.")
        return df_base
    except Exception as e:
        warnings.warn(f"No se pudo descargar el dataset público. Se usa solo dominio. Error: {e}")
        return get_custom_domain_data()


# ==============================
# Trainer con pérdida ponderada
# ==============================

class WeightedLossTrainer(Trainer):
    """Trainer con CrossEntropy ponderada por clase."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        if self.class_weights is not None:
            print(f"  › Pesos de clase: {np.round(self.class_weights.detach().cpu().numpy(), 3)}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ==============================
# Métricas
# ==============================

def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    """Calcula y devuelve métricas para la evaluación durante el entrenamiento."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}


# ==============================
# Pipeline principal
# ==============================

def train_and_evaluate_emotion_classifier(config: dict) -> Dict[str, float]:
    """
    Entrena y evalúa el clasificador; registra métricas y artefactos en MLflow.
    Devuelve métricas finales de test.
    """
    print("\n--- 🎭 Iniciando entrenamiento del Clasificador de Emociones... ---")

    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']

    # Semillas para reproducibilidad
    set_seed(RANDOM_STATE)

    # 1) Cargar base de datos
    df_base = load_base_dataset(EMOTION_LABELS)

    # Map de etiquetas
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    df_base['label'] = df_base['emotion'].map(label2id)

    # 2) Split estratificado
    train_val_df, test_df = train_test_split(
        df_base, test_size=0.20, random_state=RANDOM_STATE, stratify=df_base['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val_df['label']
    )

    # 3) Augmentation SOLO en el conjunto de entrenamiento
    if cfg_emo.get('data_augmentation', {}).get('use_back_translation', True):
        print("  › Aumentando SOLO train con Back-Translation...")
        df_train_aug = augment_with_back_translation(train_df[['text', 'emotion']])
        if not df_train_aug.empty:
            train_df = pd.concat([train_df, df_train_aug], ignore_index=True)
            train_df = train_df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
            train_df['label'] = train_df['emotion'].map(label2id)

    # 4) Tokenizador y datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'], use_fast=True)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])
    val_ds   = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds  = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_emo['model_name'], num_labels=len(EMOTION_LABELS),
        id2label=id2label, label2id=label2id
    )

    # 6) Pesos de clase
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # 7) Args de entrenamiento
    training_params = dict(cfg_emo['training_params'])
    if 'learning_rate' in training_params:
        training_params['learning_rate'] = float(training_params['learning_rate'])

    training_args = TrainingArguments(
        output_dir="./results_emotion_training",
        report_to="mlflow",
        run_name="train_emotion_classifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        **training_params
    )

    # 8) Entrenador
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg_emo.get('early_stopping_patience', 2))]
    )

    # 9) Entrenamiento y evaluación dentro de un run de MLflow
    with mlflow.start_run(run_name="train_emotion_classifier_final"):
        mlflow.log_params(cfg_emo['training_params'])
        mlflow.log_param("use_back_translation", cfg_emo.get('data_augmentation', {}).get('use_back_translation', True))
        mlflow.log_dict({'label2id': label2id, 'id2label': id2label}, "mappings/labels.json")

        print("\n--- 🚂 Entrenando modelo... ---")
        trainer.train()

        print("\n--- 📊 Evaluando en el conjunto de prueba... ---")
        predictions = trainer.predict(test_ds)
        y_pred_labels = np.argmax(predictions.predictions, axis=1)
        y_true_labels = np.array(list(test_ds['label']))

        y_pred = [id2label[i] for i in y_pred_labels]
        y_true = [id2label[i] for i in y_true_labels]

        report_txt = classification_report(y_true, y_pred, labels=list(label2id.keys()), zero_division=0)
        print("\n  › Reporte de Clasificación (TEST):")
        print(report_txt)
        mlflow.log_text(report_txt, "reports/classification_report_test.txt")

        final_metrics = compute_metrics_fn((predictions.predictions, y_true_labels))
        mlflow.log_metrics({f"final_test_{k}": v for k, v in final_metrics.items()})
        
        print(f"  › **F1-Score (Macro) Final:** {final_metrics['f1_macro']:.3f}")

        # --- Matriz de confusión como imagen ---
        try:
            import matplotlib.pyplot as plt
            cm = confusion_matrix(y_true, y_pred, labels=list(label2id.keys()))
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=list(label2id.keys()), yticklabels=list(label2id.keys()),
                   title='Matriz de Confusión (Test)',
                   ylabel='Etiqueta Real',
                   xlabel='Predicción')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            # Loop over data dimensions and create text annotations.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2. else "black")
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/confusion_matrix_test.png")
            plt.close(fig)
            print("  › Matriz de confusión registrada en MLflow.")
        except ImportError:
            warnings.warn("Matplotlib no está instalado. No se pudo generar la matriz de confusión.")
        except Exception as e:
            warnings.warn(f"No se pudo registrar la matriz de confusión: {e}")

        # 10) Guardar modelo final
        model_save_path = config['model_paths']['emotion_classifier']
        print(f"\n  › Guardando modelo en: {model_save_path}")
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        mlflow.log_artifacts(model_save_path, artifact_path="emotion_classifier_model")

    print("\n--- ✅ Pipeline del Clasificador de Emociones Finalizado. ---")
    return final_metrics
