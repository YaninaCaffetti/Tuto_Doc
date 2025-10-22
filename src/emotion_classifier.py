"""Módulo de entrenamiento e inferencia para clasificación de emociones.

Versión final con corpus enriquecido, división estratificada, cálculo
robusto de pesos de clase, y aumento opcional por retrotraducción.
Usa nomenclatura SIN tildes.
"""
from __future__ import annotations

import os
import inspect
import json
import random
import warnings
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# Importación segura de datasets
try:
    from datasets import Dataset, load_dataset
except ImportError:
    warnings.warn("La librería `datasets` no está instalada. Algunas funciones pueden fallar.")
    Dataset = None
    load_dataset = None

# Importación segura de transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        set_seed as hf_set_seed,
        MarianMTModel,
        MarianTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    warnings.warn("La librería `transformers` no está instalada. Funciones clave no estarán disponibles.")
    TRANSFORMERS_AVAILABLE = False
    # Definir clases dummy si transformers no está
    class DummyAutoClass: pass
    AutoTokenizer = DummyAutoClass
    AutoModelForSequenceClassification = DummyAutoClass
    TrainingArguments = DummyAutoClass
    Trainer = DummyAutoClass
    DataCollatorWithPadding = DummyAutoClass
    EarlyStoppingCallback = DummyAutoClass
    MarianMTModel = DummyAutoClass
    MarianTokenizer = DummyAutoClass
    def hf_set_seed(seed): pass


import mlflow


# ==============================
# Utilidades
# ==============================

def set_seed(seed: int) -> None:
    """Fija las semillas aleatorias para garantizar la reproducibilidad.

    Aplica la semilla a las librerías `random`, `numpy`, `torch` y `transformers`.

    Args:
        seed: El número entero a usar como semilla.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if TRANSFORMERS_AVAILABLE:
        hf_set_seed(seed)


def ensure_list(text: Union[str, List[str]]) -> List[str]:
    """Garantiza que la entrada de texto sea una lista de strings.

    Args:
        text: Un string o una lista de strings.

    Returns:
        Una lista de strings.

    Raises:
        TypeError: Si la entrada no es un string o una lista de strings.
    """
    if isinstance(text, str):
        return [text]
    if isinstance(text, list) and all(isinstance(t, str) for t in text):
        return text
    raise TypeError("La entrada `text` debe ser de tipo str o List[str].")


# ==============================
# Inferencia
# ==============================

class EmotionClassifier:
    """Encapsula un modelo de Transformers para la inferencia de emociones.

    Attributes:
        device: El dispositivo computacional ('cuda' o 'cpu').
        model: El modelo de Transformers cargado.
        tokenizer: El tokenizador asociado al modelo.
        id2label: Mapeo de ID (string) a nombre de etiqueta (string).
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        """Inicializa el clasificador de emociones.

        Args:
            model: Un modelo de secuencia de Hugging Face pre-entrenado.
            tokenizer: El tokenizador correspondiente al modelo.
        """
        if not TRANSFORMERS_AVAILABLE:
             raise ImportError("Transformers no está instalado. EmotionClassifier no puede funcionar.")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        # Asegura que id2label use claves string para consistencia
        self.id2label = {str(k): v for k, v in self.model.config.id2label.items()}
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")
        print(f"  › Mapa id2label cargado: {self.id2label}") # Log para verificar

    def predict_proba(self, text: Union[str, List[str]]) -> List[Dict[str, float]]:
        """Predice la distribución de probabilidad de emociones para un texto.

        Utiliza una lógica robusta para mapear los índices de salida del modelo
        a las etiquetas de emoción, manejando posibles inconsistencias.

        Args:
            text: Un string o una lista de strings a clasificar.

        Returns:
            Lista de diccionarios {emoción: probabilidad}.
        """
        texts = ensure_list(text)
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        
        results = []
        for row in probabilities:
            prob_dict = {}
            for i, p in enumerate(row):
                # Lógica de traducción robusta
                label_key = str(i)
                label_name = self.id2label.get(label_key, f"Etiqueta_{i}_DESCONOCIDA")
                prob_dict[label_name] = float(p)
            results.append(prob_dict)
            
        # Log de depuración si se detecta una etiqueta desconocida
        if any("DESCONOCIDA" in label for res_dict in results for label in res_dict):
             warnings.warn(f"Etiqueta desconocida detectada. Mapa: {self.id2label}. Probs: {probabilities}")
                           
        return results

# ==============================
# Datos y Aumento
# ==============================

def get_custom_domain_data() -> pd.DataFrame:
    """Crea y devuelve un corpus de dominio propio enriquecido (SIN tildes).

    Returns:
        Un DataFrame de Pandas con columnas ['text', 'emotion'].
    """
    data_custom_list = [
        # Corpus original y enriquecido SIN tildes
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipacion"),
        ("No puedo esperar a ver los resultados de este experimento.", "Anticipacion"), ("Esto es inaceptable, nadie responde mis correos.", "Ira"),
        ("Perdimos el financiamiento del proyecto.", "Tristeza"), ("El prototipo funciona mejor de lo que esperaba.", "Alegria"),
        ("Confío en que el equipo resolverá esto.", "Confianza"), ("Me preocupa no cumplir con el plazo de entrega.", "Miedo"),
        ("¡No puedo creer que aprobamos la auditoría!", "Sorpresa"), ("Tengo muchas ganas de empezar la capacitación.", "Anticipacion"),
        ("Con el CUD podré acceder a más beneficios.", "Confianza"), ("Gracias, la orientación me devolvió el ánimo.", "Alegria"),
        ("Basta de demoras!.", "Ira"),
        ("¡Conseguí el trabajo! No puedo creerlo, estoy tan feliz.", "Alegria"),
        ("El proyecto fue un éxito total, todo el equipo está celebrando.", "Alegria"),
        ("Me acaban de confirmar la beca. ¡Qué gran noticia!", "Alegria"),
        ("Finalmente logré superar ese obstáculo, me siento increíble.", "Alegria"),
        ("Recibir este reconocimiento me llena de orgullo y felicidad.", "Alegria"),
        ("Hoy es un día fantástico, todo está saliendo a la perfección.", "Alegria"),
        ("Qué alegria ver que mi esfuerzo está dando frutos.", "Alegria"),
        ("Estoy muy contento con los resultados obtenidos.", "Alegria"),
        ("¡Lo logramos! El plan funcionó mejor de lo esperado.", "Alegria"),
        ("Me siento optimista y lleno de energía positiva.", "Alegria"),
        ("Mañana es la entrevista final, estoy nervioso pero expectante.", "Anticipacion"),
        ("Falta solo una semana para el lanzamiento del proyecto.", "Anticipacion"),
        ("Estoy ansioso por empezar este nuevo curso.", "Anticipacion"),
        ("Ya quiero ver cómo reaccionarán cuando presente la propuesta.", "Anticipacion"),
        ("La espera para conocer los resultados me tiene en vilo.", "Anticipacion"),
        ("Contando los días para la conferencia, seguro aprenderé mucho.", "Anticipacion"),
        ("Tengo muchas expectativas sobre esta nueva etapa.", "Anticipacion"),
        ("¿Qué sorpresas nos deparará la reunión de mañana?", "Anticipacion"),
        ("Estoy a punto de recibir el feedback, espero que sea bueno.", "Anticipacion"),
        ("La próxima fase del proyecto promete ser muy interesante.", "Anticipacion"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def augment_with_back_translation(df: pd.DataFrame, lang_src: str = 'es', lang_tgt: str = 'en') -> pd.DataFrame:
    """Aumenta un DataFrame de textos mediante retrotraducción.

    Args:
        df: DataFrame que debe contener una columna 'text' y 'emotion'.
        lang_src: Código del idioma de origen (ej. 'es').
        lang_tgt: Código del idioma de destino (ej. 'en').

    Returns:
        Un nuevo DataFrame con los textos aumentados y sus emociones originales.
    """
    if not TRANSFORMERS_AVAILABLE:
        warnings.warn("Transformers no disponible. Se omite la retrotraducción.")
        return pd.DataFrame(columns=['text', 'emotion'])
        
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  ›› Realizando Back-Translation en: {device.upper()}")

        model_name_src_tgt = f'Helsinki-NLP/opus-mt-{lang_src}-{lang_tgt}'
        model_name_tgt_src = f'Helsinki-NLP/opus-mt-{lang_tgt}-{lang_src}'

        tok_st = MarianTokenizer.from_pretrained(model_name_src_tgt)
        mt_st = MarianMTModel.from_pretrained(model_name_src_tgt).to(device)
        tok_ts = MarianTokenizer.from_pretrained(model_name_tgt_src)
        mt_ts = MarianMTModel.from_pretrained(model_name_tgt_src).to(device)

        augmented_texts = []
        disable_tqdm = len(df) < 50 
        for text in tqdm(df['text'], desc="Generando paráfrasis", leave=False, 
                         mininterval=1.0, disable=disable_tqdm):
            if not isinstance(text, str) or not text.strip():
                augmented_texts.append("") 
                continue
                
            inputs = tok_st(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            translated_ids = mt_st.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
            text_tgt = tok_st.decode(translated_ids[0], skip_special_tokens=True)

            inputs_back = tok_ts(text_tgt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            back_ids = mt_ts.generate(**inputs_back, max_new_tokens=128, num_beams=4, early_stopping=True)
            text_back = tok_ts.decode(back_ids[0], skip_special_tokens=True)
            augmented_texts.append(text_back)
        
        return pd.DataFrame({'text': augmented_texts, 'emotion': df['emotion'].values})
    except Exception as e:
        warnings.warn(f"Back-Translation falló; se omite. Error: {e}")
        return pd.DataFrame(columns=['text', 'emotion'])


def load_base_dataset(emotion_labels: List[str]) -> pd.DataFrame:
    """Carga y combina el dataset público 'emotion' con el de dominio propio.

    Args:
        emotion_labels: Lista de emociones a incluir.

    Returns:
        Un DataFrame combinado y limpio con columnas ['text', 'emotion'].
    """
    print("  › Cargando dataset 'emotion' (HF) + dominio enriquecido...")
    
    df_public_clean = pd.DataFrame(columns=['text', 'emotion'])
    if load_dataset:
        try:
            dataset = load_dataset("emotion", split='train')
            df_public = dataset.to_pandas()
            label_map = {0: 'Tristeza', 1: 'Alegria', 2: 'Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
            df_public['emotion'] = df_public['label'].map(label_map)
            df_public_clean = df_public[['text', 'emotion']].dropna(subset=['emotion'])
        except Exception as e:
            warnings.warn(f"No se pudo descargar/procesar 'emotion'. Error: {e}")

    df_domain = get_custom_domain_data()
    df_base = pd.concat([df_public_clean, df_domain], ignore_index=True)
    df_base = df_base[df_base['emotion'].isin(emotion_labels)]
    df_base = df_base.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    
    print(f"  › Base combinada con {len(df_base)} ejemplos únicos.")
    return df_base

# ==============================
# Lógica de Entrenamiento Personalizada
# ==============================

class WeightedLossTrainer(Trainer):
    """Trainer con CrossEntropyLoss ponderada por clase."""
    def __init__(self, *args, class_weights=None, **kwargs):
        """Inicializa el Trainer con pesos de clase opcionales."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers no está instalado. WeightedLossTrainer no puede funcionar.")
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        if self.class_weights is not None:
            print(f"  › Pesos de clase aplicados (shape {self.class_weights.shape}): {np.round(self.class_weights.detach().cpu().numpy(), 3)}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Calcula la pérdida usando los pesos de clase proporcionados."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        num_labels = self.model.config.num_labels
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==============================
# Funciones Auxiliares del Pipeline
# ==============================

id2label_global: Dict[int, str] = {}
config_global: Dict = {}

def compute_metrics_fn(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Calcula y devuelve las métricas de evaluación durante el entrenamiento."""
    global id2label_global, config_global
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)

    all_label_names = config_global.get('constants', {}).get('emotion_labels', [])
    all_label_ids = np.arange(len(all_label_names))

    # Métricas Macro
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0, labels=all_label_ids
    )
    acc = accuracy_score(labels, preds)
    metrics = {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}
    
    # Métricas por clase
    try:
        if len(all_label_names) == len(all_label_ids):
            report = classification_report(
                labels, preds, output_dict=True, zero_division=0,
                labels=all_label_ids, target_names=all_label_names
            )
            for label_name, scores in report.items():
                label_name_str = str(label_name).replace(' ', '_')
                if isinstance(scores, dict) and 'f1-score' in scores:
                     metrics[f"f1_{label_name_str}"] = scores['f1-score']
                elif label_name_str == 'accuracy':
                    metrics['report_accuracy'] = scores
                elif 'avg' in label_name_str:
                     if isinstance(scores, dict):
                        metrics[f"{label_name_str}_precision"] = scores.get('precision', 0.0)
                        metrics[f"{label_name_str}_recall"] = scores.get('recall', 0.0)
                        metrics[f"{label_name_str}_f1"] = scores.get('f1-score', 0.0)
        else:
             warnings.warn("Inconsistencia etiquetas-IDs para reporte detallado.")
    except Exception as e:
        warnings.warn(f"Error al calcular reporte detallado: {e}")
    return metrics


def build_compatible_training_args(training_params: dict) -> TrainingArguments:
    """Construye un objeto TrainingArguments de forma retrocompatible."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers no está instalado. No se pueden construir TrainingArguments.")
        
    valid_params = inspect.signature(TrainingArguments).parameters
    args = {
        "output_dir": "./results_emotion_training", "report_to": "mlflow",
        "run_name": "train_emotion_classifier", "load_best_model_at_end": True,
        "fp16": torch.cuda.is_available(),
    }
    if 'learning_rate' in training_params:
        try:
            training_params['learning_rate'] = float(str(training_params['learning_rate']))
        except ValueError:
             warnings.warn(f"LR inválido: {training_params['learning_rate']}. Usando default.")
             training_params.pop('learning_rate')
    args.update(training_params)

    if "evaluation_strategy" in args and "evaluation_strategy" not in valid_params:
        if "eval_strategy" in valid_params:
            args["eval_strategy"] = args.pop("evaluation_strategy")
            print("  › INFO: 'evaluation_strategy' renombrado a 'eval_strategy'.")

    final_args = {k: v for k, v in args.items() if k in valid_params}
    if 'logging_dir' in final_args: os.makedirs(final_args['logging_dir'], exist_ok=True)
    if 'output_dir' in final_args: os.makedirs(final_args['output_dir'], exist_ok=True)
    print(f"  › Argumentos de entrenamiento finales: {list(final_args.keys())}")
    return TrainingArguments(**final_args)


# ==============================
# Pipeline Principal de Entrenamiento
# ==============================

def train_and_evaluate_emotion_classifier(config: dict) -> Dict[str, float]:
    """Orquesta el pipeline completo de entrenamiento y evaluación del clasificador."""
    global id2label_global, config_global
    config_global = config
    print("\n--- 🎭 Iniciando entrenamiento del Clasificador de Emociones (Final)... ---")

    if not TRANSFORMERS_AVAILABLE:
        print("❌ Error: La librería Transformers es necesaria para entrenar el modelo.")
        return {}

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']

    set_seed(RANDOM_STATE)

    df_base = load_base_dataset(EMOTION_LABELS)
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    id2label_global = id2label
    df_base['label'] = df_base['emotion'].map(label2id).fillna(-1).astype(int)
    df_base = df_base[df_base['label'] != -1]

    # --- ¡DIVISIÓN ESTRATIFICADA ACTIVADA! ---
    print("  › Realizando división de datos ESTRATIFICADA...")
    train_val_df, test_df = train_test_split(
        df_base, test_size=0.20, random_state=RANDOM_STATE, stratify=df_base['label']
    )
    if train_val_df['label'].nunique() > 1:
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val_df['label']
        )
    else:
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.125, random_state=RANDOM_STATE
        )
        warnings.warn("Solo una clase en train/val split. Estratificación no aplicada.")


    # --- RETROTRADUCCIÓN RESTAURADA Y OPCIONAL ---
    if cfg_emo.get('data_augmentation', {}).get('use_back_translation', False):
        print("  › Aumentando SOLO train con Back-Translation...")
        df_to_augment = train_df[['text', 'emotion']].copy()
        df_train_aug = augment_with_back_translation(df_to_augment)
        
        if not df_train_aug.empty:
            df_train_aug['label'] = df_train_aug['emotion'].map(label2id).fillna(-1).astype(int)
            df_train_aug = df_train_aug[df_train_aug['label'] != -1]
            original_len = len(train_df)
            train_df = pd.concat([train_df, df_train_aug], ignore_index=True)
            train_df = train_df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
            print(f"  › Tamaño de train tras aumento BT: {len(train_df)} (añadidos {len(train_df) - original_len})")
            
    # Tokenización
    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'], use_fast=True)
    def tokenize(batch):
        texts = [str(text) if text is not None else "" for text in batch['text']]
        return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

    train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_emo['model_name'], num_labels=len(EMOTION_LABELS),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )

    # --- CÁLCULO DE PESOS ROBUSTO ---
    print("  › Calculando pesos de clase de forma robusta...")
    unique_labels_in_train, counts = np.unique(train_df['label'], return_counts=True)
    
    if len(unique_labels_in_train) == 0:
        warnings.warn("Train set vacío/sin etiquetas. No se aplicarán pesos.")
        class_weights_tensor = None
    elif np.any(counts == 0):
         warnings.warn("Clases con 0 instancias en train_df. No se aplicarán pesos.")
         class_weights_tensor = None
    else:
        class_weights_present = compute_class_weight(
            class_weight='balanced', classes=unique_labels_in_train, y=train_df['label']
        )
        label_to_weight_map = dict(zip(unique_labels_in_train, class_weights_present))
        num_classes = len(EMOTION_LABELS)
        final_weights = torch.ones(num_classes, dtype=torch.float)
        for label_id, weight in label_to_weight_map.items():
            if isinstance(label_id, (int, np.integer)) and 0 <= label_id < num_classes:
                final_weights[label_id] = float(weight)
            else:
                 warnings.warn(f"ID de etiqueta inválido ({label_id}). Se ignora para pesos.")
        class_weights_tensor = final_weights
    
    training_params = dict(cfg_emo['training_params'])
    training_args = build_compatible_training_args(training_params)

    trainer = WeightedLossTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=data_collator,
        compute_metrics=compute_metrics_fn, class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg_emo.get('early_stopping_patience', 2))]
    )

    with mlflow.start_run(run_name="train_emotion_classifier_final"):
        mlflow.log_dict({'label2id': label2id, 
                         'id2label': {str(k): v for k, v in id2label.items()}}, 
                        "mappings/labels.json")
        mlflow.log_param("used_back_translation", cfg_emo.get('data_augmentation', {}).get('use_back_translation', False))


        print("\n--- 🚂 Entrenando modelo... ---")
        try:
             train_result = trainer.train() 
        except Exception as train_error:
             logging.error(f"Error durante trainer.train(): {train_error}")
             logging.error(traceback.format_exc())
             return {}

        print("\n--- 📊 Evaluando en el conjunto de prueba... ---")
        try:
            predictions = trainer.predict(test_ds)
        except Exception as predict_error:
             logging.error(f"Error durante trainer.predict(): {predict_error}")
             logging.error(traceback.format_exc())
             return {}
             
        y_pred_labels = np.argmax(predictions.predictions, axis=1)
        y_true_labels = np.array(test_ds["label"])

        y_pred = [id2label.get(i, f"UNK_{i}") for i in y_pred_labels]
        y_true = [id2label.get(i, f"UNK_{i}") for i in y_true_labels]

        # Asegurar que EMOTION_LABELS coincida con las claves de label2id
        report_labels = list(label2id.keys())
        report_txt = classification_report(
            y_true, y_pred, labels=report_labels, zero_division=0,
            target_names=report_labels
        )
        print("\n  › Reporte de Clasificación (TEST):")
        print(report_txt)
        mlflow.log_text(report_txt, "reports/classification_report_test.txt")

        final_metrics = compute_metrics_fn((predictions.predictions, y_true_labels))
        mlflow.log_metrics({f"final_test_{k}": v for k, v in final_metrics.items()})
        
        print(f"  › **F1-Score (Macro) Final:** {final_metrics.get('f1_macro', 0.0):.3f}")

        try:
            import matplotlib.pyplot as plt
            cm_labels = report_labels # Usar las mismas etiquetas que el reporte
            cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
            fig, ax = plt.subplots(figsize=(max(8, len(cm_labels)*1.2), max(6, len(cm_labels)*0.9)))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=cm_labels, yticklabels=cm_labels,
                   title='Matriz de Confusión (Test)', ylabel='Etiqueta Real', xlabel='Predicción')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            thresh = cm.max() / 1.5 if cm.max() > 0 else 0.1 
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/confusion_matrix_test.png")
            plt.close(fig)
            print("  › Matriz de confusión registrada en MLflow.")
        except ImportError:
            warnings.warn("Matplotlib no disponible.")
        except Exception as e:
            warnings.warn(f"No se pudo registrar matriz de confusión: {e}")

        model_save_path = config['model_paths']['emotion_classifier']
        print(f"\n  › Guardando modelo y tokenizador en: {model_save_path}")
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path) 
        tokenizer.save_pretrained(model_save_path) 
        mlflow.log_artifacts(model_save_path, artifact_path="emotion_classifier_model") 

    print("\n--- ✅ Pipeline del Clasificador de Emociones (Final) Finalizado. ---")
    # Devolver las métricas finales calculadas
    return final_metrics if 'final_metrics' in locals() else {}

