"""Módulo de entrenamiento e inferencia para clasificación de emociones.

Versión final con corpus enriquecido, división estratificada y cálculo
robusto de pesos de clase, usando nomenclatura SIN tildes.
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

from datasets import Dataset, load_dataset
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
    hf_set_seed(seed)


def ensure_list(text: Union[str, List[str]]) -> List[str]:
    """Garantiza que la entrada de texto sea una lista de strings.

    Si la entrada es un solo string, lo envuelve en una lista.

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

    Esta clase proporciona métodos de alto nivel para obtener predicciones
    de un modelo de clasificación de secuencias ya entrenado.

    Attributes:
        device: El dispositivo computacional ('cuda' o 'cpu') donde se ejecuta el modelo.
        model: El modelo de Transformers cargado.
        tokenizer: El tokenizador asociado al modelo.
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        """Inicializa el clasificador de emociones.

        Args:
            model: Un modelo de secuencia de Hugging Face pre-entrenado.
            tokenizer: El tokenizador correspondiente al modelo.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict_proba(self, text: Union[str, List[str]]) -> List[Dict[str, float]]:
        """Predice la distribución de probabilidad de emociones para un texto.

        Args:
            text: Un string o una lista de strings a clasificar.

        Returns:
            Una lista de diccionarios. Cada diccionario mapea el nombre de una
            emoción a su probabilidad predicha. Utiliza .get() para manejar
            posibles IDs de etiqueta faltantes en la configuración del modelo.
        """
        texts = ensure_list(text)
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        id2label = self.model.config.id2label
        return [
            {id2label.get(str(i), f"UNK_{i}"): float(p) for i, p in enumerate(row)}
            for row in probabilities
        ]

# ==============================
# Datos y Aumento
# ==============================

def get_custom_domain_data() -> pd.DataFrame:
    """Crea y devuelve un corpus de dominio propio enriquecido (SIN tildes).

    Esta versión incluye ejemplos adicionales para las clases minoritarias
    'Alegria' y 'Anticipacion' para mejorar el balance del dataset.

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

def load_base_dataset(emotion_labels: List[str]) -> pd.DataFrame:
    """Carga y combina el dataset público 'emotion' con el de dominio propio.

    Utiliza el corpus de dominio enriquecido, filtra las emociones según
    `emotion_labels` y elimina duplicados de texto.

    Args:
        emotion_labels: Una lista de las emociones a incluir en el dataset final.

    Returns:
        Un DataFrame combinado y limpio con columnas ['text', 'emotion'].
    """
    print("  › Cargando dataset 'emotion' (HF) + dominio enriquecido...")
    try:
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        # Mapa consistente SIN tildes
        label_map = {0: 'Tristeza', 1: 'Alegria', 2: 'Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
        df_public['emotion'] = df_public['label'].map(label_map)
        df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]

        df_domain = get_custom_domain_data()

        df_base = pd.concat([df_public_clean, df_domain], ignore_index=True)
        # Filtramos ANTES de eliminar duplicados para asegurar que las etiquetas estén correctas
        df_base = df_base[df_base['emotion'].isin(emotion_labels)]
        df_base = df_base.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        
        print(f"  › Base combinada con {len(df_base)} ejemplos únicos.")
        return df_base
    except Exception as e:
        warnings.warn(f"No se pudo descargar el dataset público. Se usa solo dominio. Error: {e}")
        return get_custom_domain_data()

# ==============================
# Lógica de Entrenamiento Personalizada
# ==============================

class WeightedLossTrainer(Trainer):
    """Un Trainer de Hugging Face que utiliza CrossEntropyLoss ponderada.

    Esta clase sobrescribe el método `compute_loss` para aplicar pesos a
    las clases durante el cálculo de la pérdida, lo cual es útil para
    manejar datasets desbalanceados.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        """Inicializa el Trainer con pesos de clase opcionales.

        Args:
            class_weights: Un tensor de PyTorch con los pesos para cada clase.
            *args, **kwargs: Argumentos estándar del Trainer de Hugging Face.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        if self.class_weights is not None:
            # Imprimimos los pesos que realmente se están usando
            print(f"  › Pesos de clase aplicados (shape {self.class_weights.shape}): {np.round(self.class_weights.detach().cpu().numpy(), 3)}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Calcula la pérdida usando los pesos de clase proporcionados.

        Args:
            model: El modelo que se está entrenando.
            inputs: Un diccionario con los datos de entrada (incluyendo 'labels').
            return_outputs: Si True, devuelve también las salidas del modelo.

        Returns:
            La pérdida calculada (tensor) o una tupla (pérdida, salidas del modelo).
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Aseguramos que num_labels sea correcto
        num_labels = self.model.config.num_labels
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==============================
# Funciones Auxiliares del Pipeline
# ==============================

# Variables globales para acceso en compute_metrics_fn
id2label_global: Dict[int, str] = {}
config_global: Dict = {}

def compute_metrics_fn(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Calcula y devuelve las métricas de evaluación durante el entrenamiento.

    Utiliza las variables globales `id2label_global` y `config_global` para
    acceder a la configuración y mapeos necesarios. Calcula métricas macro
    y métricas detalladas por clase.

    Args:
        eval_pred: Una tupla que contiene los logits del modelo (predicciones)
                   y las etiquetas verdaderas (labels).

    Returns:
        Un diccionario con las métricas calculadas, incluyendo accuracy,
        precision_macro, recall_macro, f1_macro y F1-score por clase.
    """
    global id2label_global, config_global
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels) # Asegurar formato numpy

    # Obtener nombres y IDs de todas las clases posibles desde la config global
    all_label_names = config_global.get('constants', {}).get('emotion_labels', [])
    all_label_ids = np.arange(len(all_label_names))

    # Calcular métricas macro promediando sobre todas las clases posibles
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0,
        labels=all_label_ids # ¡Importante! Considerar todas las clases posibles
    )
    acc = accuracy_score(labels, preds)
    
    metrics = {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}
    
    # Calcular métricas detalladas por clase usando classification_report
    try:
        # Asegurarse de que target_names tenga la longitud correcta
        if len(all_label_names) == len(all_label_ids):
            report = classification_report(
                labels, preds, output_dict=True, zero_division=0,
                labels=all_label_ids, target_names=all_label_names
            )
            for label_name, scores in report.items():
                label_name_str = str(label_name) # Clave string
                if isinstance(scores, dict) and 'f1-score' in scores:
                     metrics[f"f1_{label_name_str}"] = scores['f1-score']
                elif label_name_str == 'accuracy': # Capturar accuracy global del reporte
                    metrics['report_accuracy'] = scores # Es un float aquí
                # Capturar promedios macro y ponderado del reporte para MLflow
                elif 'macro avg' in label_name_str or 'weighted avg' in label_name_str:
                     if isinstance(scores, dict):
                        metrics[f"{label_name_str.replace(' ', '_')}_precision"] = scores.get('precision', 0.0)
                        metrics[f"{label_name_str.replace(' ', '_')}_recall"] = scores.get('recall', 0.0)
                        metrics[f"{label_name_str.replace(' ', '_')}_f1"] = scores.get('f1-score', 0.0)
        else:
             warnings.warn("Inconsistencia entre all_label_names y all_label_ids. No se calcularán métricas por clase.")

    except ValueError as ve:
         warnings.warn(f"Error al calcular classification_report (posiblemente etiquetas no vistas): {ve}")
    except Exception as e:
        warnings.warn(f"No se pudo generar el reporte detallado por clase: {e}")

    return metrics


def build_compatible_training_args(training_params: dict) -> TrainingArguments:
    """Construye un objeto TrainingArguments de forma retrocompatible.

    Inspecciona la firma del constructor de `TrainingArguments` para filtrar
    parámetros desconocidos y manejar cambios de nombre (ej. 'evaluation_strategy'
    a 'eval_strategy').

    Args:
        training_params: Diccionario de parámetros cargado desde config.yaml.

    Returns:
        Una instancia de `TrainingArguments` configurada de forma segura.
    """
    valid_params = inspect.signature(TrainingArguments).parameters
    args = {
        "output_dir": "./results_emotion_training", "report_to": "mlflow",
        "run_name": "train_emotion_classifier", "load_best_model_at_end": True,
        "fp16": torch.cuda.is_available(),
    }
    # Asegurarse de que learning_rate es float antes de actualizar
    if 'learning_rate' in training_params:
        try:
            training_params['learning_rate'] = float(str(training_params['learning_rate']))
        except ValueError:
             warnings.warn(f"Valor inválido para learning_rate: {training_params['learning_rate']}. Usando default.")
             training_params.pop('learning_rate') # Eliminar para usar default del Trainer
             
    args.update(training_params)

    # Maneja el cambio de nombre de 'evaluation_strategy'
    if "evaluation_strategy" in args and "evaluation_strategy" not in valid_params:
        if "eval_strategy" in valid_params:
            args["eval_strategy"] = args.pop("evaluation_strategy")
            print("  › INFO: 'evaluation_strategy' renombrado a 'eval_strategy' por compatibilidad.")

    # Filtra argumentos no válidos para la versión actual de TrainingArguments
    final_args = {k: v for k, v in args.items() if k in valid_params}
    
    # Asegura la existencia de directorios de salida
    if 'logging_dir' in final_args: os.makedirs(final_args['logging_dir'], exist_ok=True)
    if 'output_dir' in final_args: os.makedirs(final_args['output_dir'], exist_ok=True)
    
    print(f"  › Argumentos de entrenamiento finales: {list(final_args.keys())}")
    return TrainingArguments(**final_args)


# ==============================
# Pipeline Principal de Entrenamiento
# ==============================

def train_and_evaluate_emotion_classifier(config: dict) -> Dict[str, float]:
    """Orquesta el pipeline completo de entrenamiento y evaluación del clasificador.

    Ejecuta todos los pasos: carga de datos, división estratificada,
    tokenización, entrenamiento con pesos de clase, evaluación y guardado.

    Args:
        config: Diccionario con la configuración completa del proyecto.

    Returns:
        Diccionario con las métricas finales obtenidas en el conjunto de prueba.
    """
    global id2label_global, config_global
    config_global = config # Hace la config accesible globalmente
    print("\n--- 🎭 Iniciando entrenamiento del Clasificador de Emociones (Final)... ---")

    # Configuración de MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Extracción de parámetros de configuración
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']

    set_seed(RANDOM_STATE) # Fija la semilla para reproducibilidad

    # Carga y preparación inicial de datos
    df_base = load_base_dataset(EMOTION_LABELS)
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    id2label_global = id2label # Guarda para uso en compute_metrics_fn
    df_base['label'] = df_base['emotion'].map(label2id)

    # División estratificada de datos
    print("  › Realizando división de datos ESTRATIFICADA...")
    train_val_df, test_df = train_test_split(
        df_base, test_size=0.20, random_state=RANDOM_STATE, stratify=df_base['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val_df['label']
    )

    # Tokenización
    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'], use_fast=True)
    def tokenize(batch):
        # Asegura que todos los textos sean strings válidos
        texts = [str(text) if text is not None else "" for text in batch['text']]
        return tokenizer(texts, truncation=True, max_length=128)

    # Conversión a formato Dataset y tokenización
    train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Carga del modelo pre-entrenado
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_emo['model_name'], num_labels=len(EMOTION_LABELS),
        # Asegura que las claves id2label sean strings para compatibilidad JSON
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id
    )

    # Cálculo robusto de pesos de clase
    print("  › Calculando pesos de clase de forma robusta...")
    unique_labels_in_train = np.unique(train_df['label'])
    if len(unique_labels_in_train) == 0:
        warnings.warn("El conjunto de entrenamiento está vacío o sin etiquetas. No se aplicarán pesos.")
        class_weights_tensor = None
    else:
        # Calcula pesos solo para las clases presentes en el train set
        class_weights_present = compute_class_weight(
            class_weight='balanced', classes=unique_labels_in_train, y=train_df['label']
        )
        label_to_weight_map = dict(zip(unique_labels_in_train, class_weights_present))
        
        # Construye el tensor final con tamaño num_classes, inicializado a 1.0
        num_classes = len(EMOTION_LABELS)
        final_weights = torch.ones(num_classes, dtype=torch.float)
        # Asigna los pesos calculados a las posiciones correspondientes
        for label_id, weight in label_to_weight_map.items():
            if 0 <= label_id < num_classes: # Chequeo de seguridad
                final_weights[label_id] = float(weight)
        class_weights_tensor = final_weights
    
    # Construcción de argumentos de entrenamiento
    training_params = dict(cfg_emo['training_params'])
    training_args = build_compatible_training_args(training_params)

    # Inicialización del Trainer personalizado
    trainer = WeightedLossTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=data_collator,
        compute_metrics=compute_metrics_fn, class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg_emo.get('early_stopping_patience', 2))]
    )

    # Inicio del run de MLflow para registrar el entrenamiento
    with mlflow.start_run(run_name="train_emotion_classifier_final"):
        # Registra artefactos y parámetros iniciales
        mlflow.log_dict({'label2id': label2id, 'id2label': id2label}, "mappings/labels.json")
        # Nota: Los training_params se loguean automáticamente por el Trainer

        print("\n--- 🚂 Entrenando modelo... ---")
        train_result = trainer.train() # Guarda el resultado del entrenamiento

        # Evaluación final en el conjunto de prueba
        print("\n--- 📊 Evaluando en el conjunto de prueba... ---")
        predictions = trainer.predict(test_ds)
        y_pred_labels = np.argmax(predictions.predictions, axis=1)
        y_true_labels = np.array(test_ds["label"]) # Extraer labels del dataset

        # Mapeo de IDs a nombres para el reporte y matriz de confusión
        y_pred = [id2label.get(i, f"UNK_{i}") for i in y_pred_labels]
        y_true = [id2label.get(i, f"UNK_{i}") for i in y_true_labels]

        # Generación y registro del reporte de clasificación
        report_txt = classification_report(
            y_true, y_pred, labels=EMOTION_LABELS, zero_division=0
        )
        print("\n  › Reporte de Clasificación (TEST):")
        print(report_txt)
        mlflow.log_text(report_txt, "reports/classification_report_test.txt")

        # Cálculo y registro de métricas finales
        final_metrics = compute_metrics_fn((predictions.predictions, y_true_labels))
        # Prefijo para distinguir métricas finales de las de evaluación
        mlflow.log_metrics({f"final_test_{k}": v for k, v in final_metrics.items()})
        
        print(f"  › **F1-Score (Macro) Final:** {final_metrics.get('f1_macro', 0.0):.3f}")

        # Generación y registro de la matriz de confusión
        try:
            import matplotlib.pyplot as plt
            # Usar las etiquetas presentes en y_true o y_pred para la matriz
            cm_labels = sorted(list(set(y_true).union(set(y_pred))), key=lambda x: label2id.get(x, -1))
            if not cm_labels: cm_labels = EMOTION_LABELS # Fallback

            cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
            fig, ax = plt.subplots(figsize=(max(8, len(cm_labels)*1.2), max(6, len(cm_labels)*0.9)))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=cm_labels, yticklabels=cm_labels,
                   title='Matriz de Confusión (Test)', ylabel='Etiqueta Real', xlabel='Predicción')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            thresh = cm.max() / 2. if cm.max() > 0 else 0.1
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/confusion_matrix_test.png")
            plt.close(fig)
            print("  › Matriz de confusión registrada en MLflow.")
        except ImportError:
            warnings.warn("Matplotlib no disponible. No se generará la matriz de confusión.")
        except Exception as e:
            warnings.warn(f"No se pudo registrar la matriz de confusión: {e}")

        # Guardado final del modelo y tokenizador
        model_save_path = config['model_paths']['emotion_classifier']
        print(f"\n  › Guardando modelo y tokenizador en: {model_save_path}")
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        # Registro del modelo como artefacto en MLflow
        mlflow.log_artifacts(model_save_path, artifact_path="emotion_classifier_model")

    print("\n--- ✅ Pipeline del Clasificador de Emociones (Final) Finalizado. ---")
    return final_metrics if 'final_metrics' in locals() else {}

