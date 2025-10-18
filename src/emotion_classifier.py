"""Módulo de entrenamiento e inferencia para clasificación de emociones.

Este script encapsula un pipeline completo para entrenar un modelo de Transformers
capaz de clasificar texto en español según un conjunto predefinido de emociones.

Funcionalidades Principales:
- Carga y combinación de datasets (público y de dominio propio).
- Aumento de datos en el conjunto de entrenamiento mediante back-translation.
- División de datos estratificada para garantizar la representatividad de las clases.
- Entrenamiento de un modelo Hugging Face con una función de pérdida ponderada
  para mitigar el desbalance de clases.
- Evaluación del modelo con métricas estándar (Accuracy, Precision, Recall, F1-Score).
- Integración con MLflow para el registro de parámetros, métricas y artefactos.
- Tolerancia a versiones de la librería `transformers` mediante un constructor
  de argumentos compatible.
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
            emoción a su probabilidad predicha.
        """
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
        """Predice la emoción dominante para un único texto.

        Args:
            text: El string a clasificar.
            return_index: Si es True, devuelve el índice de la clase en lugar
                de su nombre. Por defecto es False.

        Returns:
            El nombre (string) de la emoción con la probabilidad más alta, o
            su índice (int) si `return_index` es True.
        """
        probs_list = self.predict_proba(text)[0]
        labels = list(probs_list.keys())
        values = list(probs_list.values())
        idx = int(np.argmax(values))
        return idx if return_index else labels[idx]


# ==============================
# Datos y Aumento
# ==============================

def get_custom_domain_data() -> pd.DataFrame:
    """Crea y devuelve un pequeño corpus de dominio propio.

    Returns:
        Un DataFrame de Pandas con columnas ['text', 'emotion'].
    """
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"),
        ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"),
        ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"),
        ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipacion"),
        ("No puedo esperar a ver los resultados de este experimento.", "Anticipacion"),
        ("Esto es inaceptable, nadie responde mis correos.", "Ira"),
        ("Perdimos el financiamiento del proyecto.", "Tristeza"),
        ("El prototipo funciona mejor de lo que esperaba.", "Alegria"),
        ("Confío en que el equipo resolverá esto.", "Confianza"),
        ("Me preocupa no cumplir con el plazo de entrega.", "Miedo"),
        ("¡No puedo creer que aprobamos la auditoría!", "Sorpresa"),
        ("Tengo muchas ganas de empezar la capacitación la semana próxima.", "Anticipacion"),
        ("Con el CUD podré acceder a más beneficios para postularme.", "Confianza"),
        ("Gracias, la orientación me devolvió el ánimo.", "Alegria"),
        ("Basta de demoras!.", "Ira"),
        ("¡Conseguí el trabajo! No puedo creerlo, estoy tan feliz.", "Alegria"),
        ("El proyecto fue un éxito total, todo el equipo está celebrando.", "Alegria"),
        ("Me acaban de confirmar la beca. ¡Qué gran noticia!", "Alegria"),
        ("Finalmente logré superar ese obstáculo, me siento increíble.", "Alegria"),
        ("Recibir este reconocimiento me llena de orgullo y felicidad.", "Alegria"),
        ("Hoy es un día fantástico, todo está saliendo a la perfección.", "Alegria"),
        ("Qué alegría ver que mi esfuerzo está dando frutos.", "Alegria"),
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

    Traduce cada texto de un idioma de origen a uno de destino y luego de
    vuelta al origen para generar paráfrasis.

    Args:
        df: DataFrame que debe contener una columna 'text' y 'emotion'.
        lang_src: Código del idioma de origen (ej. 'es').
        lang_tgt: Código del idioma de destino (ej. 'en').

    Returns:
        Un nuevo DataFrame con los textos aumentados y sus emociones originales.
        Devuelve un DataFrame vacío si el proceso falla.
    """
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
        for text in tqdm(df['text'], desc="Retrotraduciendo frases", mininterval=1.0):
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
    """Carga y combina el dataset público 'emotion' con el de dominio propio.

    Filtra las emociones según `emotion_labels` y elimina duplicados.

    Args:
        emotion_labels: Una lista de las emociones a incluir en el dataset final.

    Returns:
        Un DataFrame combinado y limpio con columnas ['text', 'emotion'].
    """
    print("  › Cargando dataset 'emotion' (HF) + dominio...")
    try:
        # El parámetro trust_remote_code se elimina por ser obsoleto.
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        label_map = {0: 'Tristeza', 1: 'Alegria', 2: 'Amor/Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa/Anticipacion'}
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
# Lógica de Entrenamiento Personalizada
# ==============================

class WeightedLossTrainer(Trainer):
    """Un Trainer de Hugging Face que utiliza CrossEntropyLoss ponderada.

    Esta clase sobrescribe el método `compute_loss` para aplicar pesos a
    las clases durante el cálculo de la pérdida, lo cual es útil para
    manejar datasets desbalanceados.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        """Inicializa el Trainer con pesos de clase opcionales."""
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        if self.class_weights is not None:
            print(f"  › Pesos de clase aplicados: {np.round(self.class_weights.detach().cpu().numpy(), 3)}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Calcula la pérdida usando los pesos de clase proporcionados."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ==============================
# Funciones Auxiliares del Pipeline
# ==============================

def compute_metrics_fn(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Calcula y devuelve las métricas de evaluación.

    Se utiliza durante el entrenamiento para monitorizar el rendimiento en el
    conjunto de validación.

    Args:
        eval_pred: Una tupla que contiene los logits del modelo y las
            etiquetas verdaderas.

    Returns:
        Un diccionario con las métricas calculadas: accuracy, precision_macro,
        recall_macro y f1_macro.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}


def build_compatible_training_args(training_params: dict) -> TrainingArguments:
    """Construye un objeto TrainingArguments de forma retrocompatible.

    Esta función inspecciona la firma del constructor de `TrainingArguments`
    en la versión instalada de `transformers` para filtrar parámetros no
    reconocidos y manejar cambios de nombre (ej. 'evaluation_strategy' a
    'eval_strategy'), haciendo el pipeline más robusto a actualizaciones.

    Args:
        training_params: Diccionario de parámetros cargado desde config.yaml.

    Returns:
        Una instancia de `TrainingArguments` configurada de forma segura.
    """
    valid_params = inspect.signature(TrainingArguments).parameters
    args = {
        "output_dir": "./results_emotion_training",
        "report_to": "mlflow",
        "run_name": "train_emotion_classifier",
        "load_best_model_at_end": True,
        "fp16": torch.cuda.is_available(),
    }
    args.update(training_params)

    # Maneja el cambio de nombre de 'evaluation_strategy' por retrocompatibilidad.
    if "evaluation_strategy" in args and "evaluation_strategy" not in valid_params:
        if "eval_strategy" in valid_params:
            args["eval_strategy"] = args.pop("evaluation_strategy")
            print("  › INFO: 'evaluation_strategy' renombrado a 'eval_strategy' por compatibilidad.")

    final_args = {k: v for k, v in args.items() if k in valid_params}
    print(f"  › Argumentos de entrenamiento finales: {list(final_args.keys())}")
    return TrainingArguments(**final_args)


# ==============================
# Pipeline Principal de Entrenamiento
# ==============================

def train_and_evaluate_emotion_classifier(config: dict) -> Dict[str, float]:
    """Orquesta el pipeline completo de entrenamiento y evaluación.

    Este es el punto de entrada principal que ejecuta todos los pasos:
    1. Carga y preprocesa los datos.
    2. Realiza la división estratificada en conjuntos de entrenamiento,
       validación y prueba.
    3. Aplica aumento de datos.
    4. Configura y entrena el modelo de Transformers.
    5. Evalúa el modelo final en el conjunto de prueba.
    6. Registra todos los resultados y artefactos en MLflow.
    7. Guarda el modelo entrenado en disco.

    Args:
        config: Un diccionario cargado desde el archivo config.yaml con toda
            la configuración del proyecto.

    Returns:
        Un diccionario con las métricas finales obtenidas en el conjunto de prueba.
    """
    print("\n--- 🎭 Iniciando entrenamiento del Clasificador de Emociones... ---")

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']

    set_seed(RANDOM_STATE)

    df_base = load_base_dataset(EMOTION_LABELS)
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    df_base['label'] = df_base['emotion'].map(label2id)

    print("  › Realizando división de datos ESTRATIFICADA...")
    train_val_df, test_df = train_test_split(
        df_base,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df_base['label']
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125, # 12.5% de 80% es 10% del total para validación.
        random_state=RANDOM_STATE,
        stratify=train_val_df['label']
    )

    if cfg_emo.get('data_augmentation', {}).get('use_back_translation', True):
        print("  › Aumentando SOLO train con Back-Translation...")
        df_train_aug = augment_with_back_translation(train_df[['text', 'emotion']])
        if not df_train_aug.empty:
            df_train_aug['label'] = df_train_aug['emotion'].map(label2id)
            train_df = pd.concat([train_df, df_train_aug], ignore_index=True)
            train_df = train_df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'], use_fast=True)
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_emo['model_name'], num_labels=len(EMOTION_LABELS),
        id2label=id2label, label2id=label2id
    )

    # Lógica robusta para el cálculo de pesos de clase.
    # Garantiza que el tensor de pesos siempre tenga un elemento por cada
    # clase definida, incluso si algunas clases no están presentes en el
    # split de entrenamiento, evitando el RuntimeError.
    print("  › Calculando pesos de clase de forma robusta...")
    unique_labels_in_train = np.unique(train_df['label'])
    class_weights_present = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels_in_train,
        y=train_df['label']
    )
    label_to_weight_map = dict(zip(unique_labels_in_train, class_weights_present))
    num_classes = len(EMOTION_LABELS)
    final_weights = torch.ones(num_classes, dtype=torch.float)
    for label_id, weight in label_to_weight_map.items():
        if label_id < num_classes:
            final_weights[label_id] = float(weight)
    class_weights_tensor = final_weights
    
    training_params = dict(cfg_emo['training_params'])
    if 'learning_rate' in training_params:
        training_params['learning_rate'] = float(str(training_params['learning_rate']))
    training_args = build_compatible_training_args(training_params)

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

    with mlflow.start_run(run_name="train_emotion_classifier_final"):
        # El log de parámetros manual se elimina para evitar el error de duplicación,
        # ya que el Trainer de Hugging Face lo hace automáticamente.
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

        model_save_path = config['model_paths']['emotion_classifier']
        print(f"\n  › Guardando modelo en: {model_save_path}")
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        mlflow.log_artifacts(model_save_path, artifact_path="emotion_classifier_model")

    print("\n--- ✅ Pipeline del Clasificador de Emociones Finalizado. ---")
    return final_metrics


