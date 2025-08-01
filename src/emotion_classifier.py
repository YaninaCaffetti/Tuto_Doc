# src/emotion_classifier.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset, Features, ClassLabel, Value
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os
import warnings

# --- 1. CLASE DE CLASIFICACIÓN (Para Inferencia) ---

class EmotionClassifier:
    """
    Un clasificador para inferir emociones de un texto utilizando un modelo
    pre-entrenado de Hugging Face.

    Esta clase encapsula el modelo y el tokenizador para facilitar la predicción
    y el cálculo de probabilidades.
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

    def predict(self, text: str) -> str:
        """
        Predice la emoción dominante en un texto.

        Args:
            text (str): El texto de entrada a clasificar.

        Returns:
            str: La etiqueta de la emoción con la mayor probabilidad.
        """
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = [self.model.config.id2label[i] for i in logits.argmax(dim=1).tolist()]
        return predictions[0] if len(predictions) == 1 else predictions
    
    def predict_proba(self, text: str) -> dict:
        """
        Calcula la distribución de probabilidad de todas las emociones para un texto.

        Args:
            text (str): El texto de entrada a clasificar.

        Returns:
            dict: Un diccionario que mapea cada etiqueta de emoción a su probabilidad.
        """
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}

# --- 2. LÓGICA DE PREPARACIÓN DE DATOS Y ENTRENAMIENTO ---

def get_base_emotion_data() -> pd.DataFrame:
    """
    Carga el corpus base de emociones desde una lista hardcodeada.

    En una implementación más avanzada, esto leería los datos desde un archivo
    externo (ej. un CSV) para separar completamente los datos del código.

    Returns:
        pd.DataFrame: Un DataFrame con las columnas 'text' y 'emotion'.
    """
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
        ("Me siento desmotivado con la búsqueda.", "Tristeza"), ("¡Conseguí la entrevista!", "Alegría"),
        ("La situación es frustrante.", "Ira"), ("Tengo fe en que el plan funcionará.", "Confianza"),
        ("El procedimiento es claro.", "Neutral"), ("Tengo miedo de no estar a la altura.", "Miedo"),
        ("¡Wow, no puedo creer que esto sea posible!", "Sorpresa"), ("No puedo esperar a ver los resultados.", "Anticipación"),
        ("¡Qué buena noticia! Esto me da mucho ánimo.", "Alegría"), ("Recibí la documentación para el siguiente paso.", "Neutral"),
        ("¡No puedo creer la incompetencia, es inaceptable!", "Ira"), ("He perdido toda esperanza en este proceso.", "Tristeza"),
        ("Estoy seguro de que seguiremos el camino correcto.", "Confianza"), ("Me aterra pensar en las consecuencias si esto falla.", "Miedo"),
        ("Francamente, el resultado me ha dejado sin palabras.", "Sorpresa"), ("Estoy expectante por los próximos pasos del plan.", "Anticipación"),
        ("¡Siento un gran alivio y felicidad por esta noticia!", "Alegría"), ("Se ha procesado la solicitud según lo previsto.", "Neutral")
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def augment_emotion_data(df: pd.DataFrame, num_augments: int = 4) -> pd.DataFrame:
    """
    Aplica aumentación de datos por retrotraducción (Español -> Inglés -> Español).

    Este es un proceso computacionalmente costoso que descarga modelos de traducción
    y genera nuevas muestras de texto para enriquecer el dataset de entrenamiento.

    Args:
        df (pd.DataFrame): El DataFrame original con columnas 'text' y 'emotion'.
        num_augments (int, optional): El número de versiones aumentadas a generar por cada texto original. Por defecto es 4.

    Returns:
        pd.DataFrame: Un DataFrame que contiene tanto los datos originales como los aumentados.
    """
    print(f"\n  › Aumentando el dataset de emociones... (Esto puede tardar varios minutos)")
    try:
        translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=0 if torch.cuda.is_available() else -1)
        translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        warnings.warn(f"No se pudieron cargar los modelos de traducción. Saltando aumentación. Error: {e}")
        return df

    augmented_rows = []
    for _, row in df.iterrows():
        original_text = row['text']
        for _ in range(num_augments):
            try:
                translated_text = translator_es_en(original_text, max_length=128)[0]['translation_text']
                back_translated_text = translator_en_es(translated_text, max_length=128)[0]['translation_text']
                augmented_rows.append({'text': back_translated_text, 'emotion': row['emotion']})
            except Exception as e:
                warnings.warn(f"No se pudo aumentar la frase: '{original_text}'. Error: {e}")
                continue
    
    if not augmented_rows:
        return df

    df_augmented = pd.DataFrame(augmented_rows)
    return pd.concat([df, df_augmented], ignore_index=True)

def train_and_evaluate_emotion_classifier(config: dict, use_augmentation: bool = True):
    """
    Orquesta el pipeline completo para el clasificador de emociones.

    Este proceso incluye:
    1. Carga y (opcionalmente) aumentación de datos.
    2. Fine-tuning de un modelo BERT pre-entrenado.
    3. Guardado del modelo y tokenizador entrenados.
    4. Evaluación del modelo y comparación con un benchmark clásico.

    Args:
        config (dict): El diccionario de configuración cargado desde config.yaml.
        use_augmentation (bool, optional): Si es True, se aplicará la aumentación de datos. Por defecto es True.
    """
    print("\n--- [PARTE I] Iniciando entrenamiento y evaluación del Clasificador de Emociones... ---")
    
    # --- A. Preparación de Parámetros y Datos ---
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    
    df_base = get_base_emotion_data()
    
    if use_augmentation:
        df_processed = augment_emotion_data(df_base, num_augments=cfg_emo.get('num_augments', 4))
    else:
        df_processed = df_base
        print("\n  › Saltando la aumentación de datos por configuración.")

    print(f"  › Tamaño del dataset de emociones final: {len(df_processed)} ejemplos.")
    
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    df_processed['label'] = df_processed['emotion'].map(label2id)

    train_df, test_df = train_test_split(df_processed, test_size=0.25, random_state=RANDOM_STATE, stratify=df_processed['label'])
    
    emotion_features = Features({'text': Value('string'), 'emotion': Value('string'), 'label': ClassLabel(names=EMOTION_LABELS)})
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True), features=emotion_features)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True), features=emotion_features)
    
    # --- B. Entrenamiento del Modelo BERT ---
    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label={i: l for i, l in enumerate(EMOTION_LABELS)}, label2id=label2id)
    
    def tokenize_function(examples): return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    print("\n  › Entrenando el clasificador de emociones...")
    
    training_args = TrainingArguments(
        output_dir="./results_emotion", 
        num_train_epochs=cfg_emo['epochs'], 
        per_device_train_batch_size=cfg_emo['train_batch_size'], 
        learning_rate=float(cfg_emo['learning_rate']),
        logging_strategy="epoch",
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_test)
    trainer.train()
    
    # --- C. Guardado del Modelo ---
    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo de emociones en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("  › Modelo guardado exitosamente.")

    # --- D. Evaluación y Benchmark ---
    print("\n--- 📊 Evaluación Comparativa del Clasificador de Emociones ---")
    
    emotion_classifier = EmotionClassifier(model, tokenizer)
    y_true_emotion = test_ds['emotion']
    y_pred_emotion = [emotion_classifier.predict(text) for text in test_ds['text']]
    
    print("\n  › Reporte de Clasificación (BERT fine-tuned):")
    print(classification_report(y_true_emotion, y_pred_emotion, labels=EMOTION_LABELS, zero_division=0))
    
    cm_emotion = confusion_matrix(y_true_emotion, y_pred_emotion, labels=EMOTION_LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Matriz de Confusión - Clasificador de Emociones (BERT)')
    plt.xlabel('Predicción'); plt.ylabel('Etiqueta Real')
    plt.savefig("confusion_matrix_emotion_bert.png")
    print("  › Matriz de confusión guardada en 'confusion_matrix_emotion_bert.png'")
    plt.close()

    classic_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    classic_model.fit(train_df['text'], train_df['emotion'])
    y_pred_classic = classic_model.predict(test_df['text'])
    
    print("\n  › Reporte de Clasificación (Benchmark Clásico: TF-IDF + LogReg):")
    print(classification_report(test_df['emotion'], y_pred_classic, labels=EMOTION_LABELS, zero_division=0))

    # --- E. Conclusión Final ---
    f1_bert = f1_score(y_true_emotion, y_pred_emotion, average='macro')
    f1_classic = f1_score(test_df['emotion'], y_pred_classic, average='macro')
    
    print("\n--- Resumen de Comparación (F1-Score Macro) ---")
    print(f"  - Modelo BERT fine-tuned: {f1_bert:.3f}")
    print(f"  - Modelo Clásico (TF-IDF): {f1_classic:.3f}")

    if f1_bert > f1_classic:
        print("\n  › Conclusión: El modelo BERT demuestra un rendimiento superior al benchmark clásico.")
    else:
        print("\n  › Conclusión: El modelo BERT no supera al benchmark clásico. Se recomienda revisar los datos o el modelo.")

    print("--- ✅ Clasificador de Emociones Entrenado y Evaluado. ---")
