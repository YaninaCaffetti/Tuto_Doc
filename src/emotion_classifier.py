# src/emotion_classifier.py (Corregido)

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

class EmotionClassifier:
    def __init__(self, model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict(self, text: str) -> str:
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = [self.model.config.id2label[i] for i in logits.argmax(dim=1).tolist()]
        return predictions[0] if len(predictions) == 1 else predictions
    
    def predict_proba(self, text: str) -> dict:
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}

def augment_emotion_data(df: pd.DataFrame, num_augments: int = 4) -> pd.DataFrame:
    print(f"\n  › Aumentando el dataset de emociones... (Esto puede tardar varios minutos)")
    print("  › Inicializando modelos de traducción (puede tardar la primera vez)...")
    translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=0 if torch.cuda.is_available() else -1)
    translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=0 if torch.cuda.is_available() else -1)
    print("  › Modelos de traducción cargados. Iniciando aumento de datos...")

    augmented_rows = []
    for _, row in df.iterrows():
        original_text = row['text']
        original_emotion = row['emotion']
        for _ in range(num_augments):
            try:
                translated_text = translator_es_en(original_text, max_length=128)[0]['translation_text']
                back_translated_text = translator_en_es(translated_text, max_length=128)[0]['translation_text']
                augmented_rows.append({'text': back_translated_text, 'emotion': original_emotion})
            except Exception as e:
                print(f"    - Advertencia: No se pudo aumentar la frase: '{original_text}'. Error: {e}")
                continue

    if not augmented_rows:
        return df

    df_augmented = pd.DataFrame(augmented_rows)
    return pd.concat([df, df_augmented], ignore_index=True)

def train_and_evaluate_emotion_classifier(config):
    print("\n--- [PARTE I] Iniciando entrenamiento y evaluación del Clasificador de Emociones... ---")
    
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    
    data_custom_list = [("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"),("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),("Me siento desmotivado con la búsqueda.", "Tristeza"), ("¡Conseguí la entrevista!", "Alegría"), ("La situación es frustrante.", "Ira"), ("Tengo fe en que el plan funcionará.", "Confianza"),("El procedimiento es claro.", "Neutral"), ("Tengo miedo de no estar a la altura.", "Miedo"),("¡Wow, no puedo creer que esto sea posible!", "Sorpresa"), ("No puedo esperar a ver los resultados.", "Anticipación"),("¡Qué buena noticia! Esto me da mucho ánimo.", "Alegría"), ("Recibí la documentación para el siguiente paso.", "Neutral"),("¡No puedo creer la incompetencia, es inaceptable!", "Ira"), ("He perdido toda esperanza en este proceso.", "Tristeza"),("Estoy seguro de que seguiremos el camino correcto.", "Confianza"), ("Me aterra pensar en las consecuencias si esto falla.", "Miedo"),("Francamente, el resultado me ha dejado sin palabras.", "Sorpresa"), ("Estoy expectante por los próximos pasos del plan.", "Anticipación"),("¡Siento un gran alivio y felicidad por esta noticia!", "Alegría"), ("Se ha procesado la solicitud según lo previsto.", "Neutral")]
    df_custom = pd.DataFrame(data_custom_list, columns=['text', 'emotion'])
    
    df_augmented = augment_emotion_data(df_custom, num_augments=4)
    print(f"  › Tamaño del dataset de emociones original: {len(df_custom)}. Tamaño aumentado: {len(df_augmented)}")
    df_augmented['label'] = df_augmented['emotion'].map(label2id)

    train_df, test_df = train_test_split(df_augmented, test_size=0.25, random_state=RANDOM_STATE, stratify=df_augmented['label'])
    
    emotion_features = Features({ 'text': Value('string'), 'emotion': Value('string'), 'label': ClassLabel(names=EMOTION_LABELS) })
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True), features=emotion_features)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True), features=emotion_features)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label={i: l for i, l in enumerate(EMOTION_LABELS)}, label2id=label2id)
    def tokenize_function(examples): return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    print("\n  › Entrenando el clasificador de emociones con datos aumentados...")
    
    learning_rate = float(cfg_emo['learning_rate'])
    
    training_args = TrainingArguments(
        output_dir="./results_emotion_augmented", 
        num_train_epochs=cfg_emo['epochs'], 
        per_device_train_batch_size=cfg_emo['train_batch_size'], 
        learning_rate=learning_rate, 
        logging_strategy="steps", 
        logging_steps=cfg_emo['logging_steps'],
        report_to="all",
        evaluation_strategy="epoch" # Para monitorear el rendimiento en validación
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_test)
    trainer.train()
    
    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo de emociones en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("  › Modelo guardado exitosamente.")

    print("\n--- 📊 Evaluación del Clasificador de Emociones Mejorado ---")
    emotion_classifier = EmotionClassifier(model, tokenizer)
    y_true_emotion = test_ds['emotion']
    
    # Optimización de la predicción
    predictions_probs = [emotion_classifier.predict_proba(text) for text in test_ds['text']]
    y_pred_emotion = [max(prob_dict, key=prob_dict.get) for prob_dict in predictions_probs]
    
    print("  › Reporte de Clasificación (BERT fine-tuned):")
    print(classification_report(y_true_emotion, y_pred_emotion, labels=EMOTION_LABELS, zero_division=0))
    
    cm_emotion = confusion_matrix(y_true_emotion, y_pred_emotion, labels=EMOTION_LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Matriz de Confusión - Clasificador de Emociones Mejorado')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.show()

    print("\n--- 📊 Benchmark del Clasificador de Emociones vs. Modelo Clásico ---")
    classic_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    print("\n  › Entrenando modelo clásico (TF-IDF + Regresión Logística)...")
    classic_model.fit(train_df['text'], train_df['emotion'])
    print("  › Evaluación del modelo clásico:")
    y_pred_classic = classic_model.predict(test_df['text'])
    print("  › Reporte de Clasificación (Modelo Clásico):")
    print(classification_report(test_df['emotion'], y_pred_classic, labels=EMOTION_LABELS, zero_division=0))

    print("\n--- Resumen de Comparación de Modelos de Emoción ---")
    f1_bert = f1_score(y_true_emotion, y_pred_emotion, average='macro')
    f1_classic = f1_score(test_df['emotion'], y_pred_classic, average='macro')
    print(f"  - F1-Score Macro (BERT Aumentado): {f1_bert:.2f}")
    print(f"  - F1-Score Macro (Clásico TF-IDF + LogReg): {f1_classic:.2f}")

    if f1_bert > f1_classic:
        print("\n  Conclusión: El modelo BERT supera significativamente al benchmark clásico.")
    else:
        print("\n  Conclusión: El modelo BERT NO supera al benchmark clásico, se requiere más análisis.")

    print("--- ✅ Clasificador de Emociones Entrenado y Evaluado. ---")
    return emotion_classifier
