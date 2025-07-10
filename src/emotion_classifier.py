# src/emotion_classifier.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset, Features, ClassLabel, Value
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionClassifier:
    def __init__(self, model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'; self.model = model.to(self.device); self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    def predict(self, text: str) -> str:
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = [self.model.config.id2label[i] for i in logits.argmax(dim=1).tolist()]
        return predictions[0] if len(predictions) == 1 else predictions

def train_and_evaluate_emotion_classifier(EMOTION_LABELS, label2id, EMOTION_MODEL_NAME, RANDOM_STATE):
    print("\n--- [PARTE I] Iniciando entrenamiento y evaluación del Clasificador de Emociones... ---")
    
    data_custom_list = [("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"),("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),("Me siento desmotivado con la búsqueda.", "Tristeza"), ("¡Conseguí la entrevista!", "Alegría"), ("La situación es frustrante.", "Ira"), ("Tengo fe en que el plan funcionará.", "Confianza"),("El procedimiento es claro.", "Neutral"), ("Tengo miedo de no estar a la altura.", "Miedo"),("¡Wow, no puedo creer que esto sea posible!", "Sorpresa"), ("No puedo esperar a ver los resultados.", "Anticipación"),("¡Qué buena noticia! Esto me da mucho ánimo.", "Alegría"), ("Recibí la documentación para el siguiente paso.", "Neutral"),("¡No puedo creer la incompetencia, es inaceptable!", "Ira"), ("He perdido toda esperanza en este proceso.", "Tristeza"),("Estoy seguro de que seguiremos el camino correcto.", "Confianza"), ("Me aterra pensar en las consecuencias si esto falla.", "Miedo"),("Francamente, el resultado me ha dejado sin palabras.", "Sorpresa"), ("Estoy expectante por los próximos pasos del plan.", "Anticipación"),("¡Siento un gran alivio y felicidad por esta noticia!", "Alegría"), ("Se ha procesado la solicitud según lo previsto.", "Neutral")]
    df_custom = pd.DataFrame(data_custom_list, columns=['text', 'emotion'])
    
    print(f"  › Dataset original: {len(df_custom)} ejemplos.")
    print("  › Inicializando modelos de traducción (puede tardar la primera vez)...")
    translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=0 if torch.cuda.is_available() else -1)
    translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=0 if torch.cuda.is_available() else -1)
    print("  › Modelos de traducción cargados. Iniciando aumento de datos...")
    
    augmented_rows = []
    for _, row in df_custom.iterrows():
        try:
            for _ in range(4):
                translated_text = translator_es_en(row['text'], max_length=128)[0]['translation_text']
                back_translated_text = translator_en_es(translated_text, max_length=128)[0]['translation_text']
                augmented_rows.append({'text': back_translated_text, 'emotion': row['emotion']})
        except Exception as e:
            print(f"    - Advertencia: No se pudo aumentar la frase: '{row['text']}'. Error: {e}")
            
    df_augmented = pd.concat([df_custom, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"  › Dataset aumentado: {len(df_augmented)} ejemplos.")
    df_augmented['label'] = df_augmented['emotion'].map(label2id)

    train_df, test_df = train_test_split(df_augmented, test_size=0.25, random_state=RANDOM_STATE, stratify=df_augmented['label'])
    
    emotion_features = Features({ 'text': Value('string'), 'emotion': Value('string'), 'label': ClassLabel(names=EMOTION_LABELS) })
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True), features=emotion_features)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True), features=emotion_features)
    
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME, num_labels=len(EMOTION_LABELS), id2label={i: l for i, l in enumerate(EMOTION_LABELS)}, label2id=label2id)
    def tokenize_function(examples): return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    print("\n  › Entrenando el clasificador de emociones con datos aumentados...")
    training_args = TrainingArguments(output_dir="./results_emotion_augmented", num_train_epochs=8, per_device_train_batch_size=8, learning_rate=3e-5, logging_strategy="steps", logging_steps=15, report_to="all")
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_test)
    trainer.train()
    
    print("\n--- 📊 Evaluación del Clasificador de Emociones Mejorado ---")
    emotion_classifier = EmotionClassifier(model, tokenizer)
    y_true_emotion = test_ds['emotion']
    y_pred_emotion = emotion_classifier.predict(test_ds['text'])
    
    print("  › Reporte de Clasificación (Clasificador de Emociones):")
    print(classification_report(y_true_emotion, y_pred_emotion, labels=EMOTION_LABELS, zero_division=0))
    
    cm_emotion = confusion_matrix(y_true_emotion, y_pred_emotion, labels=EMOTION_LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Matriz de Confusión - Clasificador de Emociones Mejorado')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.show()

    print("--- ✅ Clasificador de Emociones Entrenado y Evaluado. ---")
    return emotion_classifier
