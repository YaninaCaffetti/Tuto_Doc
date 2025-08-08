
import pandas as pd
import torch
from torch import nn
import numpy as np
import os
import warnings
from typing import Union, List, Dict

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
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Clasificador de Emociones inicializado en: {self.device.upper()}")

    ### ¡MEJORA! ###
    # Se ha refactorizado `predict_proba` para que maneje correctamente tanto
    # strings individuales como lotes (batches) de strings. Esto hace la clase más
    # robusta y versátil para diferentes casos de uso.
    def predict_proba(self, text: Union[str, List[str]]) -> List[Dict[str, float]]:
        """
        Calcula la distribución de probabilidad de todas las emociones para un texto o una lista de textos.
        
        Args:
            text (Union[str, List[str]]): El texto o lista de textos a clasificar.

        Returns:
            List[Dict[str, float]]: Una lista de diccionarios. Cada diccionario mapea
                                    cada etiqueta de emoción a su probabilidad.
        """
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        results = []
        for prob_tensor in probabilities:
            prob_dict = {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(prob_tensor)}
            results.append(prob_dict)
            
        return results

    def predict(self, text: str) -> str:
        """
        Predice la etiqueta de la emoción dominante en un único texto.
        """
        # Se toma el primer (y único) resultado de la lista devuelta por predict_proba
        probs = self.predict_proba(text)[0]
        return max(probs, key=probs.get)

# --- 2. LÓGICA DE PREPARACIÓN DE DATOS (Sin cambios, es correcta) ---
def get_custom_domain_data() -> pd.DataFrame:
    # ... (código original sin cambios)
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])


def download_and_prepare_dataset(emotion_labels: list) -> pd.DataFrame:
    # ... (código original sin cambios)
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
    # ... (código original sin cambios, es correcto)
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate_emotion_classifier(config: dict):
    """
    Orquesta el pipeline experimental completo para el clasificador de emociones.
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
    
    ### ¡CORRECCIÓN CRÍTICA! ###
    # El error original estaba aquí. `np.unique` ordena las etiquetas alfabéticamente,
    # lo que no necesariamente coincide con el orden de `label2id` definido por el config.
    # Esto causaba que los pesos de las clases se asignaran a las etiquetas incorrectas
    # durante el entrenamiento, perjudicando gravemente el rendimiento del modelo.
    # La corrección consiste en pasar explícitamente `EMOTION_LABELS` (que tiene el
    # orden correcto) al parámetro `classes` para garantizar la correspondencia.
    class_weights = compute_class_weight(
        'balanced',
        classes=EMOTION_LABELS, # Forzar el orden correcto de las clases
        y=train_df['emotion']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    training_params = cfg_emo['training_params']
    training_params['learning_rate'] = float(training_params['learning_rate'])
    
    # Esta sección es correcta para versiones recientes de Transformers.
    training_args = TrainingArguments(
        output_dir="./results_emotion_training",
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
    
    final_f1_score = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"  › **F1-Score (Macro) Final en el conjunto de prueba:** {final_f1_score:.3f}")

    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo final de producción en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print("\n--- ✅ Pipeline Experimental del Clasificador de Emociones Finalizado. ---")
