
import pandas as pd
import torch
from torch import nn
import numpy as np
import os
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, Features, ClassLabel, Value

class EmotionClassifier:
    def __init__(self, model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
    def predict_proba(self, text: str) -> dict:
        if isinstance(text, str): text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        return {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}
    def predict(self, text: str) -> str:
        return max(self.predict_proba(text), key=self.predict_proba(text).get)

def get_custom_domain_data() -> pd.DataFrame:
    data_custom_list = [
        ("¡Esto nunca funciona como debería!", "Ira"), ("Mi hipótesis fue refutada", "Tristeza"),
        ("Siempre creí que este sistema funcionaría.", "Confianza"), ("No sé cómo voy a superar esto.", "Miedo"),
        ("¡Qué maravilla! No esperaba este resultado.", "Sorpresa"), ("Estoy listo para empezar, ¿cuál es el primer paso?", "Anticipación"),
        ("Me siento desmotivado con la búsqueda.", "Tristeza"), ("¡Conseguí la entrevista!", "Alegría"),
        ("La situación es frustrante.", "Ira"), ("Tengo fe en que el plan funcionará.", "Confianza"),
        ("El procedimiento es claro.", "Neutral"), ("Tengo miedo de no estar a la altura.", "Miedo"),
    ]
    return pd.DataFrame(data_custom_list, columns=['text', 'emotion'])

def download_and_prepare_dataset(emotion_labels: list) -> pd.DataFrame:
    print("  › Cargando dataset 'emotion' desde el Hub de Hugging Face...")
    try:
        # Cargar el dataset público estándar
        dataset = load_dataset("emotion", split='train')
        df_public = dataset.to_pandas()
        
        # Mapeo de etiquetas del dataset público (números) a las del proyecto (texto)
        label_map = {0: 'Tristeza', 1: 'Alegría', 2: 'Amor/Confianza', 3: 'Ira', 4: 'Miedo', 5: 'Sorpresa'}
        df_public['emotion'] = df_public['label'].map(label_map)
        
        # Mapeo secundario para alinear con tus etiquetas
        final_map = {'Amor/Confianza': 'Confianza'}
        df_public['emotion'] = df_public['emotion'].replace(final_map)
        
        df_public_clean = df_public.dropna(subset=['emotion'])[['text', 'emotion']]
        
        # Cargar y combinar con el dataset de dominio
        df_domain = get_custom_domain_data()
        df_combined = pd.concat([df_public_clean, df_domain], ignore_index=True)
        
        df_final = df_combined[df_combined['emotion'].isin(emotion_labels)].drop_duplicates(subset=['text']).reset_index(drop=True)
        
        print(f"  › Dataset combinado creado con {len(df_final)} ejemplos únicos.")
        return df_final
    except Exception as e:
        warnings.warn(f"No se pudo descargar el dataset público. Usando solo corpus de dominio. Error: {e}")
        return get_custom_domain_data()

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate_emotion_classifier(config: dict):
    cfg_emo = config['model_params']['emotion_classifier']
    EMOTION_LABELS = config['constants']['emotion_labels']
    RANDOM_STATE = config['model_params']['cognitive_tutor']['random_state']
    
    df_processed = download_and_prepare_dataset(EMOTION_LABELS)

    label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    df_processed['label'] = df_processed['emotion'].map(label2id)

    # Validación Cruzada
    n_splits = config['model_params']['emotion_classifier']['experimental_pipeline']['cv_folds']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    all_metrics = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(df_processed['text'], df_processed['emotion'])):
        print(f"\n  --- Fold {fold + 1}/{n_splits} ---")
        train_df = df_processed.iloc[train_index]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['emotion']), y=train_df['emotion'])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(df_processed.iloc[val_index])
        
        tokenizer = AutoTokenizer.from_pretrained(cfg_emo['model_name'])
        def tokenize(batch): return tokenizer(batch['text'], padding='max_length', truncation=True)
        
        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label=id2label, label2id=label2id)
        
        training_args = TrainingArguments(output_dir=f"./cv_results/fold_{fold}", **config['model_params']['emotion_classifier']['training_params'])
        
        trainer = WeightedLossTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, class_weights=class_weights_tensor)
        trainer.train()
        
        predictions = trainer.predict(val_ds)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = val_ds['label']
        
        fold_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        all_metrics.append(fold_f1)
        print(f"  › F1-Score (Macro) para el Fold {fold + 1}: {fold_f1:.3f}")

    print(f"\n--- 📊 Resultados Finales de CV: F1-Score Promedio: {np.mean(all_metrics):.3f} ± {np.std(all_metrics):.3f} ---")
    
    # Entrenamiento Final
    print("\n--- 🚂 Entrenando modelo final con TODOS los datos... ---")
    full_dataset = Dataset.from_pandas(df_processed).map(tokenize, batched=True)
    final_model = AutoModelForSequenceClassification.from_pretrained(cfg_emo['model_name'], num_labels=len(EMOTION_LABELS), id2label=id2label, label2id=label2id)
    final_args = TrainingArguments(output_dir="./results_emotion_final", **config['model_params']['emotion_classifier']['training_params'])
    final_trainer = Trainer(model=final_model, args=final_args, train_dataset=full_dataset)
    final_trainer.train()
    
    model_save_path = config['model_paths']['emotion_classifier']
    print(f"\n  › Guardando el modelo final de producción en: {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    final_trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
