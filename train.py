
import yaml, traceback
from src.emotion_classifier import train_and_evaluate_emotion_classifier

if __name__ == '__main__':
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        train_and_evaluate_emotion_classifier(config)
        print("\n✅ --- Proceso de entrenamiento de emociones finalizado. ---")
    except Exception as e:
        print(f"❌ Un error inesperado ocurrió: {e}")
        traceback.print_exc()
