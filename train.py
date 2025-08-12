import yaml
import traceback
import argparse

# Importar las funciones de entrenamiento de sus respectivos módulos
from src.emotion_classifier import train_and_evaluate_emotion_classifier
from src.cognitive_model_trainer import train_cognitive_tutor

def main():
    """
    Orquestador principal para el entrenamiento de los modelos del sistema.

    Permite entrenar el clasificador de emociones, el tutor cognitivo, o ambos,
    a través de argumentos de línea de comandos.
    """
    # --- 1. Configurar Argumentos de Línea de Comandos ---
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento para el Tutor Cognitivo-Afectivo.")
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['emotion', 'cognitive', 'all'], 
        required=True,
        help="Especifica qué modelo entrenar: 'emotion', 'cognitive', o 'all' para ambos."
    )
    args = parser.parse_args()

    # --- 2. Cargar Configuración ---
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error Crítico: No se encontró el archivo 'config.yaml'.")
        return
    except Exception as e:
        print(f"❌ Error al cargar 'config.yaml': {e}")
        return

    # --- 3. Ejecutar el Pipeline Seleccionado ---
    try:
        if args.model == 'emotion' or args.model == 'all':
            train_and_evaluate_emotion_classifier(config)
        
        if args.model == 'cognitive' or args.model == 'all':
            train_cognitive_tutor(config)

        print("\n✅ --- Proceso de entrenamiento solicitado finalizado con éxito. ---")

    except Exception as e:
        print(f"\n❌ Un error inesperado ocurrió durante el entrenamiento: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
