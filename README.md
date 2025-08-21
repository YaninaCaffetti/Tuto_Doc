# Tuto_Doc: Tutor Cognitivo Adaptativo con IA Afectiva
Prototipo de Tesis Doctoral - Mgter. Ing. Yanina A. Caffetti
Institución: Universidad Nacional de Misiones, Argentina.
Programa: Doctorado en Informática

---

## 1. Descripción del Proyecto
Este repositorio contiene el prototipo funcional y el pipeline experimental desarrollado para la tesis doctoral "Proceso para la integración de un tutor cognitivo de consultas referidas a la legislación argentina sobre discapacidad. Estudio de caso: Agente inteligente de connotaciones emocionales definidas por el usuario".

El proyecto implementa un agente inteligente que funciona como un tutor cognitivo. Su principal innovación es la capacidad de adaptar sus respuestas y planes de acción basándose en el estado afectivo del usuario, el cual es inferido en tiempo real a partir del texto de sus consultas. El sistema utiliza una arquitectura Mixture of Experts (MoE), donde un componente cognitivo es modulado por un componente afectivo para generar una interacción más empática y efectiva.

La investigación sigue un riguroso proceso de Machine Learning, incluyendo el preprocesamiento de datos, la evaluación robusta de modelos mediante validación cruzada, y la refactorización a una arquitectura de software modular y probada unitariamente.

Palabras Clave: Tutor Cognitivo, Computación Afectiva, Mezcla de Expertos (MoE), MLOps, Procesamiento del Lenguaje Natural (PLN), XAI, Validación Cruzada.


## 2. Arquitectura del Sistema
El sistema está diseñado de forma modular para facilitar su mantenimiento, prueba y escalabilidad.

Pipeline de Procesamiento de Datos (src/data_processing.py): Un script de ETL que toma el dataset crudo, aplica una serie de transformaciones basadas en reglas de negocio expertas (ingeniería de características, creación de arquetipos) y lo convierte en un conjunto de datos "fuzzificado" listo para el entrenamiento.

Clasificador de Emociones (src/emotion_classifier.py): Un modelo de lenguaje (dccuchile/bert-base-spanish-wwm-uncased) fine-tuneado para clasificar texto en 7 emociones. Este es el componente afectivo del sistema.

Tutor Cognitivo (src/cognitive_model_trainer.py): Un modelo RandomForestClassifier que asigna a cada usuario un arquetipo cognitivo. Su rendimiento se valida rigurosamente mediante validación cruzada estratificada.

Aplicación Interactiva (app.py): Una interfaz de usuario desarrollada con Streamlit que integra todos los componentes y permite la interacción en tiempo real con el tutor.


## 3. Características Principales

✨ Adaptación Afectiva (MoE): El arquetipo predicho selecciona al "tutor experto" principal, pero el vector de probabilidades de emoción modula los pesos de todos los expertos, generando un plan de acción mixto y verdaderamente adaptativo.

🔬 Evaluación Robusta: El rendimiento del modelo cognitivo se valida con Validación Cruzada Estratificada (K-Fold) para obtener una métrica fiable y académicamente defendible.

🔧 Código Refactorizado y Probado: La lógica de negocio ha sido refactorizada para máxima legibilidad y validada con una suite de pruebas unitarias (pytest), garantizando su robustez.

📊 Seguimiento de Experimentos: Integración con MLflow para registrar parámetros, métricas y artefactos de cada ejecución de entrenamiento.


## 4. Guía de Instalación y Uso

Siga estos pasos para configurar y ejecutar el proyecto en un entorno como Google Colab.

### Paso 1: Clonar el Repositorio

```
git clone https://github.com/YaninaCaffetti/Tuto_Doc.git
cd Tuto_Doc
```

### Paso 2: Configuración del Dataset Crudo (Vía Google Drive)
Este proyecto está diseñado para leer el dataset crudo (`base_estudio_discapacidad_2018.csv`) desde Google Drive para evitar problemas con los límites de Git LFS.

Suba el archivo `base_estudio_discapacidad_2018.csv` a su Google Drive.

En un entorno de Colab, monte su Drive:

```
from google.colab import drive
drive.mount('/content/drive')
```

Actualice la configuración: Abra el archivo `config.yaml` y modifique la clave raw_data para que apunte a la ruta de su archivo en Drive.

### Ejemplo de configuración en config.yaml

```
data_paths:
  raw_data: '/content/drive/MyDrive/ruta/a/su/base_estudio_discapacidad_2018.csv'
```

### Paso 3: Instalar Dependencias

```
pip install -r requirements.txt
```

### Paso 4: Ejecutar el Pipeline de Procesamiento de Datos

Este comando es obligatorio y debe ejecutarse primero. Tomará el dataset crudo de su Drive, lo procesará y generará el archivo `data/cognitive_profiles.csv` necesario para el entrenamiento.

```
python src/data_processing.py
```

### Paso 5: Entrenar los Modelos

Utilice el script `train.py` para entrenar los componentes.

Entrenar solo el Tutor Cognitivo (rápido, recomendado para pruebas):

```
python train.py --model cognitive
```

Entrenar ambos modelos (ejecución completa):

```
python train.py --model all
```

### Paso 6: Lanzar la Aplicación

Una vez que los modelos han sido entrenados (y la carpeta `saved_models` ha sido creada), puede lanzar la interfaz interactiva.

```
streamlit run app.py
```

## 5. Resumen de Hallazgos Claves

Modelo Cognitivo (Validación Cruzada): La evaluación robusta mediante K-Fold Estratificado arrojó un F1-Score Macro promedio de 0.811 ± 0.029. Este resultado realista y estable reemplaza métricas iniciales que sugerían sobreajuste, demostrando una sólida y fiable capacidad de generalización del modelo RandomForestClassifier.

Clasificador de Emociones: El modelo fine-tuneado alcanzó un F1-Score Macro de 0.759. Se identificó una debilidad específica en la clase "Anticipación" (F1-Score de 0.00), atribuida a una representación casi nula en los datos de entrenamiento, lo que constituye un hallazgo importante sobre el impacto del desbalance de clases. "Anticipación" solamente se encuentra en el corpus propio. 

Trade-off XAI vs. Rendimiento: Se demostró un claro compromiso entre la interpretabilidad y el rendimiento predictivo. Mientras que un modelo de "caja blanca" (IF-HUPM) resultó frágil en las fases iniciales, el enfoque de "caja negra" (RandomForestClassifier) proporcionó una solución robusta y superior, justificando empíricamente la necesidad de aplicar técnicas de XAI (Explainable AI) post-hoc.

## 6. Pila Tecnológica (Stack)

Lenguaje: Python 3.10+

Análisis de Datos: Pandas, NumPy

Machine Learning: Scikit-learn, PyTorch

NLP: Transformers (Hugging Face)

Interfaz de Usuario: Streamlit

Seguimiento de Experimentos: MLflow

Pruebas: Pytest

## 7. Trabajo Futuro

Explicabilidad del Modelo Final (XAI): Aplicar técnicas de XAI post-hoc (como SHAP o LIME) sobre el RandomForestClassifier para interpretar las características más influyentes en la predicción de cada arquetipo.

Validación con Usuarios: Realizar un estudio con familas y/o tutores de personas con discapacidad para medir cuantitativamente el impacto de la adaptación afectiva en la percepción de empatía y la confianza en el sistema.

Aprendizaje de Reglas Afectivas: Explorar métodos de aprendizaje por refuerzo para que el sistema aprenda las `affective_rules` desde datos de interacción, en lugar de definirlas heurísticamente.

## 8. Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo LICENSE para más detalles.
