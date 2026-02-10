# Tuto_Doc: Tutor Cognitivo Adaptativo con IA Afectiva

**Prototipo de Tesis Doctoral** **Autora:** Mgter. Ing. Yanina A. Caffetti  
**Institución:** Universidad Nacional de Misiones (UNaM), Argentina  
**Programa:** Doctorado en Informática

---

## 1. Descripción del Proyecto
Este repositorio contiene el prototipo funcional y el pipeline experimental de la investigación doctoral: **"Proceso para la integración de un tutor cognitivo de consultas referidas a la legislación argentina sobre discapacidad. Estudio de caso: Agente inteligente de connotaciones emocionales definidas por el usuario"**.

El sistema propone un modelo de **Computación Afectiva** que adapta planes de acción legales basándose en el estado emocional del usuario, inferido mediante Procesamiento de Lenguaje Natural (PLN). Utiliza una arquitectura **Mixture of Experts (MoE)** donde la respuesta técnica es modulada por un componente afectivo para garantizar una interacción empática y precisa.

**Palabras Clave:** Tutor Cognitivo, IA Afectiva, MoE, MLOps, PLN, XAI, Validación Cruzada, Lógica Difusa.

## 2. Arquitectura del Sistema
El sistema opera bajo una arquitectura de **Mezcla de Expertos** para integrar respuestas técnicas y matices emocionales de manera dinámica.

* **Pipeline de Datos (`src/data_processing.py`):** ETL avanzado que implementa la **"fuzzificación"** de variables del dataset del INDEC mediante reglas de negocio expertas y árboles de decisión difusos.
* **Componente Afectivo (`src/emotion_classifier.py`):** Modelo basado en `BERT-Spanish` (BETO) fine-tuneado para clasificar 7 estados emocionales en tiempo real.
* **Componente Cognitivo (`src/cognitive_model_trainer.py`):** Clasificador `RandomForest` validado mediante **K-Fold Estratificado**, encargado de asignar arquetipos legales específicos.
* **Interfaz Streamlit (`app.py`):** Aplicación interactiva que permite al usuario definir las connotaciones emocionales deseadas y recibir planes de acción personalizados.

## 3. Resumen de Hallazgos Claves (Resultados Reales)
* **Robustez del Modelo Cognitivo:** La evaluación mediante K-Fold Estratificado arrojó un **F1-Score Macro promedio de 0.655 ± 0.029**. Esta métrica es científicamente estable y valida el proceso de fuzzificación contra el sobreajuste (overfitting).
* **Rendimiento Afectivo:** El modelo BETO alcanzó un **F1-Score Macro de 0.858**, destacando una mejora significativa en la clase "Anticipación" (F1: 0.75) lograda mediante el ajuste de pesos de clase.
* **Validación Semántica:** Se obtuvo un **100% de precisión** en la detección de intenciones semánticas (Semantic Intent Accuracy), garantizando una orquestación precisa entre los expertos de la arquitectura MoE.
* **Trade-off XAI vs. Rendimiento:** Se justificó empíricamente el uso de modelos de "caja negra" (RandomForest) sobre modelos de "caja blanca" (IF-HUPM) mediante la aplicación de técnicas de **IA Explicable (XAI)** post-hoc.

## 4. Guía de Instalación y Uso

### Paso 1: Configuración del Dataset (Vía Google Drive)
Suba el archivo `base_estudio_discapacidad_2018.csv` (descargado del INDEC) a su Drive. En el archivo **config.yaml**, actualice la ruta correspondiente:

```yaml
data_paths:
  raw_data: '/content/drive/MyDrive/ruta/a/su/archivo.csv'

# 1. Clonar e instalar dependencias
git clone [https://github.com/YaninaCaffetti/Tuto_Doc.git](https://github.com/YaninaCaffetti/Tuto_Doc.git)
cd Tuto_Doc
pip install -r requirements.txt

# 2. Preprocesamiento y Fuzzificación (Obligatorio)
python src/data_processing.py

# 3. Entrenamiento con tracking en MLflow
python train.py --model all

# 4. Lanzar Interfaz Interactiva
streamlit run app.py
```

## 5. Pila Tecnológica (Stack)

* **Lenguaje**: Python 3.10+
* **NLP**: Hugging Face Transformers (BETO).
* **ML & Ops**: Scikit-learn, PyTorch, **MLflow** para el seguimiento riguroso de experimentos.
* **UI**: **Streamlit** para el prototipado rápido de la interfaz de usuario.
* **Pruebas**: Pytest para la suite de pruebas unitarias.

## 6. Cómo Contribuir

Este proyecto se encuentra en una etapa de consolidación académica como parte de una tesis doctoral. Si desea contribuir, siga estos pasos:

1. **Reporte de Errores**: Abra un *Issue* detallando el problema y los pasos para reproducirlo.
2. **Mejoras en el Modelo**: Se reciben propuestas para optimizar las `affective_rules` o la arquitectura de la red de compuertas (Gating Network).
3. **Pull Requests**: Antes de enviar un PR, asegúrese de que todas las pruebas de `pytest` pasen correctamente y que la documentación esté actualizada.

## 7. Trabajo Futuro

* **Validación Externa**: Realizar pruebas de campo con familias y tutores de personas con discapacidad para medir la percepción de empatía y confianza en el sistema.
* **Aprendizaje por Refuerzo**: Implementar un lazo de retroalimentación para que el sistema aprenda reglas afectivas dinámicas basadas en el *feedback* directo del usuario.

## 8. Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Consulte el archivo `LICENSE` para más detalles.
