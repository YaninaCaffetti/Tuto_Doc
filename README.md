# Tuto_Doc:Proyecto de Tesis - Tutor Cognitivo Adaptativo con IA Afectiva.

**Autora:** Mgter. Ing. Yanina A. Caffetti
**Institución:** Universidad Nacional de Misiones, Argentina.
**Programa:** Doctorado en Informática

---

## 1. Descripción del Proyecto

Este repositorio contiene el prototipo funcional y el pipeline experimental desarrollado como parte de la tesis doctoral "Diseño de un proceso para la integración de un tutor cognitivo adaptativo basado en arquetipos de usuario y computación afectiva".

El proyecto explora la sinergia entre un **módulo de razonamiento cognitivo**, que clasifica perfiles de usuario complejos, y un **módulo de percepción afectiva**, que detecta la emoción en el lenguaje del usuario. El objetivo principal es crear un sistema de tutoría que no solo ofrezca un plan de acción basado en el perfil del estudiante, sino que también **adapte su interacción y recomendaciones en tiempo real** al estado emocional detectado, generando una intervención más holística y empática.

La investigación sigue un riguroso proceso de Machine Learning Operations (MLOps), incluyendo la evaluación de modelos, benchmarking, tratamiento de desbalance de clases, validación estadística y, finalmente, la refactorización a una arquitectura de software modular.

## 2. Características Principales

* **🧠 Módulo de Razonamiento Cognitivo:** Utiliza un modelo `RandomForestClassifier` para clasificar perfiles de usuario (basados en datos de la encuesta ENDIS 2018) en arquetipos predefinidos. El rendimiento de este componente fue optimizado mediante la técnica **SMOTE** para manejar el severo desbalance de clases.
* **❤️ Módulo de Percepción Afectiva:** Emplea un modelo de lenguaje `BERT` (BETO) fine-tuned para clasificar el texto del usuario en una de 8 emociones básicas. Para superar la escasez de datos de dominio, se implementó una estrategia de **aumentación de datos por retrotraducción (back-translation)**.
* **✨ Sistema de Adaptación (MoE):** Una arquitectura de **Mezcla de Expertos (Mixture of Experts)** orquesta la respuesta final. El arquetipo predicho selecciona al "tutor experto" principal, mientras que la emoción detectada modula el plan de acción final.
* **🔬 Pipeline de Evaluación Riguroso:** El proyecto incluye un pipeline completo para el benchmarking comparativo de modelos y la validación de la significancia estadística de los resultados mediante el **Test de McNemar**.

## 3. Estructura del Proyecto

El código está organizado siguiendo las mejores prácticas para facilitar su mantenibilidad y comprensión.

¡Excelente idea! Un buen README.md es la carta de presentación de tu proyecto. Es fundamental para que otros (incluido tu tribunal de tesis y tu futuro "yo") puedan entender rápidamente el alcance, la metodología y los resultados de tu trabajo.

Aquí tienes una propuesta completa para tu archivo README.md. Está redactado en Markdown, así que puedes copiarlo y pegarlo directamente en un archivo con ese nombre en la raíz de tu proyecto.

Markdown

# Proyecto de Tesis: Tutor Cognitivo Adaptativo con IA Afectiva

**Autora:** Mgter. Ing. Yanina A. Caffetti
**Institución:** [Nombre de tu Universidad]
**Programa:** Doctorado en Informática

---

## 1. Descripción del Proyecto

Este repositorio contiene el prototipo funcional y el pipeline experimental desarrollado como parte de la tesis doctoral "Diseño de un proceso para la integración de un tutor cognitivo adaptativo basado en arquetipos de usuario y computación afectiva".

El proyecto explora la sinergia entre un **módulo de razonamiento cognitivo**, que clasifica perfiles de usuario complejos, y un **módulo de percepción afectiva**, que detecta la emoción en el lenguaje del usuario. El objetivo principal es crear un sistema de tutoría que no solo ofrezca un plan de acción basado en el perfil del estudiante, sino que también **adapte su interacción y recomendaciones en tiempo real** al estado emocional detectado, generando una intervención más holística y empática.

La investigación sigue un riguroso proceso de Machine Learning Operations (MLOps), incluyendo la evaluación de modelos, benchmarking, tratamiento de desbalance de clases, validación estadística y, finalmente, la refactorización a una arquitectura de software modular.

## 2. Características Principales

* **🧠 Módulo de Razonamiento Cognitivo:** Utiliza un modelo `RandomForestClassifier` para clasificar perfiles de usuario (basados en datos de la encuesta ENDIS 2018) en arquetipos predefinidos. El rendimiento de este componente fue optimizado mediante la técnica **SMOTE** para manejar el severo desbalance de clases.
* **❤️ Módulo de Percepción Afectiva:** Emplea un modelo de lenguaje `BERT` (BETO) fine-tuned para clasificar el texto del usuario en una de 8 emociones básicas. Para superar la escasez de datos de dominio, se implementó una estrategia de **aumentación de datos por retrotraducción (back-translation)**.
* **✨ Sistema de Adaptación (MoE):** Una arquitectura de **Mezcla de Expertos (Mixture of Experts)** orquesta la respuesta final. El arquetipo predicho selecciona al "tutor experto" principal, mientras que la emoción detectada modula el plan de acción final.
* **🔬 Pipeline de Evaluación Riguroso:** El proyecto incluye un pipeline completo para el benchmarking comparativo de modelos y la validación de la significancia estadística de los resultados mediante el **Test de McNemar**.

## 3. Estructura del Proyecto

El código está organizado siguiendo las mejores prácticas para facilitar su mantenibilidad y comprensión.

TESIS_TUTOR_COGNITIVO/
│
├── main.py             # Script principal que orquesta todo el pipeline.
│
└── src/                  # Carpeta para todo el código fuente.
│
├── init.py       # Hace que 'src' sea un paquete de Python.
│
├── data_processing.py # Funciones de ingeniería de características y fuzzificación.
│
├── emotion_classifier.py # Lógica de entrenamiento y evaluación del clasificador de emociones.
│
└── cognitive_tutor.py # Clases de los Expertos y el sistema MoESystem.


## 4. Metodología y Tecnologías

| Componente | Técnica / Modelo Utilizado |
| :--- | :--- |
| **Datos Cognitivos** | Encuesta Nacional de Discapacidad (ENDIS) 2018 |
| **Ingeniería de Feat.** | Creación de perfiles, arquetipos y fuzzificación |
| **Clasif. Cognitivo** | **RandomForestClassifier** (Modelo final), DecisionTree, IF-HUPM (Benchmarks) |
| **Balanceo de Clases** | **SMOTE** (Synthetic Minority Over-sampling Technique) |
| **Datos de Emoción** | Corpus propio aumentado con **Back-Translation** |
| **Clasif. de Emoción** | Fine-tuning de **BETO** (`dccuchile/bert-base-spanish-wwm-cased`) |
| **Validación Estadística**| **Test de McNemar** |

## 5. Instalación y Ejecución

#### **Requisitos Previos**
* Python 3.9+
* Una cuenta de Hugging Face ([huggingface.co](https://huggingface.co/)) para obtener un token de acceso.
* El dataset `base_estudio_discapacidad_2018.csv` ubicado en la ruta especificada en `main.py`. Sino puedes consultarlo de: https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos-7

#### **Instalación**
1.  Clona o descarga este repositorio.
2.  Crea un entorno virtual (recomendado).
3.  Instala las dependencias:

# --- Core de Machine Learning y Deep Learning ---
scikit-learn
torch
transformers[torch]
accelerate

# --- Manipulación y Procesamiento de Datos ---
pandas
numpy
imbalanced-learn

# --- Ecosistema Hugging Face ---
datasets
huggingface_hub

# --- Modelos de Traducción y Tokenización ---
sentencepiece
sacremoses

# --- Visualización y Estadísticas ---
seaborn
matplotlib
mlxtend

#### **Ejecución**
El script está diseñado para ser ejecutado en un entorno como Google Colab, donde puede acceder a GPUs y manejar las dependencias de manera sencilla.

1.  Asegúrate de que la estructura de archivos (`main.py` y la carpeta `src/`) esté en tu entorno.
2.  Ejecuta el script principal.
3.  La primera vez, te pedirá autenticarte en Hugging Face. Pega tu token de acceso cuando se te solicite.
4.  El pipeline completo se ejecutará, mostrando los resultados de cada fase.

## 6. Resumen de Hallazgos

1.  **Modelo Cognitivo:** Se validó que un `RandomForestClassifier` entrenado con datos balanceados por `SMOTE` es la solución óptima para la clasificación de arquetipos, alcanzando un **91% de accuracy** y un **F1-score macro de 0.79**, demostrando su capacidad para predecir clases minoritarias. La mejora sobre un `DecisionTree` simple fue estadísticamente significativa (p < 0.05).
2.  **Clasificador de Emociones:** La estrategia de aumentación de datos por retrotraducción fue **altamente efectiva**, llevando el rendimiento del clasificador a un **100% de precisión** en el conjunto de prueba del dominio.
3.  **Trade-off XAI vs. Rendimiento:** La investigación ha cuantificado empíricamente el compromiso entre rendimiento y explicabilidad. El modelo original `IF-HUPM`, aunque 100% interpretable ("caja blanca"), demostró ser frágil y de bajo rendimiento, mientras que el `RandomForest` ("caja negra") ofreció un rendimiento robusto y superior.

## 7. Trabajo Futuro

* **Adaptación Afectiva Difusa:** Implementar la lógica para que el `MoESystem` utilice el **vector completo de probabilidades** de emoción, en lugar de solo la emoción dominante, para una adaptación aún más matizada.
* **Explicabilidad del Modelo Final:** Aplicar técnicas de XAI post-hoc (como **SHAP** o **LIME**) sobre el `RandomForestClassifier` para intentar explicar sus predicciones y comparar estas explicaciones con las reglas del `IF-HUPM`.
* **Modelos Híbridos:** Explorar arquitecturas que combinen la interpretabilidad del `IF-HUPM` para casos de baja confianza con el rendimiento del `RandomForest` para predicciones de alta confianza.
