# Tuto_Doc:Proyecto de Tesis - Tutor Cognitivo Adaptativo con IA Afectiva.

**Autora:** Mgter. Ing. Yanina A. Caffetti
**Institución:** Universidad Nacional de Misiones, Argentina.
**Programa:** Doctorado en Informática.

---

## 1. Descripción del Proyecto

Este repositorio contiene el prototipo funcional y el pipeline experimental desarrollado como parte de la tesis doctoral "Diseño de un proceso para la integración de un tutor cognitivo adaptativo basado en arquetipos de usuario y computación afectiva".

El proyecto explora la sinergia entre un **módulo de razonamiento cognitivo**, que clasifica perfiles de usuario complejos, y un **módulo de percepción afectiva**, que detecta la emoción en el lenguaje del usuario. El objetivo principal es crear un sistema de tutoría que no solo ofrezca un plan de acción basado en el perfil del estudiante, sino que también **adapte su interacción y recomendaciones en tiempo real** al estado emocional detectado, generando una intervención más holística y empática.

La investigación sigue un riguroso proceso de Machine Learning Operations (MLOps), incluyendo la evaluación de modelos, benchmarking, tratamiento de desbalance de clases, validación estadística y, finalmente, la refactorización a una arquitectura de software modular.

## 2. Características Principales

* **🧠 Módulo de Razonamiento Cognitivo:** Utiliza un modelo `RandomForestClassifier` para clasificar perfiles de usuario (basados en datos de la encuesta ENDIS 2018) en arquetipos predefinidos heurísticamente y basados en el modelo MTBI (o Indicador de Tipo Myers-Briggs, es una herramienta psicométrica que clasifica a las personas en 16 tipos de personalidad basados en sus preferencias en cuatro dicotomías psicológicas: extroversión/introversión, sensación/intuición, pensamiento/sentimiento y juicio/percepción. Este modelo, desarrollado por Katharine Briggs y Isabel Myers, se basa en la teoría de los tipos psicológicos de Carl Jung. 
). El rendimiento de este componente fue optimizado mediante la técnica **SMOTE** para manejar el severo desbalance de clases.
* **❤️ Módulo de Percepción Afectiva:** Emplea un modelo de lenguaje `BERT` (BETO) fine-tuned para clasificar el texto del usuario en una de 8 emociones básicas. Para superar la escasez de datos de dominio, se implementó una estrategia de **aumentación de datos por retrotraducción (back-translation)**.
* **✨ Sistema de Adaptación (MoE):** Una arquitectura de Mezcla de Expertos (Mixture of Experts) orquesta la respuesta final. El arquetipo predicho por el módulo cognitivo selecciona al "tutor experto" principal, pero la clave de la innovación reside en que el vector completo de probabilidades de emoción modula los pesos de todos los expertos. Esto permite, por ejemplo, que una alta probabilidad de "tristeza" aumente la prioridad del "Tutor de Bienestar", generando un plan de acción mixto y verdaderamente adaptativo.
* **🔬 Pipeline de Evaluación Riguroso:** El proyecto incluye un pipeline completo para el benchmarking comparativo de modelos y la validación de la significancia estadística de los resultados mediante el **Test de McNemar**.

## 3. Estructura del Proyecto

El código está organizado siguiendo las mejores prácticas para facilitar su mantenibilidad y comprensión.

Tuto_Doc/
│
├── app.py                # Aplicación de demostración interactiva con Streamlit.
├── train.py              # Script principal para entrenar y guardar todos los modelos.
├── config.yaml           # Archivo central de configuración de modelos y parámetros.
├── requirements.txt      # Dependencias del proyecto.
│
├── data/                 # Carpeta para los datasets (e.g., endis_raw).
│
└── src/                  # Carpeta para todo el código fuente modular.
    │
    ├── __init__.py
    ├── data_processing.py    # Pipeline de ingeniería de características y fuzzificación.
    ├── emotion_classifier.py # Lógica de entrenamiento y clasificación de emociones.
    └── cognitive_tutor.py    # Clases para los Expertos y el sistema MoESystem.

La carpeta SAVED_MODELS contiene los modelos entrenados una primera vez para el lanzamiento de la aplicación.

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

Este proyecto está diseñado para ser reproducible. La aplicación interactiva se puede ejecutar localmente o desplegar en servicios como Streamlit Cloud.

#### **Instalación**
1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/YaninaCaffetti/Tuto_Doc.git](https://github.com/YaninaCaffetti/Tuto_Doc.git)
    cd Tuto_Doc
    ```
2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

#### **Ejecución de la Aplicación de Demostración**
El repositorio ya incluye los modelos pre-entrenados para una demostración inmediata. Para lanzar la aplicación interactiva, simplemente ejecuta:

```bash
streamlit run app.py
  ```

## 6. Resumen de Hallazgos

1.  **Modelo Cognitivo:** Se validó que un `RandomForestClassifier` entrenado con datos balanceados por `SMOTE` es la solución óptima para la clasificación de arquetipos, alcanzando un **91% de accuracy** y un **F1-score macro de 0.79**, demostrando su capacidad para predecir clases minoritarias. La mejora sobre un `DecisionTree` simple fue estadísticamente significativa (p < 0.05).
2.  **Clasificador de Emociones:** La estrategia de aumentación de datos por retrotraducción fue **altamente efectiva**, llevando el rendimiento del clasificador a un **rendimiento casi perfecto** en el conjunto de prueba del dominio.
3.  **Trade-off XAI vs. Rendimiento:** La investigación ha cuantificado empíricamente el compromiso entre rendimiento y explicabilidad. El modelo original `IF-HUPM`, aunque 100% interpretable ("caja blanca"), demostró ser frágil y de bajo rendimiento, mientras que el `RandomForest` ("caja negra") ofreció un rendimiento robusto y superior.

## 7. Trabajo Futuro

* **Explicabilidad del Modelo Final:** Aplicar técnicas de XAI post-hoc (como **SHAP** o **LIME**) sobre el `RandomForestClassifier` para intentar explicar sus predicciones y comparar estas explicaciones con las reglas del `IF-HUPM`.
* **Modelos Híbridos:** Explorar arquitecturas que combinen la interpretabilidad del `IF-HUPM` para casos de baja confianza con el rendimiento del `RandomForest` para predicciones de alta confianza.
* **Validación con Usuarios:** Realizar un nuevo estudio formal con usuarios finales para medir cuantitativamente el impacto de la adaptación afectiva en la percepción de empatía, la confianza en el sistema y el éxito en las tareas propuestas.
