# Tuto_Doc: Proyecto de Tesis - Tutor Cognitivo Adaptativo con IA Afectiva.

## Autora: Mgter. Ing. Yanina A. Caffetti

### Institución: Universidad Nacional de Misiones, Argentina.

### Programa: Doctorado en Informática

---

## 1. Descripción del Proyecto:
   
Este repositorio contiene el prototipo funcional y el pipeline experimental desarrollado como parte de la tesis doctoral "Diseño de un proceso para la integración de un tutor cognitivo adaptativo basado en arquetipos de usuario y computación afectiva".

El proyecto explora la sinergia entre un módulo de razonamiento cognitivo, que clasifica perfiles de usuario complejos, y un módulo de percepción afectiva, que detecta la emoción en el lenguaje del usuario. El objetivo principal es crear un sistema de tutoría que no solo ofrezca un plan de acción basado en el perfil del estudiante, sino que también adapte su interacción y recomendaciones en tiempo real al estado emocional detectado, generando una intervención más holística y empática.

La investigación sigue un riguroso proceso de Machine Learning Operations (MLOps), incluyendo la evaluación de modelos, benchmarking, tratamiento de desbalance de clases, validación estadística y, finalmente, la refactorización a una arquitectura de software modular.

### Palabras Clave: Tutor Cognitivo, Computación Afectiva, Mezcla de Expertos (MoE), MLOps, Procesamiento del Lenguaje Natural (PLN), XAI.

---

## 2. Características Principales:
   
🧠 Módulo de Razonamiento Cognitivo: Utiliza un modelo RandomForestClassifier para clasificar perfiles de usuario (basados en datos de la encuesta ENDIS 2018) en arquetipos predefinidos heurísticamente. Su rendimiento fue optimizado para manejar el desbalance de clases con la técnica SMOTE durante el entrenamiento.

❤️ Módulo de Percepción Afectiva: Emplea un modelo de la familia BETO (dccuchile/bert-base-spanish-wwm-cased) fine-tuned para clasificar el texto del usuario en una de 7 emociones clave. Para superar la escasez de datos de dominio, se implementó una estrategia de aumentación de datos por retrotraducción (back-translation).

✨ Sistema de Adaptación Afectiva: Una arquitectura de Mezcla de Expertos (Mixture of Experts) orquesta la respuesta final. El arquetipo predicho por el módulo cognitivo selecciona al "tutor experto" principal, pero la clave de la innovación reside en que el vector completo de probabilidades de emoción modula los pesos de todos los expertos. Esto permite, por ejemplo, que una alta probabilidad de "tristeza" aumente la prioridad del "Tutor de Bienestar", generando un plan de acción mixto y verdaderamente adaptativo.

🔬 Pipeline de Evaluación Riguroso: El proyecto incluye un pipeline completo para el benchmarking de modelos y la validación de la significancia estadística de los resultados, utilizando el Test de McNemar para comparar el rendimiento del modelo final contra los benchmarks establecidos.

---

## 3. Estructura del Proyecto:
   
El código está organizado siguiendo las mejores prácticas para facilitar su mantenibilidad y comprensión.
````
.
├── data/
│   ├── raw/
│   └── cognitive_profiles.csv
├── models/
│   ├── cognitive_tutor.joblib
│   └── emotion_classifier/
├── src/
│   ├── __init__.py
│   ├── cognitive_model_trainer.py
│   ├── cognitive_tutor.py
│   ├── data_processing.py
│   └── emotion_classifier.py
├── app.py
├── config.yaml
├── requirements.txt
└── train.py

````
---

## 4. Metodología y Tecnologías:
   
Lenguaje y Librerías Principales: Python, PyTorch, Scikit-learn, Pandas.

NLP y Transformers: Hugging Face (Transformers, Datasets).

MLOps y Experimentación: MLflow.

Aplicación Web: Streamlit.

---

## 5. Instalación y Ejecución:
   
Este proyecto está diseñado para ser reproducible. La aplicación interactiva se puede ejecutar localmente o desplegar en servicios como Streamlit Cloud.

Instalación
Clona el repositorio:

```
git clone https://github.com/YaninaCaffetti/Tuto_Doc.git
cd Tuto_Doc
```

Descargar los modelos grandes (Git LFS):
Este repositorio usa Git LFS. Si no lo tienes, instálalo desde git-lfs.github.com.

```
git lfs install
git lfs pull
```

Crea y activa un entorno virtual:

```
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

Instala las dependencias:
El archivo requirements.txt contiene las dependencias mínimas y probadas del proyecto.

```
pip install -r requirements.txt
```
Ejecución de la Aplicación de Demostración

El repositorio ya incluye los modelos pre-entrenados. Para lanzar la aplicación interactiva, simplemente ejecuta:

```
streamlit run app.py
```
Re-entrenamiento de Modelos (Opcional)
Si deseas volver a generar todos los modelos desde cero:

Asegúrate de tener tu dataset en la ruta correcta: data/raw/base_estudio_discapacidad_2018.csv.

Ejecuta el pipeline de entrenamiento completo:

```
python train.py --model all
```

---

## 6. Resumen de Hallazgos:
   
Modelo Cognitivo: Se validó que un RandomForestClassifier entrenado con datos balanceados por SMOTE es una solución óptima para la clasificación de arquetipos, alcanzando un 91% de accuracy y un F1-score macro de 0.79.

Clasificador de Emociones: La estrategia de aumentación de datos por retrotraducción fue altamente efectiva. El modelo BETO fine-tuned alcanzó un rendimiento perfecto en el conjunto de prueba del dominio (F1-score macro de 1.00), superando significativamente al robusto benchmark clásico (F1-score de 0.96).

Trade-off XAI vs. Rendimiento: Se demostró un claro compromiso entre la interpretabilidad y el rendimiento predictivo. Mientras que un modelo de "caja blanca" (IF-HUPM) resultó frágil, el enfoque de "caja negra" (RandomForestClassifier) proporcionó una solución robusta y superior, justificando empíricamente la necesidad de aplicar técnicas XAI post-hoc.

---

## 7. Trabajo Futuro:
   
Explicabilidad del Modelo Final: Aplicar técnicas de XAI post-hoc (como SHAP o LIME) sobre el RandomForestClassifier para intentar explicar sus predicciones.

Validación con Usuarios: Realizar un estudio formal con usuarios finales para medir cuantitativamente el impacto de la adaptación afectiva en la percepción de empatía, la confianza en el sistema y el éxito en las tareas propuestas.

Aprendizaje de Reglas Afectivas: Explorar métodos para aprender las affective_rules desde datos de interacción, en lugar de definirlas heurísticamente.
