# Tuto_Doc: Proyecto de Tesis - Tutor Cognitivo Adaptativo con IA Afectiva
**Autora**: Mgter. Ing. Yanina A. Caffetti
**Institución**: Universidad Nacional de Misiones, Argentina.
**Programa**: Doctorado en Informática

___

# 1. Descripción del Proyecto
   
Este repositorio contiene el prototipo funcional y el pipeline experimental desarrollado como parte de la tesis doctoral "Diseño de un proceso para la integración de un tutor cognitivo adaptativo basado en arquetipos de usuario y computación afectiva".

El proyecto explora la sinergia entre un módulo de razonamiento cognitivo, que clasifica perfiles de usuario complejos, y un módulo de percepción afectiva, que detecta la emoción en el lenguaje del usuario. El objetivo principal es crear un sistema de tutoría que no solo ofrezca un plan de acción basado en el perfil del estudiante, sino que también adapte su interacción y recomendaciones en tiempo real al estado emocional detectado, generando una intervención más holística y empática.

La investigación sigue un riguroso proceso de Machine Learning Operations (MLOps), incluyendo la evaluación de modelos, benchmarking, tratamiento de desbalance de clases, validación estadística y, finalmente, la refactorización a una arquitectura de software modular.

 # 2. Características Principales
   
🧠 Módulo de Razonamiento Cognitivo: Utiliza un modelo RandomForestClassifier para clasificar perfiles de usuario (basados en datos de la encuesta ENDIS 2018) en arquetipos predefinidos heurísticamente. El rendimiento de este componente fue optimizado mediante la técnica SMOTE para manejar el severo desbalance de clases.

❤️ Módulo de Percepción Afectiva: Emplea un modelo de lenguaje BERT (BETO) fine-tuned para clasificar el texto del usuario en una de 8 emociones básicas. Para superar la escasez de datos de dominio, se implementó una estrategia de aumentación de datos por retrotraducción (back-translation).

✨ Sistema de Adaptación Afectiva: Una arquitectura de Mezcla de Expertos (Mixture of Experts) orquesta la respuesta final. El arquetipo predicho por el módulo cognitivo selecciona al "tutor experto" principal, pero la clave de la innovación reside en que el vector completo de probabilidades de emoción modula los pesos de todos los expertos. Esto permite, por ejemplo, que una alta probabilidad de "tristeza" aumente la prioridad del "Tutor de Bienestar", generando un plan de acción mixto y verdaderamente adaptativo.

🔬 Pipeline de Evaluación Riguroso: El proyecto incluye un pipeline completo para el benchmarking comparativo de modelos y la validación de la significancia estadística de los resultados mediante el Test de McNemar.

# 3. Estructura del Proyecto
  
El código está organizado siguiendo las mejores prácticas para facilitar su mantenibilidad y comprensión.

<img width="673" height="402" alt="Captura de Pantalla 2025-08-04 a la(s) 18 40 46" src="https://github.com/user-attachments/assets/3f2e553b-9b0f-4718-b7dd-f04021686da7" />


# 4. Metodología y Tecnologías
   
<img width="685" height="409" alt="Captura de Pantalla 2025-08-04 a la(s) 18 36 37" src="https://github.com/user-attachments/assets/e9b959bf-5cf3-43e8-822b-4f38238f4915" />


# 5. Instalación y Ejecución

Este proyecto está diseñado para ser reproducible. La aplicación interactiva se puede ejecutar localmente o desplegar en servicios como Streamlit Cloud.

**Instalación**
Clona el repositorio:

git clone https://github.com/YaninaCaffetti/Tuto_Doc.git
cd Tuto_Doc

Descargar los modelos grandes:
Este repositorio usa Git LFS. Para descargar los modelos, necesitas tener Git LFS instalado.

   Instalar Git LFS (solo se hace una vez por máquina)
   .En macOS: brew install git-lfs
   .En Windows/Linux: ver https://git-lfs.github.com/

git lfs install
git lfs pull

Crea un entorno virtual y activa:

python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

Instala las dependencias:

pip install -r requirements.txt

Ejecución de la Aplicación de Demostración:

El repositorio ya incluye los modelos pre-entrenados. Para lanzar la aplicación interactiva, simplemente ejecuta:

streamlit run app.py

Re-entrenamiento de Modelos (Opcional)

Si deseas volver a generar todos los modelos desde cero:

Asegúrate de tener el dataset base_estudio_discapacidad_2018.csv en la carpeta data/.

Ejecuta el pipeline de entrenamiento completo:

python train.py

# 6. Resumen de Hallazgos
   
**Modelo Cognitivo**: Se validó que un RandomForestClassifier entrenado con datos balanceados por SMOTE es una solución óptima para la clasificación de arquetipos, alcanzando un 91% de accuracy y un F1-score macro de 0.79.

**Clasificador de Emociones**: La estrategia de aumentación de datos por retrotraducción fue altamente efectiva. El modelo BERT fine-tuned alcanzó un rendimiento perfecto en el conjunto de prueba del dominio (F1-score macro de 1.00), superando significativamente al robusto benchmark clásico (F1-score de 0.96).

**Trade-off XAI vs. Rendimiento**: La investigación ha cuantificado empíricamente el compromiso entre rendimiento y explicabilidad. El modelo IF-HUPM ("caja blanca") demostró ser frágil, mientras que el RandomForest ("caja negra") ofreció un rendimiento robusto y superior.

# 7. Trabajo Futuro
   
**Explicabilidad del Modelo Final**: Aplicar técnicas de XAI post-hoc (como SHAP o LIME) sobre el RandomForestClassifier para intentar explicar sus predicciones.

**Validación con Usuarios**: Realizar un estudio formal con usuarios finales para medir cuantitativamente el impacto de la adaptación afectiva en la percepción de empatía y la confianza en el sistema.

**Aprendizaje de Reglas Afectivas**: Explorar métodos para aprender las affective_rules desde datos de interacción, en lugar de definirlas heurísticamente.adaptación afectiva en la percepción de empatía, la confianza en el sistema y el éxito en las tareas propuestas.
