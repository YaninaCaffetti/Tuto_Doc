# tests/test_semantic_search.py
"""
Mide la calidad de la búsqueda semántica realizada por los tutores expertos
del MoESystem contra un conjunto de validación de paráfrasis y consultas trampa.

Métricas Clave Calculadas:
1.  **Precisión (Accuracy):** Porcentaje de prompts que mapean a la intención correcta.
2.  **Cobertura:** Porcentaje de prompts que mapean a una intención específica (no 'default').
3.  **Estabilidad (Confianza):** Media y desviación estándar del score de confianza
    para las coincidencias correctas (excluyendo 'default').

Ejecución:
    python tests/test_semantic_search.py

Salida:
    - Imprime las métricas calculadas en la consola.
    - Genera un archivo CSV ('tests/validation_report.csv') con los resultados detallados por prompt.
    - Genera un archivo JSON ('tests/validation_metrics.json') con las métricas agregadas.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import warnings # Importar warnings si no estaba ya

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    print(f"(test_semantic_search.py) Adding project root to sys.path: {project_root}") # Mensaje de depuración
    sys.path.append(project_root)


try:
    # Ahora esta importación debería funcionar Y permitir la importación anidada
    from src.cognitive_tutor import get_semantic_model, EXPERT_MAP, CUD_EXPERT #
    from sentence_transformers import util
    import torch
except ImportError as e:
    print(f"Error Crítico: No se pudieron importar los módulos necesarios.")
    print(f"Asegúrate de que 'src/cognitive_tutor.py' y 'src/expert_kb.py' existan y sean correctos.")
    print(f"Detalle del error: {e}")
    sys.exit(1)
except KeyError as e:
    print(f"Error Crítico: No se encontró '{e}' esperado en 'cognitive_tutor.py'.")
    print("Verifica las definiciones de EXPERT_MAP y CUD_EXPERT.")
    sys.exit(1)

# --- 1. SET DE VALIDACIÓN DE PARÁFRASIS ---

validation_set = [
    # --- Pruebas para TutorCarrera ---
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "Mi currículum está desactualizado, ¿qué le pongo?",
        "expected_intent_key": "¿Cómo puedo mejorar mi CV?"
    },
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "No sé cuánto debería ganar en mi próximo trabajo.",
        "expected_intent_key": "¿Cómo negociar mi salario?"
    },
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "Me siento estancado y no me valoran, creo que quiero renunciar.",
        "expected_intent_key": "¿Cómo saber si debo cambiar de trabajo?"
    },

    # --- Pruebas para TutorInteraccion ---
    {
        "tutor_name": "TutorInteraccion",
        "prompt_usuario": "Siento que la gente no me entiende cuando hablo.",
        "expected_intent_key": "¿Qué hago si no me entienden cuando explico algo?"
    },
    {
        "tutor_name": "TutorInteraccion",
        "prompt_usuario": "Mi jefe me gritó y no supe qué decir.",
        "expected_intent_key": "¿Cómo manejar una crítica injusta?"
    },

    # --- Pruebas para TutorCompetencias ---
    {
        "tutor_name": "TutorCompetencias",
        "prompt_usuario": "Me bloqueo cuando intento aprender algo nuevo.",
        "expected_intent_key": "¿Cómo superar el miedo a equivocarme cuando aprendo algo nuevo?"
    },
    {
        "tutor_name": "TutorCompetencias",
        "prompt_usuario": "Me distraigo mucho con el celular cuando quiero leer.",
        "expected_intent_key": "Me cuesta concentrarme cuando estudio"
    },

    # --- Pruebas para TutorBienestar ---
    {
        "tutor_name": "TutorBienestar",
        "prompt_usuario": "Estoy agotado todo el día.",
        "expected_intent_key": "Me siento muy cansado últimamente"
    },
    {
        "tutor_name": "TutorBienestar",
        "prompt_usuario": "No paro de pensar en todo lo que tengo que hacer y me paralizo.",
        "expected_intent_key": "Tengo miedo de no poder con todo"
    },

    # --- Pruebas para TutorApoyos ---
    {
        "tutor_name": "TutorApoyos",
        "prompt_usuario": "Si consigo un trabajo, ¿me sacan la pensión?",
        "expected_intent_key": "Tengo miedo de perder mi pensión si empiezo a trabajar"
    },
    {
        "tutor_name": "TutorApoyos",
        "prompt_usuario": "El CUD me sirve para viajar gratis?",
        # Prueba difícil: la respuesta está *dentro* de la intención, no es la clave exacta.
        "expected_intent_key": "¿Qué beneficios tengo con el Certificado Único de Discapacidad?"
    },
    {
        "tutor_name": "TutorApoyos",
        "prompt_usuario": "Fui a la municipalidad y no tienen rampa.",
        "expected_intent_key": "Me enoja que las oficinas públicas no sean accesibles"
    },

    # --- Pruebas para TutorPrimerEmpleo ---
    {
        "tutor_name": "TutorPrimerEmpleo",
        "prompt_usuario": "Quiero trabajar pero nunca trabajé, ¿qué hago?",
        "expected_intent_key": "No tengo experiencia laboral, ¿cómo puedo empezar?"
    },
    {
        "tutor_name": "TutorPrimerEmpleo",
        "prompt_usuario": "Me dijeron que me van a pagar en negro.",
        "expected_intent_key": "¿Qué hago si me piden trabajar sin registrarme?"
    },
    {
        "tutor_name": "TutorPrimerEmpleo",
        "prompt_usuario": "Me preguntaron por mi discapacidad en una entrevista y me sentí mal.",
        "expected_intent_key": "¿Qué hago si me discriminan en una entrevista?"
    },

    # --- Pruebas para GestorCUD ---
    {
        "tutor_name": "GestorCUD",
        "prompt_usuario": "¿Qué es el CUD?",
        "expected_intent_key": "¿Qué es el Certificado Único de Discapacidad?"
    },
    {
        "tutor_name": "GestorCUD",
        "prompt_usuario": "¿Se paga para sacar el CUD?",
        "expected_intent_key": "¿El trámite del CUD tiene costo?"
    },
    {
        "tutor_name": "GestorCUD",
        "prompt_usuario": "Me bocharon el CUD, ¿qué hago ahora?",
        "expected_intent_key": "Me rechazaron el CUD, ¿qué puedo hacer?"
    },


    # --- Prueba de "Consulta Trampa" (True Negative) ---
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "Tengo una charla la semana que viene y estoy nervioso.",
        # Intención pertenece a TutorInteraccion ("Me da miedo presentar en público").
        # TutorCarrera debería devolver 'default'.
        "expected_intent_key": "default"
    },
]

def run_validation():
    """
    Ejecuta el proceso de validación semántica completo.

    Carga el modelo, itera sobre el `validation_set`, realiza la búsqueda
    semántica para cada prompt contra el tutor especificado, calcula las
    métricas de Precisión, Cobertura y Estabilidad, y guarda los resultados.
    """
    print("--- 🔬 Iniciando Validación del Oído Semántico (Tarea 1.2) ---")

    # Cargar el modelo semántico una sola vez
    model = get_semantic_model() #
    if not model:
        print("Error Crítico: No se pudo cargar el modelo semántico. Abortando validación.")
        return

    # Unir todos los expertos (arquetipos + CUD) en un diccionario
    all_experts = EXPERT_MAP.copy() #
    all_experts["GestorCUD"] = CUD_EXPERT #

    results = [] # Lista para almacenar los resultados detallados

    # Pre-calcular embeddings para todos los expertos si aún no lo están
    print(f"› Verificando/Calculando embeddings para {len(all_experts)} expertos...")
    for expert_name, expert_instance in all_experts.items():
        if expert_instance.kb_embeddings is None:
            print(f"  - Calculando para {expert_name}...")
            expert_instance._initialize_knowledge_base() #
    print("› Embeddings listos.")

    # Iterar sobre cada caso de prueba en el set de validación
    # --- 🔧 Mapa de alias entre nombres de tutores y arquetipos (compatibilidad para testeo) ---
    alias_map = {
        "TutorCarrera": "Prof_Subutil",
        "TutorInteraccion": "Com_Desafiado",
        "TutorCompetencias": "Nav_Informal",
        "TutorBienestar": "Potencial_Latente",
        "TutorApoyos": "Cand_Nec_Sig",
        "TutorPrimerEmpleo": "Joven_Transicion",
        "GestorCUD": "GestorCUD"
    }

    print(f"› Ejecutando {len(validation_set)} pruebas de validación...")
    for item in validation_set:
        tutor_name = item["tutor_name"]
        prompt = item["prompt_usuario"]
        expected_key = item["expected_intent_key"]

        # Usar alias si existe
        mapped_name = alias_map.get(tutor_name, tutor_name)
        tutor = all_experts.get(mapped_name)

        # --- Validaciones Previas ---
        if not tutor:
            warnings.warn(f"Saltando prueba: Tutor '{tutor_name}' no encontrado en all_experts.")
            continue
        # Verificar si el tutor tiene embeddings (puede ser None si la KB está vacía o falló la codificación)
        # O si kb_keys está vacío (importante para evitar errores de índice)
        if tutor.kb_embeddings is None or not tutor.kb_keys:
            # Si se espera 'default', y no hay embeddings, es un "acierto" técnico
            if expected_key == "default":
                 found_key = "default"
                 best_match_score = 0.0 # No hubo cálculo de score
                 is_correct = True
            else:
                 # Si se esperaba una clave específica pero no hay embeddings, es un fallo
                 warnings.warn(f"Saltando prueba para {tutor_name}: KB vacía o embeddings no calculados. Prompt: '{prompt}'")
                 found_key = "error_no_kb"
                 best_match_score = 0.0
                 is_correct = False

            results.append({
                "tutor": tutor_name, "prompt": prompt, "expected_key": expected_key,
                "found_key": found_key, "confidence": best_match_score,
                "is_correct": is_correct, "is_default": (found_key == "default")
            })
            continue # Pasar al siguiente item

        # --- Realizar la Búsqueda Semántica ---
        try:
            prompt_embedding = model.encode(prompt, convert_to_tensor=True)
            # Calcular similitud coseno entre el prompt y todas las claves de la KB del tutor
            cos_scores = util.cos_sim(prompt_embedding, tutor.kb_embeddings)[0]
            # Encontrar el índice y score de la mejor coincidencia
            best_match_idx = torch.argmax(cos_scores).item()
            best_match_score = cos_scores[best_match_idx].item()

            found_key = "default" # Asumir default inicialmente
            # Aplicar el umbral específico de CADA tutor
            if best_match_score > tutor.similarity_threshold: #
                # Si supera el umbral, obtener la 'pregunta_clave' correspondiente
                # Asegurarse de que el índice no esté fuera de los límites
                if 0 <= best_match_idx < len(tutor.kb_keys):
                     found_key = tutor.kb_keys[best_match_idx] #
                else:
                     warnings.warn(f"Índice fuera de rango ({best_match_idx}) para {tutor_name}. Usando default.")

            # Comparar la clave encontrada con la esperada
            is_correct = (found_key == expected_key)

            results.append({
                "tutor": tutor_name,
                "prompt": prompt,
                "expected_key": expected_key,
                "found_key": found_key,
                "confidence": best_match_score, # Guardar el score MÁXIMO encontrado
                "is_correct": is_correct,
                "is_default": (found_key == "default")
            })

        except Exception as search_error:
            warnings.warn(f"Error durante búsqueda semántica para {tutor_name} con prompt '{prompt}': {search_error}")
            results.append({
                "tutor": tutor_name, "prompt": prompt, "expected_key": expected_key,
                "found_key": "error_search", "confidence": 0.0,
                "is_correct": False, "is_default": False
            })


    if not results:
        print("Error Crítico: No se generaron resultados. Verifica el `validation_set` y los logs.")
        return

    # --- 2. CALCULAR MÉTRICAS AGREGADAS ---
    df = pd.DataFrame(results)

    # 1. Precisión (Accuracy): % de aciertos sobre el total de pruebas
    accuracy = df["is_correct"].mean() if not df.empty else 0.0

    # 2. Cobertura: % de respuestas que NO fueron 'default'
    coverage = 1.0 - df["is_default"].mean() if not df.empty else 0.0

    # 3. Estabilidad (Media y Std Dev de Confianza): Solo sobre aciertos que NO son 'default'
    correct_matches_df = df[(df["is_correct"] == True) & (df["is_default"] == False)]

    if len(correct_matches_df) > 1:
        confidence_mean = correct_matches_df["confidence"].mean()
        confidence_std = correct_matches_df["confidence"].std()
    elif len(correct_matches_df) == 1:
        # Si hay un solo acierto, la media es su confianza y la std dev es 0
        confidence_mean = correct_matches_df["confidence"].iloc[0]
        confidence_std = 0.0
    else:
        # Si no hay aciertos (correctos y no default), métricas son 0
        confidence_mean = 0.0
        confidence_std = 0.0

    # --- 3. MOSTRAR RESULTADOS ---
    print("\n--- 📊 Resultados de la Validación ---")
    print(f"Total de Pruebas Ejecutadas: {len(df)}")

    # Imprimir métricas con formato y comparación con objetivos
    print(f"\n[Métrica 1: Precisión] (Objetivo: >= 85%)")
    print(f"  › Semantic Intent Accuracy: {accuracy:.2%} {'✅' if accuracy >= 0.85 else '❌'}")

    print(f"\n[Métrica 2: Cobertura] (Objetivo: >= 90%)")
    print(f"  › Coverage (No-Default): {coverage:.2%} {'✅' if coverage >= 0.90 else '❌'}")

    print(f"\n[Métrica 3: Estabilidad] (Objetivo: <= 0.1)")
    print(f"  › Avg. Confidence (Correct Matches): {confidence_mean:.3f}")
    print(f"  › Std. Dev Confidence (Correct Matches): {confidence_std:.3f} {'✅' if confidence_std <= 0.1 else '⚠️'}") # Advertencia si es alta

    # Opcional: Entropía (como se mencionó en las notas doctorales)
    # (Implementación más avanzada, omitida por ahora para claridad)

    # --- 4. DETALLE DE FALLOS ---
    print("\n--- 🧐 Fallos Detallados ---")
    failures_df = df[df["is_correct"] == False]
    if failures_df.empty:
        print("✅ ¡Todas las pruebas pasaron!")
    else:
        print(f"({len(failures_df)} fallos detectados de {len(df)} pruebas)")
        # Imprimir detalles de cada fallo
        for _, row in failures_df.iterrows():
            print(f"  --- Fallo ---")
            print(f"    Tutor: {row['tutor']}")
            print(f"    Prompt: '{row['prompt']}'")
            print(f"    Esperado: '{row['expected_key']}'")
            print(f"    Obtenido: '{row['found_key']}' (Confianza Máx: {row['confidence']:.3f})")

    # --- 5. GUARDAR REPORTES ---
    try:
        # Asegurarse de que el directorio 'tests' exista
        os.makedirs("tests", exist_ok=True)

        # Guardar reporte detallado en CSV
        report_path = os.path.join("tests", "validation_report.csv")
        df.to_csv(report_path, index=False, encoding='utf-8-sig') # Usar encoding para caracteres especiales
        print(f"\nReporte detallado guardado en: '{report_path}'")

        # Guardar métricas agregadas en JSON
        metrics = {
            "accuracy": accuracy,
            "coverage": coverage,
            "confidence_mean_correct": confidence_mean,
            "confidence_std_correct": confidence_std,
            "total_tests": len(df),
            "total_failures": len(failures_df)
        }
        metrics_path = os.path.join("tests", "validation_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"Métricas agregadas guardadas en: '{metrics_path}'")

    except Exception as e:
        warnings.warn(f"Error al guardar los reportes: {e}")

    print("\n--- Validación Finalizada ---")

# Punto de entrada principal para ejecutar la validación
if __name__ == "__main__":
    run_validation()
