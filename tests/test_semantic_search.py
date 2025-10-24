# tests/test_semantic_search.py
"""
Script de Validación

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
    import torch.nn.functional as F # *** NECESARIO PARA NORMALIZAR ***
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
# Define las consultas de prueba y la intención esperada para cada una.
# Incluye paráfrasis (True Positives) y consultas trampa (True Negatives).
# *** ESTA LISTA HA SIDO ACTUALIZADA A LAS NUEVAS CLAVES FUNCIONALES ***

validation_set = [
    # --- Pruebas para TutorCarrera ---
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "Mi currículum está desactualizado, ¿qué le pongo?",
        "expected_intent_key": "Optimización y mejora de un CV existente" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "No sé cuánto debería ganar en mi próximo trabajo.",
        "expected_intent_key": "Estrategias y técnicas de negociación salarial" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "Me siento estancado y no me valoran, creo que quiero renunciar.",
        "expected_intent_key": "Evaluación personal para decidir un cambio de empleo" # ACTUALIZADO
    },

    # --- Pruebas para TutorInteraccion ---
    {
        "tutor_name": "TutorInteraccion",
        "prompt_usuario": "Siento que la gente no me entiende cuando hablo.",
        "expected_intent_key": "Mejora de la claridad didáctica al explicar conceptos" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorInteraccion",
        "prompt_usuario": "Mi jefe me gritó y no supe qué decir.",
        "expected_intent_key": "Recepción asertiva de críticas laborales (justas o injustas)" # ACTUALIZADO
    },

    # --- Pruebas para TutorCompetencias ---
    {
        "tutor_name": "TutorCompetencias",
        "prompt_usuario": "Me bloqueo cuando intento aprender algo nuevo.",
        "expected_intent_key": "Gestión del miedo al error durante el aprendizaje" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorCompetencias",
        "prompt_usuario": "Me distraigo mucho con el celular cuando quiero leer.",
        "expected_intent_key": "Técnicas de concentración para el estudio (Pomodoro)" # ACTUALIZADO
    },

    # --- Pruebas para TutorBienestar ---
    {
        "tutor_name": "TutorBienestar",
        "prompt_usuario": "Estoy agotado todo el día.",
        "expected_intent_key": "Manejo de la fatiga, apatía y agotamiento emocional (burnout)" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorBienestar",
        "prompt_usuario": "No paro de pensar en todo lo que tengo que hacer y me paralizo.",
        "expected_intent_key": "Estrategias de gestión de la ansiedad y el estrés por sobrecarga" # ACTUALIZADO
    },

    # --- Pruebas para TutorApoyos ---
    {
        "tutor_name": "TutorApoyos",
        "prompt_usuario": "Si consigo un trabajo, ¿me sacan la pensión?",
        "expected_intent_key": "Compatibilidad entre pensión por discapacidad y empleo formal" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorApoyos",
        "prompt_usuario": "El CUD me sirve para viajar gratis?",
        # Nota: Esta es una intención difícil. "Listado general..." habla de transporte.
        # "Reglamentación específica..." también. Mantenemos el más general.
        "expected_intent_key": "Reglamentación específica del transporte gratuito para acompañante con CUD. Listado general de beneficios y derechos otorgados por el CUD" 
    },
    {
        "tutor_name": "TutorApoyos",
        "prompt_usuario": "Fui a la municipalidad y no tienen rampa.",
        "expected_intent_key": "Gestión de reclamos por falta de accesibilidad física (Ley 24.314)" # ACTUALIZADO
    },

    # --- Pruebas para TutorPrimerEmpleo ---
    {
        "tutor_name": "TutorPrimerEmpleo",
        "prompt_usuario": "Quiero trabajar pero nunca trabajé, ¿qué hago?",
        "expected_intent_key": "Estrategias de inserción laboral sin experiencia previa (Programa Jóvenes)" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorPrimerEmpleo",
        "prompt_usuario": "Me dijeron que me van a pagar en negro.",
        "expected_intent_key": "Acciones y denuncias contra el trabajo no registrado" # ACTUALIZADO
    },
    {
        "tutor_name": "TutorPrimerEmpleo",
        "prompt_usuario": "Me preguntaron por mi discapacidad en una entrevista y me sentí mal.",
        "expected_intent_key": "Procedimiento legal por discriminación en entrevistas (Ley 23.592, ADAJUS)" 
    },

    # --- Pruebas para GestorCUD ---
    {
        "tutor_name": "GestorCUD",
        "prompt_usuario": "¿Qué es el CUD?",
        "expected_intent_key": "Explicación fundamental: Qué es y para qué sirve el CUD" 
    },
    {
        "tutor_name": "GestorCUD",
        "prompt_usuario": "¿Se paga para sacar el CUD?",
        "expected_intent_key": "Confirmación de gratuidad y denuncia de cobros indebidos" 
    },
    {
        "tutor_name": "GestorCUD",
        "prompt_usuario": "Me bocharon el CUD, ¿qué hago ahora?",
        "expected_intent_key": "Procedimiento de apelación o revisión ante rechazo del CUD" 
    },


    # --- Prueba de "Consulta Trampa" (True Negative) ---
    {
        "tutor_name": "TutorCarrera",
        "prompt_usuario": "Tengo una charla la semana que viene y estoy nervioso.",
        # Intención pertenece a TutorInteraccion ("Gestión del miedo a hablar en público...").
        # TutorCarrera debería devolver 'default' (score < 0.50).
        "expected_intent_key": "default" # REVISADO (OK)
    },
]

def run_validation():
    """
    Ejecuta el proceso de validación semántica completo.

    Carga el modelo, itera sobre el `validation_set`, realiza la búsqueda
    semántica para cada prompt contra el tutor especificado, calcula las
    métricas de Precisión, Cobertura y Estabilidad, y guarda los resultados.
    """
    print("--- 🔬 Iniciando Validación del Oído Semántico")

    # Cargar el modelo semántico una sola vez
    model = get_semantic_model() #
    if not model:
        print("Error Crítico: No se pudo cargar el modelo semántico. Abortando validación.")
        return

    # Unir todos los expertos (arquetipos + CUD) en un diccionario
    all_experts = EXPERT_MAP.copy() #
    all_experts["GestorCUD"] = CUD_EXPERT #

    results = [] # Lista para almacenar los resultados detallados

    # --- 🔧 Mapa de alias (MOVIDO AL INICIO) ---
    # Mapa de alias entre nombres de tutores y arquetipos (compatibilidad para testeo)
    alias_map = {
        "TutorCarrera": "Prof_Subutil",
        "TutorInteraccion": "Com_Desafiado",
        "TutorCompetencias": "Nav_Informal",
        "TutorBienestar": "Potencial_Latente",
        "TutorApoyos": "Cand_Nec_Sig",
        "TutorPrimerEmpleo": "Joven_Transicion",
        "GestorCUD": "GestorCUD"
    }

    # Pre-calcular embeddings para todos los expertos si aún no lo están
    print(f"› Forzando inicialización de embeddings para {len(all_experts)} expertos...")
    model_check = get_semantic_model() # Asegurar que el modelo esté cargado
    if not model_check:
        print("Error Crítico: Modelo semántico no disponible para inicializar embeddings.")
        return
    
    # *** INICIO DE LA CORRECCIÓN ***
    # Usar el alias_map (definido arriba) para el log
    for expert_name, expert_instance in all_experts.items():
        # Mapeo inverso para log
        tutor_display_name = {v: k for k, v in alias_map.items()}.get(expert_name, expert_name)
        print(f"  - Inicializando {tutor_display_name}...")
        expert_instance._initialize_knowledge_base() # Llamada explícita
    # *** FIN DE LA CORRECCIÓN ***
    
    print("› Embeddings (re)inicializados.")


    print(f"› Ejecutando {len(validation_set)} pruebas de validación...")
    for item in validation_set:
        tutor_name = item["tutor_name"]
        prompt = item["prompt_usuario"]
        expected_key = item["expected_intent_key"]

        # Usar el alias si existe; si no, mantener el nombre original
        mapped_name = alias_map.get(tutor_name, tutor_name)
        tutor = all_experts.get(mapped_name)

        # --- Validaciones Previas ---
        if not tutor:
            warnings.warn(f"Saltando prueba: Tutor '{tutor_name}' (Mapeado a '{mapped_name}') no encontrado en all_experts.")
            continue
        if tutor.kb_embeddings is None or not tutor.kb_keys:
            if expected_key == "default":
                found_key = "default"
                best_match_score = 0.0
                is_correct = True
            else:
                warnings.warn(f"Saltando prueba para {tutor_name}: KB vacía o embeddings no calculados. Prompt: '{prompt}'")
                found_key = "error_no_kb"
                best_match_score = 0.0
                is_correct = False

            results.append({
                "tutor": tutor_name, "prompt": prompt, "expected_key": expected_key,
                "found_key": found_key, "confidence": best_match_score,
                "is_correct": is_correct, "is_default": (found_key == "default")
            })
            continue

        # --- Realizar la Búsqueda Semántica ---
        try:
            # 1. Codificar el prompt
            prompt_embedding_raw = model.encode(prompt, convert_to_tensor=True)
            
            # 2. *** ARREGLO CRÍTICO: Normalizar el embedding del prompt ***
            # Esto debe coincidir con la lógica en cognitive_tutor.py
            prompt_embedding = F.normalize(prompt_embedding_raw, p=2, dim=0)

            # 3. Calcular similitud (dot product de vectores normalizados)
            # Mover ambos tensores al mismo dispositivo (ej. 'cuda:0' o 'cpu')
            device = prompt_embedding.device
            kb_embeds_device = tutor.kb_embeddings.to(device)
            
            cos_scores = util.cos_sim(prompt_embedding, kb_embeds_device)[0]
            best_match_idx = torch.argmax(cos_scores).item()
            best_match_score = cos_scores[best_match_idx].item()

            found_key = "default"
            if best_match_score > tutor.similarity_threshold: #
                if 0 <= best_match_idx < len(tutor.kb_keys):
                    found_key = tutor.kb_keys[best_match_idx] #
                else:
                    warnings.warn(f"Índice fuera de rango ({best_match_idx}) para {tutor_name}. Usando default.")

            is_correct = (found_key == expected_key)

            results.append({
                "tutor": tutor_name,
                "prompt": prompt,
                "expected_key": expected_key,
                "found_key": found_key,
                "confidence": best_match_score,
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
    accuracy = df["is_correct"].mean() if not df.empty else 0.0
    coverage = 1.0 - df["is_default"].mean() if not df.empty else 0.0
    correct_matches_df = df[(df["is_correct"] == True) & (df["is_default"] == False)]

    if len(correct_matches_df) > 1:
        confidence_mean = correct_matches_df["confidence"].mean()
        confidence_std = correct_matches_df["confidence"].std()
    elif len(correct_matches_df) == 1:
        confidence_mean = correct_matches_df["confidence"].iloc[0]
        confidence_std = 0.0
    else:
        confidence_mean = 0.0
        confidence_std = 0.0

    # --- 3. MOSTRAR RESULTADOS ---
    print("\n--- 📊 Resultados de la Validación ---")
    print(f"Total de Pruebas Ejecutadas: {len(df)}")
    print(f"\n[Métrica 1: Precisión] (Objetivo: >= 85%)")
    print(f"  › Semantic Intent Accuracy: {accuracy:.2%} {'✅' if accuracy >= 0.85 else '❌'}")
    print(f"\n[Métrica 2: Cobertura] (Objetivo: >= 90%)")
    print(f"  › Coverage (No-Default): {coverage:.2%} {'✅' if coverage >= 0.90 else '❌'}")
    print(f"\n[Métrica 3: Estabilidad] (Objetivo: <= 0.1)")
    print(f"  › Avg. Confidence (Correct Matches): {confidence_mean:.3f}")
    print(f"  › Std. Dev Confidence (Correct Matches): {confidence_std:.3f} {'✅' if confidence_std <= 0.1 else '⚠️'}")

    # --- 4. DETALLE DE FALLOS ---
    print("\n--- 🧐 Fallos Detallados ---")
    failures_df = df[df["is_correct"] == False]
    if failures_df.empty:
        print("✅ ¡Todas las pruebas pasaron!")
    else:
        print(f"({len(failures_df)} fallos detectados de {len(df)} pruebas)")
        for _, row in failures_df.iterrows():
            print(f"  --- Fallo ---")
            print(f"    Tutor: {row['tutor']}")
            print(f"    Prompt: '{row['prompt']}'")
            print(f"    Esperado: '{row['expected_key']}'")
            print(f"    Obtenido: '{row['found_key']}' (Confianza Máx: {row['confidence']:.3f})")

    # --- 5. GUARDAR REPORTES ---
    try:
        os.makedirs("tests", exist_ok=True)
        report_path = os.path.join("tests", "validation_report.csv")
        df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\nReporte detallado guardado en: '{report_path}'")
        metrics = {
            "accuracy": accuracy, "coverage": coverage,
            "confidence_mean_correct": confidence_mean, "confidence_std_correct": confidence_std,
            "total_tests": len(df), "total_failures": len(failures_df)
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
