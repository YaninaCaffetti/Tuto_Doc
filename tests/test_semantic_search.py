# tests/test_semantic_search.py
"""
Script de Validación Semántica - Tesis Doctoral
----------------------------------------------
Mide la calidad de la búsqueda semántica realizada por los tutores expertos
del MoESystem contra un conjunto de validación de paráfrasis.

Métricas: Accuracy, Cobertura y Estabilidad (Confianza).
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
import torch
import torch.nn.functional as F

# Configuración de rutas para importar desde 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
   
    from src.cognitive_tutor import get_semantic_model, _build_expert_map, GestorCUD, Experto
    from sentence_transformers import util
except ImportError as e:
    print(f"Error Crítico de Dependencias: {e}")
    sys.exit(1)

# --- 1. SET DE VALIDACIÓN (Paráfrasis y Consultas Reales) ---
validation_set = [
    {"tutor_name": "TutorCarrera", "prompt_usuario": "Mi currículum está desactualizado, ¿qué le pongo?", "expected_intent_key": "Optimización y mejora de un CV existente"},
    {"tutor_name": "TutorCarrera", "prompt_usuario": "No sé cuánto debería ganar en mi próximo trabajo.", "expected_intent_key": "Estrategias y técnicas de negociación salarial"},
    {"tutor_name": "TutorInteraccion", "prompt_usuario": "Siento que la gente no me entiende cuando hablo.", "expected_intent_key": "Mejora de la claridad didáctica al explicar conceptos"},
    {"tutor_name": "TutorBienestar", "prompt_usuario": "Estoy agotado todo el día.", "expected_intent_key": "Manejo de la fatiga, apatía y agotamiento emocional (burnout)"},
    {"tutor_name": "TutorApoyos", "prompt_usuario": "Si consigo un trabajo, ¿me sacan la pensión?", "expected_intent_key": "Compatibilidad entre pensión por discapacidad y empleo formal"},
    {"tutor_name": "TutorApoyos", "prompt_usuario": "El CUD me sirve para viajar gratis?", "expected_intent_key": "Reglamentación específica del transporte gratuito para acompañante con CUD"},
    {"tutor_name": "GestorCUD", "prompt_usuario": "¿Qué es el CUD?", "expected_intent_key": "Explicación fundamental: Qué es y para qué sirve el CUD"},
    {"tutor_name": "TutorCarrera", "prompt_usuario": "Tengo una charla la semana que viene y estoy nervioso.", "expected_intent_key": "default"} 
]

def run_validation():
    print("--- 🔬 Iniciando Validación del Oído Semántico (FIX Aplicado)")

    model = get_semantic_model()
    if not model:
        print("Error: Modelo BERT no disponible."); return

    # RECONSTRUCCIÓN DEL MAPA DE EXPERTOS (Alineado con src.cognitive_tutor)
    all_experts = _build_expert_map()
    all_experts["GestorCUD"] = GestorCUD()

    # Mapeo de nombres para compatibilidad con el set de validación
    alias_map = {
        "TutorCarrera": "Prof_Subutil",
        "TutorInteraccion": "Com_Desafiado",
        "TutorCompetencias": "Nav_Informal",
        "TutorBienestar": "Potencial_Latente",
        "TutorApoyos": "Cand_Nec_Sig",
        "TutorPrimerEmpleo": "Joven_Transicion",
        "GestorCUD": "GestorCUD"
    }

    print(f"› Inicializando KB y Embeddings para {len(all_experts)} expertos...")
    for expert in all_experts.values():
        expert._initialize_knowledge_base()

    results = []
    for item in validation_set:
        t_name = item["tutor_name"]
        prompt = item["prompt_usuario"]
        expected = item["expected_intent_key"]
        
        mapped_key = alias_map.get(t_name, t_name)
        tutor = all_experts.get(mapped_key)

        if not tutor or tutor.kb_embeddings is None:
            results.append({"tutor": t_name, "is_correct": False, "found_key": "error_init"}); continue

        # Ejecución de Búsqueda Semántica
        with torch.no_grad():
            p_emb = F.normalize(model.encode(prompt, convert_to_tensor=True), p=2, dim=0)
            scores = util.cos_sim(p_emb, tutor.kb_embeddings)[0]
            best_idx = torch.argmax(scores).item()
            conf = scores[best_idx].item()
            
            found = tutor.kb_keys[best_idx] if conf > tutor.similarity_threshold else "default"
            
        results.append({
            "tutor": t_name, "prompt": prompt, "expected_key": expected,
            "found_key": found, "confidence": conf, "is_correct": (found == expected),
            "is_default": (found == "default")
        })

    # Procesamiento de Métricas para la Tesis
    df = pd.DataFrame(results)
    accuracy = df["is_correct"].mean()
    print(f"\n--- 📊 Resultados Finales ---")
    print(f"Precisión (Accuracy): {accuracy:.2%} {'✅' if accuracy >= 0.85 else '⚠️'}")
    
    # Guardar reporte para auditoría del Capítulo V
    os.makedirs("tests", exist_ok=True)
    df.to_csv("tests/validation_report.csv", index=False)
    print("› Reporte 'tests/validation_report.csv' generado exitosamente.")

if __name__ == "__main__":
    run_validation()
