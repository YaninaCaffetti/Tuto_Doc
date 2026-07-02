# tests/test_semantic_search.py
"""
Test de Validación Semántica - Tesis Doctoral
----------------------------------------------
Mide la calidad de la búsqueda semántica realizada por los tutores expertos
del MoESystem contra un conjunto de validación de paráfrasis.

NOTA: Este test requiere `torch` y `sentence-transformers`, y descarga el
modelo `hiiamsid/sentence_similarity_spanish_es` la primera vez que corre.
Por eso está marcado como lento (`@pytest.mark.slow`) y se salta
automáticamente si las dependencias no están instaladas, en lugar de
abortar toda la suite de pytest.
"""

import os
import sys

import pandas as pd
import pytest

# Configuración de rutas para importar desde 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

torch = pytest.importorskip("torch", reason="torch no está instalado")
st_util = pytest.importorskip(
    "sentence_transformers", reason="sentence-transformers no está instalado"
).util

from src.cognitive_tutor import get_semantic_model, _build_expert_map, GestorCUD  # noqa: E402

# --- Set de validación (paráfrasis y consultas reales) ---
VALIDATION_SET = [
    {"tutor_name": "TutorCarrera", "prompt_usuario": "Mi currículum está desactualizado, ¿qué le pongo?", "expected_intent_key": "Optimización y mejora de un CV existente"},
    {"tutor_name": "TutorCarrera", "prompt_usuario": "No sé cuánto debería ganar en mi próximo trabajo.", "expected_intent_key": "Estrategias y técnicas de negociación salarial"},
    {"tutor_name": "TutorInteraccion", "prompt_usuario": "Siento que la gente no me entiende cuando hablo.", "expected_intent_key": "Mejora de la claridad didáctica al explicar conceptos"},
    {"tutor_name": "TutorBienestar", "prompt_usuario": "Estoy agotado todo el día.", "expected_intent_key": "Manejo de la fatiga, apatía y agotamiento emocional (burnout)"},
    {"tutor_name": "TutorApoyos", "prompt_usuario": "Si consigo un trabajo, ¿me sacan la pensión?", "expected_intent_key": "Compatibilidad entre pensión por discapacidad y empleo formal"},
    {"tutor_name": "TutorApoyos", "prompt_usuario": "El CUD me sirve para viajar gratis?", "expected_intent_key": "Reglamentación específica del transporte gratuito para acompañante con CUD"},
    {"tutor_name": "GestorCUD", "prompt_usuario": "¿Qué es el CUD?", "expected_intent_key": "Explicación fundamental: Qué es y para qué sirve el CUD"},
    {"tutor_name": "TutorCarrera", "prompt_usuario": "Tengo una charla la semana que viene y estoy nervioso.", "expected_intent_key": "default"},
]

ALIAS_MAP = {
    "TutorCarrera": "Prof_Subutil",
    "TutorInteraccion": "Com_Desafiado",
    "TutorCompetencias": "Nav_Informal",
    "TutorBienestar": "Potencial_Latente",
    "TutorApoyos": "Cand_Nec_Sig",
    "TutorPrimerEmpleo": "Joven_Transicion",
    "GestorCUD": "GestorCUD",
}


@pytest.fixture(scope="module")
def experts():
    """Construye e inicializa el mapa de expertos (carga el modelo BERT una sola vez)."""
    model = get_semantic_model()
    if not model:
        pytest.skip("Modelo semántico no disponible (fallo de descarga o entorno offline).")

    all_experts = _build_expert_map()
    all_experts["GestorCUD"] = GestorCUD()
    for expert in all_experts.values():
        expert._initialize_knowledge_base()
    return all_experts


@pytest.mark.slow
def test_semantic_search_accuracy(experts):
    """La precisión de intención semántica debe superar un piso razonable (>= 0.70)."""
    results = []
    for item in VALIDATION_SET:
        t_name = item["tutor_name"]
        prompt = item["prompt_usuario"]
        expected = item["expected_intent_key"]

        mapped_key = ALIAS_MAP.get(t_name, t_name)
        tutor = experts.get(mapped_key)

        if not tutor or tutor.kb_embeddings is None:
            results.append({"tutor": t_name, "is_correct": False, "found_key": "error_init"})
            continue

        with torch.no_grad():
            import torch.nn.functional as F

            p_emb = F.normalize(get_semantic_model().encode(prompt, convert_to_tensor=True), p=2, dim=0)
            scores = st_util.cos_sim(p_emb, tutor.kb_embeddings)[0]
            best_idx = torch.argmax(scores).item()
            conf = scores[best_idx].item()
            found = tutor.kb_keys[best_idx] if conf > tutor.similarity_threshold else "default"

        results.append({
            "tutor": t_name, "prompt": prompt, "expected_key": expected,
            "found_key": found, "confidence": conf, "is_correct": (found == expected),
        })

    df = pd.DataFrame(results)
    accuracy = df["is_correct"].mean()

    os.makedirs(os.path.join(project_root, "tests"), exist_ok=True)
    df.to_csv(os.path.join(project_root, "tests", "validation_report.csv"), index=False)

    assert accuracy >= 0.70, (
        f"Precisión semántica insuficiente: {accuracy:.2%}\n{df.to_string()}"
    )
