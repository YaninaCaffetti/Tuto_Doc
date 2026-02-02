"""
Define la arquitectura del Tutor Cognitivo y el Sistema de Mezcla de Expertos (MoE).

Este módulo contiene:
- La clase base `Experto` que maneja la carga de la Base de Conocimiento (KB)
  y la búsqueda semántica de intenciones basada en centroides.
- Subclases de `Experto` para cada rol de tutoría específico (ej. TutorCarrera).
- La clase `MoESystem` que orquesta la predicción cognitiva (con Guardrails Neuro-Simbólicos),
  la modulación afectiva/contextual y la selección final de recomendaciones.
- Un gestor singleton para el modelo de embeddings semánticos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
from importlib import import_module
import logging
import datetime
import os

# Configuración del Logger local (Buenas prácticas)
logger = logging.getLogger(__name__)

# Constantes de Calibración Neuro-Simbólica
UNI_MEMB_THRESHOLD = 0.7  # Umbral de membresía difusa para considerar Título Universitario

# --- 1. IMPORTACIONES PARA LA BÚSQUEDA SEMÁNTICA ---
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    import torch.nn.functional as F  # Para normalización L2
    SEMANTIC_SEARCH_ENABLED = True
except ImportError:
    warnings.warn(
        "`sentence-transformers` no está instalado. "
        "El tutor recurrirá a la búsqueda simple por palabras clave."
    )
    SEMANTIC_SEARCH_ENABLED = False
    # Clases Dummy para evitar errores de importación
    class SentenceTransformer: pass
    class util: pass
    class torch:
        class Tensor: pass
        @staticmethod
        def mean(*args, **kwargs): pass
        @staticmethod
        def stack(*args, **kwargs): pass
        @staticmethod
        def argmax(*args, **kwargs): pass
        @staticmethod
        def tensor(*args, **kwargs): pass
    class F:
        @staticmethod
        def normalize(*args, **kwargs): pass

# --- 2. FUNCIONES AUXILIARES (Neuro-Simbólicas / Guardrails) ---

def _truthy(x) -> bool:
    """
    Normaliza valores heterogéneos (Excel, CSV, bools) a un booleano estricto.
    """
    if x is None: return False
    if pd.isna(x): return False
    if isinstance(x, (bool, np.bool_)): return bool(x)
    s = str(x).strip().lower()
    return s in {"si", "sí", "true", "1", "t", "y", "yes", "presenta", "tiene", "s"}

def _get_university_flag(user_profile: pd.Series) -> bool:
    """
    Heurística robusta para detectar título universitario.
    Prioriza variables semánticas duras sobre membresías difusas.
    """
    candidate_cols = [
        "TIENE_TITULO_UNI", "TIENE_TITULO_UNIVERSITARIO", "TITULO_UNIVERSITARIO",
        "POSEE_TITULO_UNIVERSITARIO", "UNIVERSITARIO", "NIVEL_UNIVERSITARIO",
        "MNEA",          # Variable dura (Nivel educativo numérico, 5=Uni)
        "CH_Alto_memb"   # Fallback: Membresía difusa
    ]
    
    for col in candidate_cols:
        if col in user_profile.index:
            val = user_profile[col]
            if pd.isna(val): continue

            if col == 'MNEA':
                try:
                    return int(float(val)) == 5
                except (ValueError, TypeError): continue 
            
            if col == 'CH_Alto_memb':
                try:
                    return float(val) >= UNI_MEMB_THRESHOLD
                except (ValueError, TypeError): continue

            if _truthy(val): return True

    # Fallback heurístico por nombre de columna
    for col in user_profile.index:
        c = str(col).lower()
        if "titulo" in c and ("uni" in c or "univers" in c):
            if not pd.isna(user_profile[col]) and _truthy(user_profile[col]):
                return True
    return False

# --- 3. GESTOR DEL MODELO SEMÁNTICO (SINGLETON) ---

_SEMANTIC_MODEL = None
_MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es'

def get_semantic_model() -> Optional[SentenceTransformer]:
    """Carga y devuelve el modelo de SentenceTransformer como un singleton."""
    global _SEMANTIC_MODEL
    if not SEMANTIC_SEARCH_ENABLED: return None

    if _SEMANTIC_MODEL is None:
        print(f"› Cargando modelo semántico: {_MODEL_NAME}...")
        try:
            _SEMANTIC_MODEL = SentenceTransformer(_MODEL_NAME)
            print("› ✅ Modelo semántico cargado exitosamente.")
        except Exception as e:
            warnings.warn(f"Error cargando modelo semántico: {e}. Se desactivará.")
            _SEMANTIC_MODEL = None
    return _SEMANTIC_MODEL

# --- 4. ARQUITECTURA DE EXPERTOS CONVERSACIONALES ---

class Experto:
    """Clase base para los tutores expertos del sistema MoE."""
    def __init__(self, nombre_experto: str):
        self.nombre = nombre_experto
        self.knowledge_base: List[Dict[str, Any]] = []
        self.kb_keys: List[str] = []
        self.kb_embeddings = None
        self.similarity_threshold = 0.40
        self._load_kb()

    def _load_kb(self):
        """Carga la KB dinámicamente desde src.expert_kb."""
        try:
            kb_module = import_module("src.expert_kb")
            self.knowledge_base = kb_module.EXPERT_KB.get(self.nombre, [])
            if not self.knowledge_base:
                warnings.warn(f"KB vacía para: {self.nombre}")
        except Exception as e:
            warnings.warn(f"Error cargando KB para {self.nombre}: {e}")

    def _initialize_knowledge_base(self):
        """Pre-calcula embeddings de centroides para la KB."""
        model = get_semantic_model()
        if not model or not self.knowledge_base:
            self.kb_embeddings = None
            return

        new_kb_keys = []
        all_centroids = []
        try:
            for item in self.knowledge_base:
                key = item.get("pregunta_clave")
                if not key or key == "default": continue
                
                variantes = [str(t) for t in item.get("variantes", []) if t]
                if not variantes: continue

                embeddings = model.encode(variantes, convert_to_tensor=True)
                if embeddings.shape[0] > 0:
                    centroid = torch.mean(embeddings, dim=0)
                    all_centroids.append(F.normalize(centroid, p=2, dim=0))
                    new_kb_keys.append(key)

            self.kb_keys = new_kb_keys
            if all_centroids:
                self.kb_embeddings = torch.stack(all_centroids)
            else:
                self.kb_embeddings = None
        except Exception as e:
            warnings.warn(f"Error codificando KB para {self.nombre}: {e}")
            self.kb_embeddings = None

    def _get_intention_by_key(self, key: str) -> Dict[str, Any]:
        for item in self.knowledge_base:
            if item.get("pregunta_clave") == key: return item
        return {
            "pregunta_clave": "default",
            "respuesta": f"Analicemos tu situación con {self.nombre}.",
            "contexto_emocional_esperado": "neutral",
            "tags": [], "variantes": []
        }

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        model = get_semantic_model()
        matched_intention = None
        
        # 1. Búsqueda Semántica
        if model and self.kb_embeddings is not None and len(self.kb_keys) > 0:
            try:
                prompt_emb = F.normalize(model.encode(prompt, convert_to_tensor=True), p=2, dim=0)
                scores = util.cos_sim(prompt_emb, self.kb_embeddings.to(prompt_emb.device))[0]
                best_idx = torch.argmax(scores).item()
                if scores[best_idx].item() > self.similarity_threshold:
                    matched_intention = self._get_intention_by_key(self.kb_keys[best_idx])
            except Exception as e:
                warnings.warn(f"Error búsqueda semántica {self.nombre}: {e}")

        # 2. Fallback: Palabras Clave
        if matched_intention is None:
            prompt_lower = prompt.lower()
            for item in self.knowledge_base:
                key = item.get("pregunta_clave", "").lower()
                vars_lower = [str(v).lower() for v in item.get("variantes", [])]
                if key in prompt_lower or any(v in prompt_lower for v in vars_lower if len(v) > 3):
                    matched_intention = item
                    break

        # 3. Default
        if matched_intention is None:
            matched_intention = self._get_intention_by_key("default")

        return f"[{self.nombre}]: {matched_intention.get('respuesta')}", matched_intention

# --- Subclases de Expertos ---

class GestorCUD(Experto):
    def __init__(self): super().__init__("GestorCUD")
    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        base_resp, intent = super().generate_recommendation(prompt, **kwargs)
        if intent.get("pregunta_clave") == "default" and kwargs.get("is_proactive_call"):
            intent = intent.copy()
            intent.update({"pregunta_clave": "accion_cud_proactiva", "tags": ["cud", "legal"]})
            return ("[Acción CUD]: Detecté que no tienes CUD. Es vital para acceder a derechos (Ley 24.901).", intent)
        return base_resp, intent

class TutorCarrera(Experto):
    def __init__(self): 
        super().__init__("TutorCarrera")
        self.similarity_threshold = 0.55
    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        resp, intent = super().generate_recommendation(prompt, **kwargs)
        prof = kwargs.get('original_profile')
        if prof is not None and prof.get('TIENE_CUD') == 'Si_Tiene_CUD' and "cud" not in intent.get("tags", []):
            resp += "\n [Tip Legal]: Con tu CUD, recuerda el cupo laboral del 4% en el Estado (Ley 22.431)."
        return resp, intent

class TutorInteraccion(Experto):
    def __init__(self): super().__init__("TutorInteraccion")
class TutorCompetencias(Experto):
    def __init__(self): super().__init__("TutorCompetencias")
class TutorBienestar(Experto):
    def __init__(self): super().__init__("TutorBienestar")
class TutorApoyos(Experto):
    def __init__(self): super().__init__("TutorApoyos")
class TutorPrimerEmpleo(Experto):
    def __init__(self): super().__init__("TutorPrimerEmpleo")

# -----------------------------------------------------------------
# --- 5. SISTEMA DE ORQUESTACIÓN MoE (Neuro-Simbólico) ---
# -----------------------------------------------------------------

EXPERT_MAP = {
    'Com_Desafiado': TutorInteraccion(),
    'Nav_Informal': TutorCompetencias(),
    'Prof_Subutil': TutorCarrera(),
    'Potencial_Latente': TutorBienestar(),
    'Cand_Nec_Sig': TutorApoyos(),
    'Joven_Transicion': TutorPrimerEmpleo()
}
CUD_EXPERT = GestorCUD()

class MoESystem:
    """
    Sistema MoE que orquesta la predicción cognitiva, aplica Guardrails Neuro-Simbólicos
    y modula la respuesta afectiva.
    """
    def __init__(self, cognitive_model: Any, feature_columns: List[str],
                 affective_rules: Dict, thresholds: Dict):
        self.cognitive_model = cognitive_model
        self.feature_columns = list(feature_columns) if feature_columns else []
        self.expert_map = EXPERT_MAP
        self.cud_expert = CUD_EXPERT
        self.affective_rules = affective_rules or {}
        self.thresholds = thresholds.get('affective_engine', thresholds or {})
        self.affective_congruence_log: List[Dict[str, str]] = []

        print("› Inicializando modelo semántico y expertos...")
        get_semantic_model()
        for expert in list(self.expert_map.values()) + [self.cud_expert]:
            if isinstance(expert, Experto): expert._initialize_knowledge_base()

    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated = base_weights.copy()
        for emo, prob in emotion_probs.items():
            emo_norm = str(emo).strip().capitalize()
            if prob > min_prob and emo_norm in self.affective_rules:
                for arch, factor in self.affective_rules[emo_norm].items():
                    if arch in modulated and isinstance(factor, (int, float)):
                        try: modulated[arch] *= factor ** prob
                        except: pass
        return modulated

    def _apply_conversational_modulation(self, weights: Dict, context: Dict, neg_emotions: List[str]) -> Dict:
        modulated = weights.copy()
        traj = context.get("emotional_trajectory", [])[-2:]
        if len(traj) == 2 and all(e in neg_emotions for e in traj):
            if 'Potencial_Latente' in modulated:
                logger.info("Racha negativa: Potenciando Tutor Bienestar.")
                modulated['Potencial_Latente'] *= 1.5
        return modulated

    def _normalize_weights(self, weights: Dict) -> Dict:
        valid = {k: v for k, v in weights.items() if isinstance(v, (int, float)) and v >= 0}
        total = sum(valid.values())
        if total > 0:
            return {k: valid[k] / total for k in valid}
        return {k: 0.0 for k in weights}

    def _get_top_emotion(self, probs: Dict, labels: List[str]) -> str:
        if not probs: return "Neutral"
        try:
            top_raw = max(probs, key=probs.get)
            label_map = {l.strip().lower(): l for l in labels}
            norm = label_map.get(top_raw.strip().lower(), top_raw.capitalize())
            return norm if norm in labels + ["Neutral"] else (labels[0] if labels else "Desconocida")
        except: return "Neutral"

    def _log_gating_event(self, tutor: str, intent: Dict, emo_det: str, labels: List[str]) -> Dict:
        emo_esp = intent.get("contexto_emocional_esperado", "neutral").capitalize()
        valid = labels + ["Neutral"]
        emo_esp = emo_esp if emo_esp in valid else "Neutral"
        
        tipo = "Congruente" if emo_det == emo_esp else ("Neutral" if "Neutral" in (emo_det, emo_esp) else "Incongruente")
        entry = {
            "tutor": tutor, "intencion": intent.get("pregunta_clave"),
            "emo_detectada": emo_det, "emo_esperada": emo_esp, "tipo": tipo
        }
        self.affective_congruence_log.append(entry)
        return entry

    def get_cognitive_plan(self, user_profile: pd.Series, emotion_probs: Dict, 
                           conversation_context: Dict, config: Dict, user_prompt: str) -> Tuple[str, str, Dict[str, float]]:
        """
        Orquesta el plan. Integra GUARDRAILS NEURO-SIMBÓLICOS en la predicción.
        """
        constants = config.get('constants', {})
        official_labels = constants.get('emotion_labels', [])
        
        # --- 1. Predicción Cognitiva con Guardrails ---
        predicted_archetype = list(self.expert_map.keys())[0] # Default
        
        try:
            # Limpieza defensiva
            user_profile.index = user_profile.index.astype(str).str.strip()
            profile_for_pred = user_profile.reindex(self.feature_columns).fillna(0.0)
            profile_df = pd.DataFrame([profile_for_pred])

            # A. Inferencia Base (o Probabilística si es posible)
            raw_prediction = None
            ranked_predictions = []
            
            if hasattr(self.cognitive_model, "predict_proba") and hasattr(self.cognitive_model, "classes_"):
                proba = self.cognitive_model.predict_proba(profile_df)[0]
                classes = [str(c) for c in self.cognitive_model.classes_]
                ranked_predictions = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)
                raw_prediction = ranked_predictions[0][0]
            else:
                raw_prediction = self.cognitive_model.predict(profile_df)[0]
                ranked_predictions = [(raw_prediction, 1.0)]

            # B. Aplicación de Guardrails (Veto Simbólico)
            has_uni_title = _get_university_flag(user_profile)
            forbidden = set()
            
            # Regla R1: Universitario NO puede ser Joven_Transicion
            if has_uni_title:
                forbidden.add("Joven_Transicion")

            # Selección del mejor arquetipo permitido
            predicted_archetype = raw_prediction
            if has_uni_title and raw_prediction in forbidden:
                logger.warning(f"🛡️ GUARDRAIL: Veto aplicado a '{raw_prediction}' (Usuario Universitario).")
                # Buscar el siguiente mejor en el ranking
                found_next = False
                for cls, _ in ranked_predictions:
                    if cls not in forbidden:
                        predicted_archetype = cls
                        found_next = True
                        break
                # Fallback de seguridad
                if not found_next:
                    predicted_archetype = "Potencial_Latente"
            
        except Exception as e:
            logger.error(f"Error en predicción cognitiva: {e}. Usando default.")

        # --- 2. Pesos Base ---
        base_weights = {arch: (1.0 if arch == predicted_archetype else 0.0) for arch in self.expert_map}

        # --- 3. Modulación Afectiva ---
        affective_weights = self._apply_affective_modulation(base_weights, emotion_probs)

        # --- 4. Modulación Contextual ---
        conversational_weights = self._apply_conversational_modulation(
            affective_weights, conversation_context, constants.get('negative_emotions', [])
        )

        # --- 5. Normalización Final ---
        final_weights = self._normalize_weights(conversational_weights)

        # --- 6. Construcción del Plan ---
        sorted_plan = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
        final_recs = []
        top_emotion = self._get_top_emotion(emotion_probs, official_labels)

        # Acción CUD
        has_cud = user_profile.get('TIENE_CUD') == 'Si_Tiene_CUD'
        if not has_cud:
            rec_cud, intent_cud = self.cud_expert.generate_recommendation(user_prompt, original_profile=user_profile, is_proactive_call=True)
            final_recs.append(rec_cud)
            self._log_gating_event("GestorCUD", intent_cud, top_emotion, official_labels)

        # Recomendaciones de Expertos
        final_recs.append("**[Plan de Acción Adaptativo]**")
        added = 0
        min_weight = self.thresholds.get('min_recommendation_weight', 0.15)

        for arch, w in sorted_plan:
            if w > min_weight and arch in self.expert_map:
                expert = self.expert_map[arch]
                rec_str, intent = expert.generate_recommendation(user_prompt, original_profile=user_profile)
                
                # Evitar redundancia CUD
                if not ("cud" in intent.get("tags", []) and not has_cud):
                    final_recs.append(f" - ({w:.0%}) {rec_str}")
                    added += 1
                    if added == 1: # Loguear solo el experto principal
                        self._log_gating_event(expert.nombre, intent, top_emotion, official_labels)

        # Default si no hay recomendaciones
        if added == 0:
            def_exp = self.expert_map.get(predicted_archetype, list(self.expert_map.values())[0])
            _, def_intent = def_exp.generate_recommendation("default", original_profile=user_profile)
            final_recs.append(f" - {def_exp.nombre}: {def_intent.get('respuesta')}")

        return "\n".join(final_recs), predicted_archetype, final_weights

if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    print("Módulo cognitive_tutor cargado. Ejecuta main.py para iniciar el sistema.")
