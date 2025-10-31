

import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
from importlib import import_module
import logging

# Configuración básica del logging para trazabilidad del Gating Afectivo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. IMPORTACIONES PARA LA BÚSQUEDA SEMÁNTICA ---
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    import torch.nn.functional as F  # Importar para normalización L2
    SEMANTIC_SEARCH_ENABLED = True
except ImportError:
    warnings.warn(
        "`sentence-transformers` no está instalado. "
        "El tutor recurrirá a la búsqueda simple por palabras clave."
    )
    SEMANTIC_SEARCH_ENABLED = False
    # Definir Clases Dummy si falta la librería para evitar errores de importación
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


# --- 2. GESTOR DEL MODELO SEMÁNTICO (SINGLETON) ---

_SEMANTIC_MODEL = None
# --- CORRECCIÓN SEMÁNTICA ---
# Revertido a un modelo más estándar y fiable para evitar "alucinaciones"
_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def get_semantic_model() -> SentenceTransformer | None:
    """
    Carga y devuelve el modelo de SentenceTransformer como un singleton.
    """
    global _SEMANTIC_MODEL
    if not SEMANTIC_SEARCH_ENABLED:
        return None

    if _SEMANTIC_MODEL is None:
        print(f"› Cargando modelo semántico: {_MODEL_NAME} (esto ocurre solo una vez)...")
        try:
            _SEMANTIC_MODEL = SentenceTransformer(_MODEL_NAME)
            print("› ✅ Modelo semántico cargado exitosamente.")
        except Exception as e:
            warnings.warn(
                f"Error al cargar el modelo semántico {_MODEL_NAME}: {e}. "
                "Se desactivará la búsqueda semántica."
            )
            _SEMANTIC_MODEL = None

    return _SEMANTIC_MODEL

# --- 3. ARQUITECTURA DE EXPERTOS CONVERSACIONALES (SEMÁNTICA) ---

class Experto:
    """
    Clase base abstracta para los tutores expertos del sistema MoE.
    """
    def __init__(self, nombre_experto: str):
        """
        Inicializa el experto, cargando su nombre y KB.
        """
        self.nombre = nombre_experto
        self.knowledge_base: List[Dict[str, Any]] = []
        self.kb_keys: List[str] = []
        self.kb_embeddings = None
        self.similarity_threshold = 0.40 # Umbral base

        self._load_kb()

    def _load_kb(self):
        """
        Carga la Base de Conocimiento (KB) dinámicamente desde `src.expert_kb`.
        """
        try:
            kb_module = import_module("src.expert_kb")
            self.knowledge_base = kb_module.EXPERT_KB.get(self.nombre, [])
            if not self.knowledge_base:
                warnings.warn(f"KB no encontrada o vacía para el experto: {self.nombre}")
        except ImportError:
            warnings.warn("No se pudo importar 'src.expert_kb'. La KB estará vacía.")
        except Exception as e:
            warnings.warn(f"Error cargando KB para {self.nombre}: {e}")

    def _initialize_knowledge_base(self):
        """
        Pre-calcula los embeddings de *centroide* normalizados para cada intención.
        """
        model = get_semantic_model()
        if not model or not self.knowledge_base:
            self.kb_keys = []
            self.kb_embeddings = None
            if not model: warnings.warn(f"Modelo semántico no disponible para {self.nombre}.")
            return

        new_kb_keys = []
        all_centroids = []
        try:
            for item in self.knowledge_base:
                key = item.get("pregunta_clave")
                if not key or key == "default": continue

                variantes = item.get("variantes", [])
                corpus_texts = [str(t) for t in variantes if t and isinstance(t, str)]

                if not corpus_texts:
                    warnings.warn(f"Intención '{key}' en {self.nombre} sin 'variantes' válidas. Omitiendo.")
                    continue

                corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

                if corpus_embeddings.shape[0] > 0:
                    centroid = torch.mean(corpus_embeddings, dim=0)
                    centroid_normalized = F.normalize(centroid, p=2, dim=0)
                    all_centroids.append(centroid_normalized)
                    new_kb_keys.append(key)
                else:
                    warnings.warn(f"Intención '{key}' en {self.nombre} generó embeddings vacíos.")

            self.kb_keys = new_kb_keys
            if all_centroids:
                self.kb_embeddings = torch.stack(all_centroids)
            else:
                self.kb_embeddings = None
                warnings.warn(f"No se generaron centroides para {self.nombre}.")
        except Exception as e:
            warnings.warn(f"Error al codificar KB (centroides) para {self.nombre}: {e}.")
            self.kb_embeddings = None
            self.kb_keys = []

    def _get_intention_by_key(self, key: str) -> Dict[str, Any]:
        """
        Busca y devuelve el diccionario completo de una intención por su 'pregunta_clave'.
        """
        for item in self.knowledge_base:
            if item.get("pregunta_clave") == key:
                return item
        # Fallback genérico
        return {
            "pregunta_clave": "default",
            "respuesta": f"Analicemos tu situación general relacionada con {self.nombre.lower()}.",
            "contexto_emocional_esperado": "neutral",
            "tags": [],
            "variantes": []
        }

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], float]:
        """
        Genera la recomendación buscando la intención más relevante al prompt.
        (MODIFICADO: Devuelve el score de similitud).
        """
        model = get_semantic_model()
        matched_intention = None
        best_match_score = -1.0 # Inicializar score

        # --- Intento de Búsqueda Semántica ---
        if model and self.kb_embeddings is not None and len(self.kb_keys) > 0:
            try:
                prompt_embedding_raw = model.encode(prompt, convert_to_tensor=True)
                if prompt_embedding_raw is None or prompt_embedding_raw.nelement() == 0:
                        raise ValueError("Embedding del prompt resultó vacío.")

                prompt_embedding = F.normalize(prompt_embedding_raw, p=2, dim=0)
                kb_embeds_device = self.kb_embeddings.to(prompt_embedding.device)
                cos_scores = util.cos_sim(prompt_embedding, kb_embeds_device)[0]

                best_match_idx = torch.argmax(cos_scores).item()
                best_match_score = cos_scores[best_match_idx].item()

                if best_match_score > self.similarity_threshold:
                    best_match_key = self.kb_keys[best_match_idx]
                    matched_intention = self._get_intention_by_key(best_match_key)
            except Exception as e:
                warnings.warn(f"Error búsqueda semántica {self.nombre}: {e}. Intentando fallback.")
                best_match_score = -1.0

        # --- Fallback: Búsqueda por Palabra Clave (si semántica falló o score bajo) ---
        if matched_intention is None:
            prompt_lower = prompt.lower()
            for item in self.knowledge_base:
                key = item.get("pregunta_clave", "").lower()
                variantes_lower = [str(v).lower() for v in item.get("variantes", [])]
                if key in prompt_lower or any(var in prompt_lower for var in variantes_lower if len(var) > 3):
                    matched_intention = item
                    best_match_score = 0.5 # Score artificial para keyword match
                    break

        # --- Último Recurso: Respuesta Default ---
        if matched_intention is None:
            matched_intention = self._get_intention_by_key("default")
            # El score se mantiene (será bajo o -1.0)

        response_str = f"[{self.nombre}]: {matched_intention.get('respuesta', 'No encontré una respuesta específica.')}"
        return response_str, matched_intention, best_match_score


# --- Implementaciones concretas de los Expertos ---

class GestorCUD(Experto):
    def __init__(self):
        super().__init__("GestorCUD")

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], float]:
        """
        Genera recomendación CUD, con lógica proactiva.
        """
        base_response, matched_intention, best_match_score = super().generate_recommendation(prompt, **kwargs)
        is_proactive_call = kwargs.get("is_proactive_call", False)

        if matched_intention.get("pregunta_clave") == "default" and is_proactive_call:
            proactive_response = (
                "[Acción CUD]: He detectado que no posees el CUD. Se recomienda iniciar el trámite.\n"
                "  › **Qué es:** El CUD (Ley 22.431) es un documento gratuito que da acceso a derechos "
                "clave en salud (Ley 24.901) y transporte.\n"
                "  › **Consulta:** Puedes preguntarme '¿Qué beneficios tengo con el CUD?' o '¿Dónde se tramita?'."
            )
            proactive_intention = matched_intention.copy()
            proactive_intention["respuesta"] = proactive_response
            proactive_intention["pregunta_clave"] = "accion_cud_proactiva"
            proactive_intention["tags"] = ["cud", "legal", "proactivo"]
            return proactive_response, proactive_intention, 0.0 # Score 0.0 para proactivo

        return base_response, matched_intention, best_match_score

class TutorCarrera(Experto):
    def __init__(self):
        super().__init__("TutorCarrera")
        self.similarity_threshold = 0.55

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], float]:
        """
        Genera recomendación de carrera, añadiendo consejo sobre cupo CUD.
        """
        base_response, matched_intention, best_match_score = super().generate_recommendation(prompt, **kwargs)
        original_profile = kwargs.get('original_profile')
        if original_profile is not None and original_profile.get('TIENE_CUD') == 'Si_Tiene_CUD':
            if "cud" not in matched_intention.get("tags", []):
                base_response += (
                    "\n  [Consejo Legal Adicional]: Dado que posees el CUD, recuerda que postular a "
                    "concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)."
                )
        return base_response, matched_intention, best_match_score

class TutorInteraccion(Experto):
    def __init__(self):
        super().__init__("TutorInteraccion")

class TutorCompetencias(Experto):
    def __init__(self):
        super().__init__("TutorCompetencias")

class TutorBienestar(Experto):
    def __init__(self):
        super().__init__("TutorBienestar")

class TutorApoyos(Experto):
    def __init__(self):
        super().__init__("TutorApoyos")

class TutorPrimerEmpleo(Experto):
    def __init__(self):
        super().__init__("TutorPrimerEmpleo")


# -----------------------------------------------------------------
# --- 4. SISTEMA DE ORQUESTACIÓN MoE ---
# -----------------------------------------------------------------

# Mapeo central de Arquetipos a instancias de Tutores Expertos
EXPERT_MAP = {
    'Com_Desafiado': TutorInteraccion(),
    'Nav_Informal': TutorCompetencias(),
    'Prof_Subutil': TutorCarrera(),
    'Potencial_Latente': TutorBienestar(),
    'Cand_Nec_Sig': TutorApoyos(),
    'Joven_Transicion': TutorPrimerEmpleo()
}

# Instancia separada del Experto CUD
CUD_EXPERT = GestorCUD()

class MoESystem:
    """
    Sistema de Mezcla de Expertos (MoE) que orquesta la respuesta del tutor.
    """
    def __init__(self, cognitive_model: Any, feature_columns: List[str],
                 affective_rules: Dict, thresholds: Dict):
        """
        Inicializa el MoESystem y pre-calcula embeddings de todos los expertos.
        """
        self.cognitive_model = cognitive_model
        self.feature_columns = list(feature_columns) if feature_columns is not None else []
        self.expert_map = EXPERT_MAP
        self.cud_expert = CUD_EXPERT
        self.affective_rules = affective_rules or {}
        self.thresholds = thresholds.get('affective_engine', thresholds or {})
        self.affective_congruence_log: List[Dict[str, str]] = []

        # Inicializar modelo semántico y embeddings de TODOS los expertos
        print("› Inicializando modelo semántico y embeddings de expertos...")
        get_semantic_model() # Carga el modelo si no está cargado
        all_experts = list(self.expert_map.values()) + [self.cud_expert]
        if all(isinstance(e, Experto) for e in all_experts):
             for expert in all_experts:
                expert._initialize_knowledge_base()
             print("› Embeddings de expertos listos.")
        else:
            warnings.warn("Error: No todos los elementos en expert_map/cud_expert son instancias de Experto.")


    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        """
        Modula los pesos base usando reglas afectivas y probabilidad emocional.
        (CORREGIDO: BUG 1 - Ahora "despierta" expertos con peso 0.0).
        """
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated_weights = base_weights.copy()

        for emotion, prob in emotion_probs.items():
            emotion_norm = str(emotion).strip().capitalize()
            if prob > min_prob and emotion_norm in self.affective_rules:
                rules_for_emotion = self.affective_rules[emotion_norm]
                if isinstance(rules_for_emotion, dict):
                    for archetype, factor in rules_for_emotion.items():
                        if archetype in modulated_weights and isinstance(factor, (int, float)) and factor > 0:
                            try:
                                # Obtener el peso base (será 1.0 o 0.0)
                                base_weight = modulated_weights.get(archetype, 0.0)
                                
                                if base_weight > 0:
                                    # --- CASO 1: Modular al experto principal ---
                                    modulated_weights[archetype] *= (factor ** prob)
                                else:
                                    # --- CASO 2: "Despertar" a un experto secundario ---
                                    # (BUG 1 Corregido)
                                    boost = (factor - 1.0) * prob
                                    modulated_weights[archetype] = base_weight + boost

                            except (OverflowError, ValueError):
                                warnings.warn(f"Cálculo inválido en modulación: {factor} ** {prob}")
                        elif archetype not in modulated_weights:
                            pass
                        else:
                            warnings.warn(f"Factor inválido en regla afectiva: {emotion_norm} -> {archetype}: {factor}")
                else:
                    warnings.warn(f"Regla afectiva mal formada para '{emotion_norm}': No es un diccionario.")

        return modulated_weights


    def _apply_conversational_modulation(self, weights: Dict, context: Dict,
                                         negative_emotions: List[str]) -> Dict:
        """
        Ajusta pesos basado en historial conversacional (ej. racha negativa).
        """
        modulated_weights = weights.copy()
        emotional_trajectory = context.get("emotional_trajectory", [])[-2:] # Últimas 2

        # Potenciar Bienestar ante racha negativa
        if len(emotional_trajectory) == 2 and all(e in negative_emotions for e in emotional_trajectory):
            if 'Potencial_Latente' in modulated_weights:
                boost_factor = 1.5
                logging.info(f"Racha negativa detectada. Potenciando Tutor Bienestar x{boost_factor}.")
                modulated_weights['Potencial_Latente'] *= boost_factor

        return modulated_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """
        Normaliza un diccionario de pesos para que sumen 1.0.
        """
        valid_weights = {k: v for k, v in weights.items() if isinstance(v, (int, float)) and v >= 0}
        total_weight = sum(valid_weights.values())

        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
            all_keys = set(weights.keys()) | set(valid_weights.keys())
            return {k: normalized_weights.get(k, 0.0) for k in all_keys}
        
        return {k: 0.0 for k in weights.keys()}


    def _get_top_emotion(self, emotion_probs: Dict, official_labels: List[str]) -> str:
        """
        Extrae la emoción dominante normalizada según la lista oficial.
        """
        if not emotion_probs or not isinstance(emotion_probs, dict):
            return "Neutral"

        try:
            # Esta es la lógica simple (de app-6.py) que funcionaba
            top_emotion_raw = max(emotion_probs, key=emotion_probs.get)
            top_emotion_normalized = top_emotion_raw.strip().lower().capitalize()
            
            # Validación final contra las etiquetas oficiales + Neutral
            valid_labels = official_labels + ["Neutral", "Desconocida"]
            if top_emotion_normalized not in valid_labels or "Etiqueta_" in top_emotion_normalized:
                 return "Desconocida"

            return top_emotion_normalized

        except (ValueError, AttributeError, KeyError):
            logging.warning(f"Error procesando emotion_probs: {emotion_probs}. Devolviendo Desconocida.")
            return "Desconocida"


    def _log_gating_event(self, tutor_name: str, intention: Dict, detected_emotion: str,
                          official_labels: List[str]) -> Dict[str, str]:
        """
        Calcula congruencia y registra evento de Gating Afectivo.
        """
        emo_esperada_raw = intention.get("contexto_emocional_esperado", "neutral").strip().capitalize()
        valid_labels = official_labels + ["Neutral", "Desconocida"]
        emo_esperada = emo_esperada_raw if emo_esperada_raw in valid_labels else "Neutral"

        if detected_emotion == "Desconocida" or detected_emotion == "Neutral" or emo_esperada == "Neutral":
            tipo_congruencia = "Neutral"
        elif detected_emotion == emo_esperada:
            tipo_congruencia = "Congruente"
        else:
            tipo_congruencia = "Incongruente"

        gating_entry = {
            "tutor": tutor_name,
            "intencion_mapeada": intention.get("pregunta_clave", "default"),
            "emo_detectada": detected_emotion,
            "emo_esperada": emo_esperada,
            "tipo": tipo_congruencia
        }
        self.affective_congruence_log.append(gating_entry)
        logging.info(f"GATING AFECTIVO ({tutor_name}): {gating_entry}")
        return gating_entry

    def get_cognitive_plan(
        self,
        user_profile: pd.Series,
        emotion_probs: Dict,
        conversation_context: Dict,
        config: Dict,
        user_prompt: str
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Orquesta la generación completa del plan de acción adaptativo.
        (CORREGIDO: Implementa Gating Semántico, CUD Proactivo y Modulación Afectiva).
        """
        # --- 0. Extracción de Constantes y Setup ---
        constants = config.get('constants', {})
        official_labels = constants.get('emotion_labels', [])
        negative_emotions = constants.get('negative_emotions', [])
        
        final_recs = []
        gating_log_for_ui = {}
        top_detected_emotion = self._get_top_emotion(emotion_probs, official_labels)
        has_cud = user_profile.get('TIENE_CUD') == 'Si_Tiene_CUD'

        # --- 1. GATING DE INTENCIÓN EXPLÍCITA (CUD Reactivo) ---
        _rec_cud_str, rec_cud_int, cud_react_score = self.cud_expert.generate_recommendation(
            prompt=user_prompt, original_profile=user_profile, is_proactive_call=False
        )
        is_cud_query = rec_cud_int.get("pregunta_clave") != "default"

        if is_cud_query:
            # --- RUTA A: Consulta Reactiva de CUD ---
            final_recs.append(_rec_cud_str)
            gating_cud = self._log_gating_event("GestorCUD (Reactivo)", rec_cud_int, top_detected_emotion, official_labels)
            gating_log_for_ui["cud_expert"] = gating_cud
            
            predicted_archetype = "N/A (Consulta CUD)"
            final_weights = {k: 0.0 for k in self.expert_map.keys()}
        
        else:
            # --- RUTA B: Consulta de Arquetipo (Lógica MoE) ---
            
            # --- 2. Predicción Cognitiva (Perfil Base) ---
            try:
                 profile_for_prediction = user_profile.reindex(self.feature_columns).fillna(0.0)
                 profile_df = pd.DataFrame([profile_for_prediction])
                 predicted_archetype = self.cognitive_model.predict(profile_df)[0]
            except Exception as e:
                 logging.error(f"Error prediciendo arquetipo: {e}. Usando default.")
                 predicted_archetype = list(self.expert_map.keys())[0]

            # --- 3. Gating Semántico de Anulación (BUG 2) ---
            semantic_override_threshold = self.thresholds.get('semantic_override_threshold', 0.75)
            best_semantic_match = {'score': -1.0, 'archetype': None, 'intention': None, 'response': ''}

            for archetype_name, expert in self.expert_map.items():
                rec_str, intention, score = expert.generate_recommendation(user_prompt, original_profile=user_profile)
                if score > best_semantic_match['score'] and intention.get("pregunta_clave") != "default":
                    best_semantic_match = {'score': score, 'archetype': archetype_name, 'intention': intention, 'response': rec_str}

            # Aplicar la anulación si se cumple la condición
            if (best_semantic_match['score'] > semantic_override_threshold and
                best_semantic_match['archetype'] is not None and
                best_semantic_match['archetype'] != predicted_archetype):
                
                logging.info(f"ANULACIÓN SEMÁNTICA: Fuerte match ({best_semantic_match['score']:.2f}) "
                             f"con {best_semantic_match['archetype']} anula predicción cognitiva ({predicted_archetype}).")
                predicted_archetype = best_semantic_match['archetype'] # Sobrescribir el arquetipo

            # --- 4. Pesos Base ---
            base_weights = {archetype: 0.0 for archetype in self.expert_map.keys()}
            if predicted_archetype in base_weights:
                base_weights[predicted_archetype] = 1.0

            # --- 5. Modulación Afectiva (BUG 1 Corregido) ---
            affective_weights = self._apply_affective_modulation(base_weights, emotion_probs)

            # --- 6. Modulación Contextual ---
            conversational_weights = self._apply_conversational_modulation(
                affective_weights, conversation_context, negative_emotions
            )

            # --- 7. Normalización Final ---
            final_weights = self._normalize_weights(conversational_weights)

            # --- 8. Construcción del Plan (con CUD Proactivo - BUG 3 Corregido) ---
            
            # (BUG 3 Corregido) Añadir CUD proactivo solo si NO se mencionó "cud"
            if not has_cud and "cud" not in user_prompt.lower():
                rec_str_cud_pro, rec_int_cud_pro, _score_pro = self.cud_expert.generate_recommendation(
                    prompt="default", # Forzar prompt "default"
                    original_profile=user_profile, 
                    is_proactive_call=True
                )
                final_recs.append(rec_str_cud_pro)
                gating_cud = self._log_gating_event("GestorCUD (Proactivo)", rec_int_cud_pro, top_detected_emotion, official_labels)
                gating_log_for_ui["cud_expert"] = gating_cud
            
            # Añadir Recomendaciones de Arquetipo
            final_recs.append("**[Plan de Acción Adaptativo (Arquetipo)]**")
            min_rec_weight = self.thresholds.get('min_recommendation_weight', 0.15)
            recommendations_added = 0

            # Iterar sobre los expertos ordenados por su peso final
            for archetype, weight in sorted(final_weights.items(), key=lambda item: item[1], reverse=True):
                
                if weight > min_rec_weight and archetype in self.expert_map:
                    expert = self.expert_map[archetype]
                    rec_str_arch, rec_int_arch, rec_score_arch = "", None, -1.0
                    
                    # Si este experto fue el ganador de la anulación semántica, usar su intención ya calculada
                    if (archetype == best_semantic_match['archetype'] and 
                        best_semantic_match['score'] > semantic_override_threshold):
                         rec_str_arch = best_semantic_match['response']
                         rec_int_arch = best_semantic_match['intention']
                    else:
                        # Si no, hacer una búsqueda semántica normal (o default)
                         rec_str_arch, rec_int_arch, rec_score_arch = expert.generate_recommendation(
                            prompt=user_prompt, original_profile=user_profile
                        )
                         # Si el match no es bueno, usar el default del experto
                         if rec_score_arch < expert.similarity_threshold:
                             rec_str_arch, rec_int_arch, _ = expert.generate_recommendation("default", original_profile=user_profile)


                    final_recs.append(f"  - ({weight:.0%}) {rec_str_arch}")
                    recommendations_added += 1
                    
                    if recommendations_added == 1: # Loguear solo el principal
                        gating_arch = self._log_gating_event(expert.nombre, rec_int_arch, top_detected_emotion, official_labels)
                        gating_log_for_ui["archetype_expert"] = gating_arch

            # Mensaje Default si no hubo recomendaciones
            if recommendations_added == 0:
                default_expert_name = predicted_archetype
                default_expert = self.expert_map[default_expert_name]
                _def_str, def_int, _def_score = default_expert.generate_recommendation("default", original_profile=user_profile)
                default_response = def_int.get("respuesta", "Analicemos tu situación.")
                final_recs.append(f"  - [{default_expert.nombre} (Default)]: {default_response}")
                
                gating_default = self._log_gating_event(default_expert.nombre + " (Default)", def_int, top_detected_emotion, official_labels)
                gating_log_for_ui["default_expert"] = gating_default
        
        # --- 9. Finalización ---
        plan_string = "\n".join(final_recs)
        return plan_string, predicted_archetype, final_weights

