"""
Módulo de Arquitectura del Tutor Cognitivo y Sistema de Mezcla de Expertos (MoE).

Este módulo define la lógica central del sistema híbrido Neuro-Simbólico desarrollado para la tesis.
Integra modelos de aprendizaje automático para la predicción de arquetipos con reglas simbólicas
de dominio (Guardrails) y un motor de búsqueda semántica para la recuperación de intenciones.

Componentes principales:
------------------------
1.  **Gestor Semántico Singleton:** Administra la carga eficiente y Thread-Safe del modelo BERT.
2.  **Clase Base `Experto`:** Abstracción para los agentes especialistas. Maneja la carga de
    la KB y la búsqueda vectorial optimizada (agnóstica del dispositivo).
3.  **Subclases de Expertos:** Implementaciones específicas para cada dominio de tutoría.
4.  **Sistema `MoESystem`:** Orquestador principal con lógica Neuro-Simbólica y XAI.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
from importlib import import_module
import logging
import datetime
import threading
import re  # Para tokenización robusta

# Configuración del Logger local para trazabilidad del módulo
logger = logging.getLogger(__name__)

# Constantes de Calibración Neuro-Simbólica
UNI_MEMB_THRESHOLD = 0.7  # Umbral de membresía difusa para considerar Título Universitario
MISMATCH_TOLERANCE = 0.05 # Umbral para alertar sobre masa de probabilidad no mapeada
PROBA_SUM_TOLERANCE = 0.01 # Tolerancia para chequear sum(probas) ~ 1.0

# --- 1. IMPORTACIONES PARA LA BÚSQUEDA SEMÁNTICA ---
try:
    from sentence_transformers import SentenceTransformer, util
    from transformers.utils import logging as hf_logging
    import torch
    import torch.nn.functional as F
    
    # Silenciar logs verbosos de Hugging Face para mantener la consola limpia durante demostraciones
    hf_logging.set_verbosity_error()
    
    SEMANTIC_SEARCH_ENABLED = True
except ImportError:
    warnings.warn("`sentence-transformers` no está instalado. Se degradará a búsqueda simple por palabras clave.")
    SEMANTIC_SEARCH_ENABLED = False
    # Definición de Clases Dummy para evitar errores de importación en tiempo de ejecución
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
        @staticmethod
        def no_grad():
            class NoGradContext:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_value, traceback): pass
            return NoGradContext()
        @staticmethod
        def device(dev): return dev
        class cuda:
            @staticmethod
            def is_available(): return False
    class F:
        @staticmethod
        def normalize(*args, **kwargs): pass

# --- 2. FUNCIONES AUXILIARES (Lógica Neuro-Simbólica / XAI) ---

def _truthy(x: Any) -> bool:
    """
    Normaliza valores heterogéneos de entrada a un valor booleano estricto.

    Diseñada para manejar la variabilidad de formatos en datasets provenientes de
    encuestas o archivos CSV/Excel (e.g., 'Si', 'Sí', 'True', 1, 'Presenta').

    Args:
        x (Any): El valor a evaluar (puede ser string, numérico, booleano o nulo).

    Returns:
        bool: `True` si el valor representa una afirmación positiva, `False` en caso contrario
              o si es nulo (NaN).
    """
    if x is None: return False
    if pd.isna(x): return False
    if isinstance(x, (bool, np.bool_)): return bool(x)
    s = str(x).strip().lower()
    return s in {"si", "sí", "true", "1", "t", "y", "yes", "presenta", "tiene", "s"}

def _get_university_flag(user_profile: pd.Series, threshold: float = UNI_MEMB_THRESHOLD) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Implementa una heurística robusta para detectar la posesión de un título universitario.

    Esta función actúa como un sensor simbólico dentro de la arquitectura. Prioriza la
    evidencia explícita ("variables duras") sobre la inferencia difusa, proporcionando
    trazabilidad (XAI) sobre el origen de la decisión.

    Args:
        user_profile (pd.Series): Serie de pandas con los atributos del usuario.
        threshold (float): Valor mínimo de membresía difusa para considerar positivo.

    Returns:
        Tuple[bool, Optional[Dict[str, Any]]]: 
            - bool: `True` si se detecta título universitario.
            - dict: Metadatos de evidencia (fuente, valor, tipo) para explicabilidad, 
                    o `None` si no se detecta.
    """
    # Prioridad de evaluación: Variables Duras > Variables Difusas
    candidate_cols = [
        "TIENE_TITULO_UNI", "TIENE_TITULO_UNIVERSITARIO", "TITULO_UNIVERSITARIO",
        "POSEE_TITULO_UNIVERSITARIO", "UNIVERSITARIO", "NIVEL_UNIVERSITARIO",
        "MNEA",          # Variable dura: Nivel educativo numérico (5 = Universitario en ENDIS)
        "CH_Alto_memb"   # Fallback: Variable difusa (Membresía Capital Humano Alto)
    ]
    
    for col in candidate_cols:
        if col in user_profile.index:
            val = user_profile[col]
            if pd.isna(val): continue

            # Caso 1: Nivel Educativo Numérico (MNEA)
            if col == 'MNEA':
                try:
                    val_int = int(float(val))
                    if val_int == 5:
                        return True, {"source": col, "value": val_int, "type": "hard_evidence"}
                except (ValueError, TypeError): continue 
            
            # Caso 2: Membresía Difusa (Lógica Fuzzy)
            elif col == 'CH_Alto_memb':
                try:
                    val_float = float(val)
                    if val_float >= threshold:
                        return True, {"source": col, "value": round(val_float, 4), "type": "fuzzy_inference"}
                except (ValueError, TypeError): continue

            # Caso 3: Booleanos/Strings Explícitos
            elif _truthy(val):
                return True, {"source": col, "value": str(val), "type": "explicit_flag"}

    # Fallback heurístico: Búsqueda semántica en los nombres de las columnas
    for col in user_profile.index:
        c = str(col).lower()
        if "titulo" in c and ("uni" in c or "univers" in c):
            if not pd.isna(user_profile[col]) and _truthy(user_profile[col]):
                return True, {"source": col, "value": str(user_profile[col]), "type": "heuristic_match"}
                
    return False, None

# --- 3. GESTOR DEL MODELO SEMÁNTICO (SINGLETON & THREAD-SAFE) ---

_SEMANTIC_MODEL = None
_MODEL_LOCK = threading.Lock() # Lock para concurrencia segura
_MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es'

def get_semantic_model() -> Optional[SentenceTransformer]:
    """
    Obtiene la instancia única (Singleton) del modelo de embeddings semánticos.

    Implementa el patrón Singleton para evitar la recarga costosa del modelo BERT
    en memoria cada vez que se consulta a un experto.

    Returns:
        Optional[SentenceTransformer]: La instancia del modelo cargado, o `None` si 
                                       la librería no está disponible o falla la carga.
    """
    global _SEMANTIC_MODEL
    if not SEMANTIC_SEARCH_ENABLED: return None
    
    # Patrón Double-Check Locking para eficiencia
    if _SEMANTIC_MODEL is not None:
        return _SEMANTIC_MODEL

    with _MODEL_LOCK:
        if _SEMANTIC_MODEL is None:
            logger.info(f"› Cargando modelo semántico: {_MODEL_NAME}...")
            try:
                _SEMANTIC_MODEL = SentenceTransformer(_MODEL_NAME)
                logger.info("› ✅ Modelo semántico cargado exitosamente.")
            except Exception as e:
                warnings.warn(f"Error cargando modelo semántico: {e}. Se desactivará la búsqueda vectorial.")
                _SEMANTIC_MODEL = None
        return _SEMANTIC_MODEL

# --- 4. ARQUITECTURA DE EXPERTOS CONVERSACIONALES ---

class Experto:
    """
    Clase base abstracta que define el comportamiento de un Tutor Experto.

    Cada experto encapsula una Base de Conocimiento (KB) específica y métodos
    para recuperar información relevante mediante similitud semántica.

    Atributos:
        nombre (str): Identificador único del experto.
        knowledge_base (List[Dict]): Estructura de datos con intenciones y respuestas.
        kb_keys (List[str]): Claves de las intenciones indexadas.
        kb_embeddings (torch.Tensor): Tensores pre-calculados de los centroides de las intenciones.
        similarity_threshold (float): Umbral mínimo de similitud coseno para aceptar un match.
    """
    def __init__(self, nombre_experto: str):
        self.nombre = nombre_experto
        self.knowledge_base: List[Dict[str, Any]] = []
        self.kb_keys: List[str] = []
        self.kb_embeddings = None
        self.similarity_threshold = 0.40
        self._load_kb()

    def _load_kb(self):
        """
        Carga dinámicamente la Base de Conocimiento desde el módulo `src.expert_kb`.
        Permite desacoplar la lógica del experto de los datos.
        """
        try:
            kb_module = import_module("src.expert_kb")
            self.knowledge_base = kb_module.EXPERT_KB.get(self.nombre, [])
            if not self.knowledge_base: warnings.warn(f"KB vacía para el experto: {self.nombre}")
        except Exception as e: warnings.warn(f"Error crítico cargando KB para {self.nombre}: {e}")

    def _initialize_knowledge_base(self):
        """
        Inicializa los índices semánticos en el dispositivo correcto (GPU/CPU).
        Calcula el 'embedding centroide' para cada intención en la KB.
        """
        model = get_semantic_model()
        if not model or not self.knowledge_base:
            self.kb_embeddings = None
            return
        
        new_kb_keys = []
        all_centroids = []
        try:
            # Detección robusta del dispositivo (GPU/CPU) con fallbacks profundos
            try:
                device = model.device
            except AttributeError:
                try:
                    # Fallback para versiones donde device no es atributo directo
                    device = next(model._first_module().parameters()).device
                except Exception:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for item in self.knowledge_base:
                # FIX: Sanitización de claves pero SIN lowercase para almacenamiento
                # (Preserva CamelCase si existe en la KB original)
                key = (item.get("pregunta_clave") or "").strip()
                if not key or key.lower() == "default": continue
                
                variantes = [str(t) for t in item.get("variantes", []) if t]
                if not variantes: continue
                
                # Codificación y cálculo del centroide con optimización de memoria (no_grad)
                with torch.no_grad():
                    # encode devuelve tensores. Si convert_to_tensor=True, ST intenta usar el device del modelo.
                    embeddings = model.encode(variantes, convert_to_tensor=True)
                
                if embeddings.shape[0] > 0:
                    centroid = torch.mean(embeddings, dim=0)
                    all_centroids.append(F.normalize(centroid, p=2, dim=0)) # Normalización L2
                    new_kb_keys.append(key)
            
            self.kb_keys = new_kb_keys
            if all_centroids: 
                # Stackear y asegurar que estén en el dispositivo correcto para inferencia rápida
                self.kb_embeddings = torch.stack(all_centroids).to(device)
            else: 
                self.kb_embeddings = None
        except Exception as e:
            logger.warning(f"Error al generar embeddings para {self.nombre}: {e}")
            self.kb_embeddings = None

    def _get_intention_by_key(self, key: str) -> Dict[str, Any]:
        """Recupera el objeto de intención completo dado su identificador clave."""
        target = key.strip().lower()
        for item in self.knowledge_base:
            current = (item.get("pregunta_clave") or "").strip().lower()
            if current == target: return item
        
        # Retorno de objeto por defecto seguro
        return {
            "pregunta_clave": "default",
            "respuesta": f"Analicemos tu situación específica con el {self.nombre}.",
            "contexto_emocional_esperado": "neutral", "tags": [], "variantes": []
        }

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], str]:
        """
        Genera una recomendación contextualizada basada en la consulta del usuario.

        Utiliza una estrategia de recuperación híbrida:
        1.  **Búsqueda Semántica:** Comparación vectorial coseno contra centroides.
        2.  **Fallback por Palabras Clave:** Búsqueda de tokens robusta si la semántica falla.
        3.  **Respuesta por Defecto:** Si no hay coincidencias relevantes.

        Args:
            prompt (str): Consulta del usuario.
            **kwargs: Parámetros adicionales (e.g., perfil del usuario).

        Returns:
            Tuple[str, Dict, str]: (Texto de respuesta, Metadatos intención, Modo de búsqueda usado).
        """
        model = get_semantic_model()
        matched_intention = None
        search_mode = "default" # Modo por defecto
        
        # 1. Estrategia Principal: Búsqueda Semántica
        if model and self.kb_embeddings is not None and len(self.kb_keys) > 0:
            try:
                # Detección del dispositivo de los embeddings pre-calculados
                target_device = self.kb_embeddings.device
                
                with torch.no_grad():
                    # Codificar prompt. Evitar .to(device) redundante si ya está ahí.
                    raw_emb = model.encode(prompt, convert_to_tensor=True)
                    # FIX: Verificación defensiva antes de mover al dispositivo
                    if hasattr(raw_emb, "device") and raw_emb.device != target_device:
                        raw_emb = raw_emb.to(target_device)
                    
                    prompt_emb = F.normalize(raw_emb, p=2, dim=0)
                
                scores = util.cos_sim(prompt_emb, self.kb_embeddings)[0]
                best_idx = torch.argmax(scores).item()
                if scores[best_idx].item() > self.similarity_threshold:
                    matched_intention = self._get_intention_by_key(self.kb_keys[best_idx])
                    search_mode = "semantic"
            except Exception as e: 
                logger.warning(f"Fallo en búsqueda semántica ({self.nombre}): {e}")

        # 2. Estrategia de Respaldo: Tokens / Palabras Clave (Robusto con Regex)
        if matched_intention is None:
            prompt_lower = prompt.lower()
            # Tokenización que ignora puntuación: "CUD?" -> "cud"
            tokens = set(re.findall(r"\b\w+\b", prompt_lower, flags=re.UNICODE))
            
            # Whitelist de siglas cortas válidas en el dominio
            valid_short_tokens = {'cud', 'ong', 'tea', 'ilp', 'lsa'} 
            
            for item in self.knowledge_base:
                # FIX: Sanitización de claves vacías o nulas
                key = (item.get("pregunta_clave") or "").strip().lower()
                vars_lower = [str(v).lower() for v in item.get("variantes", [])]
                
                # Coincidencia de clave exacta (con boundaries y control de clave "default" o vacía)
                # FIX: No ejecutar regex si key es "default"
                if key and key != "default" and re.search(rf"\b{re.escape(key)}\b", prompt_lower):
                    matched_intention = item
                    search_mode = "tokens"
                    break
                
                # Búsqueda en variantes
                for v in vars_lower:
                    # Tokenizar variante
                    v_tokens = re.findall(r"\b\w+\b", v, flags=re.UNICODE)
                    if not v_tokens: continue
                    
                    match_found = False
                    if len(v_tokens) == 1:
                        token = v_tokens[0]
                        # Filtro de calidad para unigramas: longitud >= 4 o whitelist
                        if len(token) >= 4 or token in valid_short_tokens:
                            if token in tokens: match_found = True
                    else:
                        # Para frases, verificamos que todos los tokens estén presentes (subset)
                        if set(v_tokens).issubset(tokens): match_found = True
                    
                    if match_found:
                        matched_intention = item
                        search_mode = "tokens"
                        break
                
                if matched_intention: break
        
        # 3. Estrategia Final: Default
        if matched_intention is None: 
            matched_intention = self._get_intention_by_key("default")
            search_mode = "default"
            
        return f"[{self.nombre}]: {matched_intention.get('respuesta')}", matched_intention, search_mode

# --- Subclases de Expertos (Implementaciones de Dominio) ---

class GestorCUD(Experto):
    """Especialista en el Certificado Único de Discapacidad (CUD) y trámites legales asociados."""
    def __init__(self): super().__init__("GestorCUD")
    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], str]:
        base_resp, intent, mode = super().generate_recommendation(prompt, **kwargs)
        # Lógica proactiva: Si se detecta ausencia de CUD en el perfil, sugerir tramitación.
        if intent.get("pregunta_clave") == "default" and kwargs.get("is_proactive_call"):
            intent = intent.copy()
            intent.update({"pregunta_clave": "accion_cud_proactiva", "tags": ["cud", "legal"]})
            return ("[Acción CUD]: Detecté que no tienes CUD vigente. Es un documento vital para acceder a derechos (Ley 24.901).", intent, mode)
        return base_resp, intent, mode

class TutorCarrera(Experto):
    """Especialista en orientación vocacional y trayectoria profesional."""
    def __init__(self): 
        super().__init__("TutorCarrera")
        self.similarity_threshold = 0.55 # Umbral más estricto para consejos de carrera
    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], str]:
        resp, intent, mode = super().generate_recommendation(prompt, **kwargs)
        prof = kwargs.get('original_profile')
        # Inyección de consejo legal contextual: Cupo laboral si tiene CUD.
        if prof is not None and prof.get('TIENE_CUD') == 'Si_Tiene_CUD' and "cud" not in intent.get("tags", []):
            resp += "\n [Tip Legal]: Con tu CUD, recuerda el derecho al cupo laboral del 4% en el Estado (Ley 22.431)."
        return resp, intent, mode

class TutorInteraccion(Experto):
    """Especialista en habilidades sociales y comunicación."""
    def __init__(self): super().__init__("TutorInteraccion")

class TutorCompetencias(Experto):
    """Especialista en certificación de competencias y habilidades técnicas."""
    def __init__(self): super().__init__("TutorCompetencias")

class TutorBienestar(Experto):
    """Especialista en calidad de vida, bienestar emocional y manejo del estrés."""
    def __init__(self): super().__init__("TutorBienestar")

class TutorApoyos(Experto):
    """Especialista en tecnologías de asistencia y ajustes razonables."""
    def __init__(self): super().__init__("TutorApoyos")

class TutorPrimerEmpleo(Experto):
    """Especialista en inserción laboral inicial y primer empleo."""
    def __init__(self): super().__init__("TutorPrimerEmpleo")

# -----------------------------------------------------------------
# --- 5. SISTEMA DE ORQUESTACIÓN MoE (Neuro-Simbólico + XAI) ---
# -----------------------------------------------------------------

def _build_expert_map():
    """Factoría para instanciar el mapa de expertos de forma perezosa."""
    return {
        'Com_Desafiado': TutorInteraccion(), 'Nav_Informal': TutorCompetencias(),
        'Prof_Subutil': TutorCarrera(), 'Potencial_Latente': TutorBienestar(),
        'Cand_Nec_Sig': TutorApoyos(), 'Joven_Transicion': TutorPrimerEmpleo()
    }

class MoESystem:
    """
    Sistema de Mezcla de Expertos (MoE) que implementa la lógica Neuro-Simbólica.

    Responsabilidades:
    1.  **Inferencia:** Predice el arquetipo cognitivo usando un modelo ML (Random Forest).
    2.  **Guardrails:** Aplica reglas simbólicas de veto para corregir inconsistencias del modelo ML.
    3.  **Gating Afectivo:** Modula la importancia de los expertos según el estado emocional.
    4.  **Selección:** Construye el plan de acción final priorizando recomendaciones.
    """
    def __init__(self, cognitive_model: Any, feature_columns: List[str],
                 affective_rules: Dict, thresholds: Dict):
        self.cognitive_model = cognitive_model
        self.feature_columns = list(feature_columns) if feature_columns else []
        
        # Inicialización perezosa de expertos para evitar side-effects al importar
        self.expert_map = _build_expert_map()
        self.cud_expert = GestorCUD()
        
        self.affective_rules = affective_rules or {}
        # Inyección de configuración para el umbral difuso
        self.thresholds = thresholds.get('affective_engine', thresholds or {})
        
        self.affective_congruence_log: List[Dict[str, str]] = []
        
        # Metadatos para auditoría del sistema
        self.model_metadata = {
            "loaded": True, 
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        logger.info("› Inicializando modelo semántico y bases de conocimiento de expertos...")
        get_semantic_model()
        # Inicializar KB de todos los expertos registrados
        for expert in list(self.expert_map.values()) + [self.cud_expert]:
            if isinstance(expert, Experto): expert._initialize_knowledge_base()

    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        """Modula los pesos de los expertos basándose en las probabilidades emocionales."""
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated = base_weights.copy()
        
        # Mapeo normalizado de claves de reglas (Case-Insensitive)
        rule_map = {k.strip().lower(): k for k in self.affective_rules.keys()}
        
        for emo, prob in emotion_probs.items():
            emo_key = str(emo).strip().lower()
            if prob > min_prob and emo_key in rule_map:
                real_rule_key = rule_map[emo_key]
                for arch, factor in self.affective_rules[real_rule_key].items():
                    if arch in modulated and isinstance(factor, (int, float)):
                        try: modulated[arch] *= factor ** prob
                        except: pass
        return modulated

    def _apply_conversational_modulation(self, weights: Dict[str, float], context: Dict, neg_emotions: List[str]) -> Dict[str, float]:
        """Ajusta pesos basado en el historial conversacional (e.g., rachas negativas)."""
        modulated = weights.copy()
        traj = (context or {}).get("emotional_trajectory", [])[-2:]

        # Normalización robusta para comparación
        neg_set = {str(e).strip().lower() for e in (neg_emotions or [])}
        traj_norm = [str(e).strip().lower() for e in traj]

        if len(traj_norm) == 2 and all(e in neg_set for e in traj_norm):
            if 'Potencial_Latente' in modulated:
                logger.info("Racha negativa detectada: Potenciando Tutor Bienestar.")
                modulated['Potencial_Latente'] *= 1.5
        return modulated

    def _normalize_weights(self, weights: Dict) -> Dict:
        """Normaliza los pesos para que sumen 1.0."""
        valid = {k: v for k, v in weights.items() if isinstance(v, (int, float)) and v >= 0}
        total = sum(valid.values())
        if total > 0: return {k: valid[k] / total for k in valid}
        return {k: 0.0 for k in weights}

    def _get_top_emotion(self, probs: Dict, labels: List[str]) -> str:
        """Determina la emoción dominante normalizada."""
        if not probs: return "Neutral"
        try:
            top_raw = max(probs, key=probs.get)
            label_map = {l.strip().lower(): l for l in labels}
            norm = label_map.get(top_raw.strip().lower(), top_raw.capitalize())
            return norm if norm in labels + ["Neutral"] else (labels[0] if labels else "Desconocida")
        except: return "Neutral"

    def _log_gating_event(self, tutor: str, intent: Dict, emo_det: str, labels: List[str]) -> Dict:
        """Registra eventos de congruencia afectiva para análisis posterior."""
        emo_esp = intent.get("contexto_emocional_esperado", "neutral").capitalize()
        valid = labels + ["Neutral"]
        emo_esp = emo_esp if emo_esp in valid else "Neutral"
        tipo = "Congruente" if emo_det == emo_esp else ("Neutral" if "Neutral" in (emo_det, emo_esp) else "Incongruente")
        entry = {"tutor": tutor, "intencion": intent.get("pregunta_clave"), "emo_detectada": emo_det, "emo_esperada": emo_esp, "tipo": tipo}
        self.affective_congruence_log.append(entry)
        return entry

    def get_cognitive_plan(self, user_profile: pd.Series, emotion_probs: Dict, 
                           conversation_context: Dict, config: Dict, user_prompt: str) -> Tuple[str, str, Dict[str, float], Dict[str, Any]]:
        """Método principal de orquestación. Genera el plan de acción final."""
        constants = config.get('constants', {})
        official_labels = constants.get('emotion_labels', [])
        
        # --- 1. Predicción Cognitiva (Neuro-Simbólica + XAI) ---
        predicted_archetype = list(self.expert_map.keys())[0] # Default seguro
        
        # Inmutabilidad: Trabajar sobre una copia para evitar side-effects en la sesión
        user_profile_safe = user_profile.copy()
        user_profile_safe.index = user_profile_safe.index.astype(str).str.strip()
        
        # A. Guardrail Simbólico: Detección con Evidencia
        # Obtener umbral desde la configuración inyectada (Fallback 0.7)
        uni_threshold = float(self.thresholds.get('uni_memb_threshold', UNI_MEMB_THRESHOLD))
        has_uni_title, uni_evidence = _get_university_flag(user_profile_safe, threshold=uni_threshold)
        
        forbidden = set()
        guardrail_reason = None
        
        # Regla R1: Si es Universitario, VETAR 'Joven_Transicion'
        if has_uni_title:
            forbidden.add("Joven_Transicion")
            guardrail_reason = {
                "rule_id": "R1_UNI_NOT_TRANSICION",
                "trigger": "is_university=True",
                "evidence": uni_evidence, # <-- XAI: Trazabilidad completa de la causa
                "forbidden": ["Joven_Transicion"]
            }

        # Definición de fallbacks seguros en caso de veto total
        fallback_options = ["Prof_Subutil", "Potencial_Latente", "Com_Desafiado"]
        if hasattr(self.cognitive_model, "classes_"):
            model_classes = set(map(str, self.cognitive_model.classes_))
            fallback_options = [c for c in fallback_options if c in model_classes]
        
        default_cls = "Potencial_Latente"
        if has_uni_title:
            # Buscar el primer fallback disponible que no esté prohibido
            for opt in fallback_options:
                if opt not in forbidden:
                    default_cls = opt
                    break

        # B. Inferencia Neuronal (Random Forest)
        raw_prediction = None
        ranked_predictions = []
        sum_probs = 0.0

        try:
            # Conversión explícita a float para evitar FutureWarnings de Pandas
            profile_for_pred = (
                user_profile_safe.reindex(self.feature_columns)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            profile_df = pd.DataFrame([profile_for_pred])

            # Obtener probabilidades si el modelo lo soporta, para un ranking más rico
            if hasattr(self.cognitive_model, "predict_proba") and hasattr(self.cognitive_model, "classes_"):
                proba = self.cognitive_model.predict_proba(profile_df)[0]
                classes = [str(c) for c in self.cognitive_model.classes_]
                
                # Sanity Check de Probabilidades para auditoría
                ranked_predictions = []
                sum_probs = 0.0 # Reinicio defensivo
                for c, p in zip(classes, proba):
                    if np.isfinite(p) and p >= 0:
                        fp = float(p)
                        ranked_predictions.append((c, fp))
                        sum_probs += fp
                
                ranked_predictions.sort(key=lambda x: x[1], reverse=True)
                raw_prediction = ranked_predictions[0][0] if ranked_predictions else None
                
                # =========================
                # FIX A: Fallback si predict_proba no deja ranking utilizable
                # =========================
                if not ranked_predictions:
                    raw_prediction = str(self.cognitive_model.predict(profile_df)[0])
                    ranked_predictions = [(raw_prediction, 1.0)]
                    sum_probs = 1.0
            else:
                raw_prediction = str(self.cognitive_model.predict(profile_df)[0])
                ranked_predictions = [(raw_prediction, 1.0)]
                sum_probs = 1.0

            # C. Fusión Neuro-Simbólica: Aplicación de Veto
            predicted_archetype = raw_prediction or predicted_archetype
            if has_uni_title and raw_prediction in forbidden:
                logger.warning(f"🛡️ GUARDRAIL: Veto a '{raw_prediction}' por regla {guardrail_reason['rule_id']}. Evidencia: {uni_evidence}.")
                # Buscar la siguiente mejor opción permitida en el ranking probabilístico
                found_next = False
                for cls, _ in ranked_predictions:
                    if cls not in forbidden:
                        predicted_archetype = cls
                        found_next = True
                        break
                if not found_next: predicted_archetype = default_cls
            
        except Exception as e:
            logger.error(f"Error crítico en predicción cognitiva: {e}. Usando default.")
            predicted_archetype = default_cls
            raw_prediction = raw_prediction or "Desconocido"
            # safety net útil: ranking mínimo consistente
            ranked_predictions = [(predicted_archetype, 1.0)]
            sum_probs = 1.0

        # --- 2. Orquestación MoE (Cálculo de Pesos - TRUE MoE) ---
        base_weights = {arch: 0.0 for arch in self.expert_map}
        
        # Métricas de auditoría XAI
        covered_mass_total = 0.0
        vetoed_mass = 0.0
        unmapped_mass = 0.0

        if ranked_predictions:
            for arch, prob in ranked_predictions:
                p = float(prob)
                if arch not in base_weights:
                    unmapped_mass += p
                    continue
                covered_mass_total += p
                if arch in forbidden:
                    base_weights[arch] = 0.0
                    vetoed_mass += p
                else:
                    base_weights[arch] = p
        else:
            # Fallback binario (usa predicted_archetype ya corregido por veto en el paso anterior)
            if predicted_archetype in base_weights and predicted_archetype not in forbidden:
                base_weights[predicted_archetype] = 1.0
                covered_mass_total = 1.0

        effective_mass = max(0.0, covered_mass_total - vetoed_mass)

        # Chequeo de Mismatch para auditoría
        if unmapped_mass > MISMATCH_TOLERANCE:
            logger.warning(f"⚠️ MoE mismatch: unmapped_mass={unmapped_mass:.3f} > tol={MISMATCH_TOLERANCE}")

        affective_weights = self._apply_affective_modulation(base_weights, emotion_probs)
        conversational_weights = self._apply_conversational_modulation(
            affective_weights, conversation_context, constants.get('negative_emotions', [])
        )
        final_weights = self._normalize_weights(conversational_weights)

        # --- 3. Generación de Respuesta (Plan de Acción) ---
        sorted_plan = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
        final_recs = []
        top_emotion = self._get_top_emotion(emotion_probs, official_labels)
        
        cud_search_mode = None
        expert_search_mode = None
        primary_congruence = "N/A" # Nuevo para XAI

        # 3.1. Acción Proactiva CUD (Gestor Legal)
        # Robustez en detección booleana de CUD
        val_cud = user_profile_safe.get("TIENE_CUD")
        has_cud = _truthy(val_cud) or (str(val_cud) == "Si_Tiene_CUD")
        
        if not has_cud:
            rec_cud, intent_cud, mode_cud = self.cud_expert.generate_recommendation(user_prompt, original_profile=user_profile_safe, is_proactive_call=True)
            final_recs.append(rec_cud)
            self._log_gating_event("GestorCUD", intent_cud, top_emotion, official_labels)
            cud_search_mode = f"cud:{mode_cud}"

        # 3.2. Recomendaciones de Expertos Ponderados
        final_recs.append("**[Plan de Acción Adaptativo]**")
        added = 0
        min_weight = self.thresholds.get('min_recommendation_weight', 0.15)

        for arch, w in sorted_plan:
            if w > min_weight and arch in self.expert_map:
                expert = self.expert_map[arch]
                rec_str, intent, mode = expert.generate_recommendation(user_prompt, original_profile=user_profile_safe)
                
                # =========================
                # FIX B: Capturar el primer modo de búsqueda que aparezca
                # =========================
                if expert_search_mode is None:
                    expert_search_mode = mode

                # Evitar redundancia si el consejo es sobre CUD y el usuario ya recibió la alerta proactiva
                if not ("cud" in intent.get("tags", []) and not has_cud):
                    final_recs.append(f" - ({w:.0%}) {rec_str}")
                    added += 1
                    if added == 1: 
                        gating_data = self._log_gating_event(expert.nombre, intent, top_emotion, official_labels)
                        primary_congruence = gating_data["tipo"] # XAI: Congruencia del experto principal

        # 3.3. Fallback por defecto
        if added == 0:
            def_exp = self.expert_map.get(predicted_archetype, list(self.expert_map.values())[0])
            _, def_intent, mode_def = def_exp.generate_recommendation("default", original_profile=user_profile_safe)
            final_recs.append(f" - {def_exp.nombre}: {def_intent.get('respuesta')}")
            if not expert_search_mode:
                expert_search_mode = mode_def
            
            # Loguear evento para fallback si no hubo previos
            gating_data = self._log_gating_event(def_exp.nombre, def_intent, top_emotion, official_labels)
            primary_congruence = gating_data["tipo"]

        # Logging de Auditoría XAI (Visible en consola)
        if guardrail_reason:
            logger.info(f"XAI Log (Justificación de Veto): {guardrail_reason}")

        # 4. Construcción de Metadatos de Salida (XAI Output)
        
        # Calcular flags antes de sanitizar variables (Fix Lógico: verificar raw_prediction existe)
        # FIX A: Lógica estricta para el trigger XAI
        constraint_triggered = bool(has_uni_title and raw_prediction and (raw_prediction in forbidden))
        was_veto_applied = bool(guardrail_reason and constraint_triggered)
        
        # Safety net: asegurar que raw_prediction tenga valor para el log final
        raw_prediction = raw_prediction or "Desconocido"
        
        # Log Estructurado de Ejecución (Para Tesis)
        execution_metrics = {
            "cud_search_mode": cud_search_mode or "N/A",
            "expert_search_mode": expert_search_mode or "N/A", # FIX: Fallback a N/A
            "veto_applied": was_veto_applied,
            "raw_prediction": raw_prediction,
            "selected_archetype": predicted_archetype,
            "top_emotion": top_emotion,
            "affective_congruence": primary_congruence, # XAI Goal: Congruencia expuesta
            "moe_effective_mass": round(effective_mass, 4),
            "moe_vetoed_mass": round(vetoed_mass, 4),
            "moe_unmapped_mass": round(unmapped_mass, 4),
            "has_class_mismatch": unmapped_mass > MISMATCH_TOLERANCE,
            "proba_sum": round(sum_probs, 4),
            "proba_sum_ok": abs(sum_probs - 1.0) < PROBA_SUM_TOLERANCE
        }
        logger.info(f"🛡️ EXECUTION LOG: {execution_metrics}")

        xai_metadata = {
            "raw_prediction": raw_prediction,
            "selected_archetype": predicted_archetype,
            "guardrail_reason": guardrail_reason,
            "ranked_predictions": ranked_predictions[:3] if ranked_predictions else [],
            "xai_flags": {
                "is_constraint_applicable": bool(has_uni_title),
                "is_constraint_triggered": constraint_triggered,
                "was_veto_applied": was_veto_applied,
                "raw_equals_selected": (raw_prediction == predicted_archetype),
                "veto_reason_id": guardrail_reason.get("rule_id") if guardrail_reason else None
            },
            "execution_metrics": execution_metrics
        }

        return "\n".join(final_recs), predicted_archetype, final_weights, xai_metadata

if __name__ == "__main__":
    # Bloque de prueba de carga
    logging.basicConfig(level=logging.INFO)
    logger.info("Módulo cognitive_tutor cargado correctamente.")
