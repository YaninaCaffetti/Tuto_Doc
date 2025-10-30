"""
Define la arquitectura del Tutor Cognitivo y el Sistema de Mezcla de Expertos (MoE).

Este módulo contiene:
- La clase base `Experto` que maneja la carga de la Base de Conocimiento (KB)
  y la búsqueda semántica de intenciones basada en centroides.
- Subclases de `Experto` para cada rol de tutoría específico (ej. TutorCarrera).
- La clase `MoESystem` que orquesta la predicción cognitiva, la modulación
  afectiva/contextual y la selección final de recomendaciones, incluyendo
  la lógica de Gating Afectivo.
- Un gestor singleton para el modelo de embeddings semánticos.
"""

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
        def tensor(*args, **kwargs): pass # Añadir dummy para tensor
    class F:
        @staticmethod
        def normalize(*args, **kwargs): pass


# --- 2. GESTOR DEL MODELO SEMÁNTICO (SINGLETON) ---

_SEMANTIC_MODEL = None
# Modelo optimizado para similitud semántica en español
_MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es'

def get_semantic_model() -> SentenceTransformer | None:
    """
    Carga y devuelve el modelo de SentenceTransformer como un singleton.

    Evita recargar el modelo pesado en memoria repetidamente.
    Si la librería `sentence-transformers` no está disponible o el modelo
    falla al cargar, devuelve None y desactiva la búsqueda semántica.

    Returns:
        Una instancia del modelo SentenceTransformer o None si falla la carga.
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
            _SEMANTIC_MODEL = None # Marcar como fallido para no reintentar

    return _SEMANTIC_MODEL

# --- 3. ARQUITECTURA DE EXPERTOS CONVERSACIONALES (SEMÁNTICA) ---

class Experto:
    """
    Clase base abstracta para los tutores expertos del sistema MoE.

    Cada experto carga dinámicamente su Base de Conocimiento (KB) desde
    `src.expert_kb` y utiliza un modelo semántico para encontrar la
    intención más relevante a la consulta del usuario mediante comparación
    con centroides de embeddings normalizados.

    Attributes:
        nombre (str): Nombre identificador del experto (ej. "TutorCarrera").
        knowledge_base (List[Dict[str, Any]]): KB cargada desde `expert_kb.py`.
        kb_keys (List[str]): Lista de las 'pregunta_clave' (etiquetas) de la KB.
        kb_embeddings (torch.Tensor | None): Embeddings de centroides normalizados
            pre-calculados para cada intención (basados en 'variantes').
        similarity_threshold (float): Umbral de similitud coseno para match semántico.
    """
    def __init__(self, nombre_experto: str):
        """
        Inicializa el experto, cargando su nombre y KB.

        Args:
            nombre_experto (str): Nombre clave del experto (debe coincidir
                                  con una clave en `EXPERT_KB` de `expert_kb.py`).
        """
        self.nombre = nombre_experto
        self.knowledge_base: List[Dict[str, Any]] = []
        self.kb_keys: List[str] = []
        self.kb_embeddings = None # Se inicializa en MoESystem
        self.similarity_threshold = 0.40 # Umbral base (puede ser sobrescrito)

        self._load_kb() # Carga dinámica al instanciar

    def _load_kb(self):
        """
        Carga la Base de Conocimiento (KB) dinámicamente desde `src.expert_kb`.

        Utiliza `importlib` para mantener el desacoplamiento. Si el módulo
        o la clave del experto no se encuentran, la KB permanecerá vacía y se
        emitirá una advertencia.
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

        Para cada intención en la KB, calcula el vector promedio (centroide)
        de los embeddings de sus 'variantes' y lo normaliza (L2). Almacena
        estos centroides y las 'pregunta_clave' correspondientes.

        Esta función es llamada por `MoESystem.__init__` después de cargar el modelo.
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
                # print(f"› Centroides normalizados para {self.nombre} (KB size: {len(self.kb_keys)})") # Debug
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

        Args:
            key (str): La 'pregunta_clave' a buscar.

        Returns:
            Dict[str, Any]: Diccionario de la intención o uno 'default' si no se encuentra.
        """
        for item in self.knowledge_base:
            if item.get("pregunta_clave") == key:
                return item
        # Fallback genérico si la clave (incluida 'default') no está explícitamente en la KB
        return {
            "pregunta_clave": "default",
            "respuesta": f"Analicemos tu situación general relacionada con {self.nombre.lower()}.",
            "contexto_emocional_esperado": "neutral",
            "tags": [],
            "variantes": [] # Asegurar que la clave exista
        }

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Genera la recomendación buscando la intención más relevante al prompt.

        Compara el embedding normalizado del prompt contra los centroides
        normalizados de la KB. Prioriza búsqueda semántica, con fallback a keyword y default.

        Args:
            prompt (str): Consulta textual del usuario.
            **kwargs: Argumentos adicionales (usados en subclases).

        Returns:
            Tuple[str, Dict[str, Any]]: Tupla con (respuesta_formateada, intencion_completa).
        """
        model = get_semantic_model()
        matched_intention = None
        best_match_score = -1.0 # Inicializar score

        # --- Intento de Búsqueda Semántica ---
        if model and self.kb_embeddings is not None and len(self.kb_keys) > 0:
            try:
                prompt_embedding_raw = model.encode(prompt, convert_to_tensor=True)
                # Asegurar que el embedding no sea nulo o vacío
                if prompt_embedding_raw is None or prompt_embedding_raw.nelement() == 0:
                        raise ValueError("Embedding del prompt resultó vacío.")

                prompt_embedding = F.normalize(prompt_embedding_raw, p=2, dim=0)
                # Mover embeddings a la misma device antes de cos_sim
                kb_embeds_device = self.kb_embeddings.to(prompt_embedding.device)
                cos_scores = util.cos_sim(prompt_embedding, kb_embeds_device)[0]

                best_match_idx = torch.argmax(cos_scores).item()
                best_match_score = cos_scores[best_match_idx].item()

                if best_match_score > self.similarity_threshold:
                    best_match_key = self.kb_keys[best_match_idx]
                    matched_intention = self._get_intention_by_key(best_match_key)
                    # print(f"DEBUG {self.nombre}: Match SEMÁNTICO '{best_match_key}' (Score: {best_match_score:.3f})") # Debug
            except Exception as e:
                warnings.warn(f"Error búsqueda semántica {self.nombre}: {e}. Intentando fallback.")
                # No asignar matched_intention aquí, continuar al fallback

        # --- Fallback: Búsqueda por Palabra Clave (si semántica falló o score bajo) ---
        if matched_intention is None: # Solo si no hubo match semántico válido
            prompt_lower = prompt.lower()
            # Buscar coincidencias exactas en pregunta_clave o variantes
            for item in self.knowledge_base:
                key = item.get("pregunta_clave", "").lower()
                variantes_lower = [str(v).lower() for v in item.get("variantes", [])]
                # Buscar prompt en clave o variantes
                if key in prompt_lower or any(var in prompt_lower for var in variantes_lower if len(var) > 3): # Evitar matches cortos
                    matched_intention = item
                    # print(f"DEBUG {self.nombre}: Match KEYWORD '{item.get('pregunta_clave')}'") # Debug
                    break

        # --- Último Recurso: Respuesta Default ---
        if matched_intention is None:
            # print(f"DEBUG {self.nombre}: Match DEFAULT (Sem score: {best_match_score:.3f})") # Debug
            matched_intention = self._get_intention_by_key("default")

        response_str = f"[{self.nombre}]: {matched_intention.get('respuesta', 'No encontré una respuesta específica.')}"
        return response_str, matched_intention


# --- Implementaciones concretas de los Expertos ---
# (Se mantienen las clases hijas, pero se eliminan los docstrings redundantes
#  ya que heredan de Experto y la lógica específica está documentada)

class GestorCUD(Experto):
    def __init__(self):
        super().__init__("GestorCUD")
        # No necesita umbral específico, usa el base
        # _initialize_knowledge_base() se llama en MoESystem

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Genera recomendación CUD, con lógica proactiva si `is_proactive_call` es True."""
        base_response, matched_intention = super().generate_recommendation(prompt, **kwargs)
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
            return proactive_response, proactive_intention

        return base_response, matched_intention

class TutorCarrera(Experto):
    def __init__(self):
        super().__init__("TutorCarrera")
        self.similarity_threshold = 0.55 # Umbral específico
        # _initialize_knowledge_base() se llama en MoESystem

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Genera recomendación de carrera, añadiendo consejo sobre cupo CUD si aplica."""
        base_response, matched_intention = super().generate_recommendation(prompt, **kwargs)
        original_profile = kwargs.get('original_profile')
        if original_profile is not None and original_profile.get('TIENE_CUD') == 'Si_Tiene_CUD':
            if "cud" not in matched_intention.get("tags", []):
                base_response += (
                    "\n  [Consejo Legal Adicional]: Dado que posees el CUD, recuerda que postular a "
                    "concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)."
                )
        return base_response, matched_intention

class TutorInteraccion(Experto):
    def __init__(self):
        super().__init__("TutorInteraccion")
        # Usa umbral base
        # _initialize_knowledge_base() se llama en MoESystem

class TutorCompetencias(Experto):
    def __init__(self):
        super().__init__("TutorCompetencias")
        # Usa umbral base
        # _initialize_knowledge_base() se llama en MoESystem

class TutorBienestar(Experto):
    def __init__(self):
        super().__init__("TutorBienestar")
        # Usa umbral base
        # _initialize_knowledge_base() se llama en MoESystem

class TutorApoyos(Experto):
    def __init__(self):
        super().__init__("TutorApoyos")
        # Usa umbral base
        # _initialize_knowledge_base() se llama en MoESystem

class TutorPrimerEmpleo(Experto):
    def __init__(self):
        super().__init__("TutorPrimerEmpleo")
        # Usa umbral base
        # _initialize_knowledge_base() se llama en MoESystem


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

    Utiliza un modelo cognitivo para predecir el arquetipo, modula los pesos
    de los expertos según emoción y contexto, selecciona recomendaciones
    semánticamente relevantes y aplica Gating Afectivo.

    Attributes:
        cognitive_model: Modelo predictivo de arquetipos (ej. RandomForest).
        feature_columns (List[str]): Nombres de features para el modelo cognitivo.
        expert_map (Dict[str, Experto]): Mapeo Arquetipo -> Instancia Experto.
        cud_expert (GestorCUD): Instancia del experto CUD.
        affective_rules (Dict): Reglas de modulación afectiva (desde config).
        thresholds (Dict): Umbrales del sistema (desde config).
        affective_congruence_log (List[Dict[str, str]]): Log histórico de eventos de gating.
    """
    def __init__(self, cognitive_model: Any, feature_columns: List[str],
                 affective_rules: Dict, thresholds: Dict):
        """
        Inicializa el MoESystem y pre-calcula embeddings de todos los expertos.

        Args:
            cognitive_model: Modelo predictivo de arquetipos entrenado.
            feature_columns (List[str]): Lista de nombres de features que espera el modelo.
            affective_rules (Dict): Diccionario de reglas de modulación afectiva.
            thresholds (Dict): Diccionario de umbrales del sistema.
        """
        self.cognitive_model = cognitive_model
        # Asegurarse de que feature_columns sea una lista (no un Index de Pandas)
        self.feature_columns = list(feature_columns) if feature_columns is not None else []
        self.expert_map = EXPERT_MAP
        self.cud_expert = CUD_EXPERT
        self.affective_rules = affective_rules or {}
        self.thresholds = thresholds.get('affective_engine', thresholds or {})
        self.affective_congruence_log: List[Dict[str, str]] = []

        # Inicializar modelo semántico y embeddings de TODOS los expertos
        print("› Inicializando modelo semántico y embeddings de expertos...")
        get_semantic_model() # Carga el modelo si no está cargado
        # Iterar sobre todos los expertos (incluido CUD) para inicializar KB
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
        Aplica modulación exponencial (factor^probabilidad).

        Args:
            base_weights (Dict): Pesos iniciales (ej. {ArquetipoPredicho: 1.0}).
            emotion_probs (Dict): Diccionario {emocion: probabilidad}.

        Returns:
            Dict: Pesos modulados.
        """
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated_weights = base_weights.copy()

        for emotion, prob in emotion_probs.items():
            emotion_norm = str(emotion).strip().capitalize()
            if prob > min_prob and emotion_norm in self.affective_rules:
                rules_for_emotion = self.affective_rules[emotion_norm]
                # Asegurarse de que rules_for_emotion sea un diccionario
                if isinstance(rules_for_emotion, dict):
                    for archetype, factor in rules_for_emotion.items():
                        if archetype in modulated_weights and isinstance(factor, (int, float)) and factor > 0:
                            # Modulación exponencial: peso *= factor^probabilidad
                            try:
                                modulated_weights[archetype] *= factor ** prob
                            except (OverflowError, ValueError):
                                warnings.warn(f"Cálculo inválido en modulación: {factor} ** {prob}")
                        elif archetype not in modulated_weights:
                            # Ignorar si el arquetipo no está en los pesos base (raro)
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

        Args:
            weights (Dict): Pesos actuales.
            context (Dict): Contexto con 'emotional_trajectory'.
            negative_emotions (List[str]): Lista de emociones negativas.

        Returns:
            Dict: Pesos ajustados.
        """
        modulated_weights = weights.copy()
        emotional_trajectory = context.get("emotional_trajectory", [])[-2:] # Últimas 2

        # Potenciar Bienestar ante racha negativa
        if len(emotional_trajectory) == 2 and all(e in negative_emotions for e in emotional_trajectory):
            # 'Potencial_Latente' es el arquetipo asociado a TutorBienestar
            if 'Potencial_Latente' in modulated_weights:
                boost_factor = 1.5 # Factor de potenciación
                logging.info(f"Racha negativa detectada. Potenciando Tutor Bienestar x{boost_factor}.")
                modulated_weights['Potencial_Latente'] *= boost_factor

        # (Placeholder para lógica de frustración de Épica 3)
        # Por ejemplo: if context.get("user_frustration_flag"): ...

        return modulated_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """
        Normaliza un diccionario de pesos para que sumen 1.0.

        Args:
            weights (Dict): Diccionario {nombre_experto: peso}.

        Returns:
            Dict: Pesos normalizados, o el original si la suma es 0.
        """
        # Filtrar valores no numéricos o negativos antes de sumar
        valid_weights = {k: v for k, v in weights.items() if isinstance(v, (int, float)) and v >= 0}
        total_weight = sum(valid_weights.values())

        if total_weight > 0:
            # Normalizar solo los pesos válidos
            normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
            # Rellenar con 0 los pesos que no eran válidos o estaban ausentes
            all_keys = set(weights.keys()) | set(valid_weights.keys())
            return {k: normalized_weights.get(k, 0.0) for k in all_keys}

        # Si la suma es 0 o no hay pesos válidos, devolver pesos como 0
        return {k: 0.0 for k in weights.keys()}


    def _get_top_emotion(self, emotion_probs: Dict, official_labels: List[str]) -> str:
        """
        Extrae la emoción dominante normalizada según la lista oficial.
        (Refactorizado: ya no depende de st.session_state).

        Args:
            emotion_probs (Dict): Diccionario {emocion: probabilidad}.
            official_labels (List[str]): Lista de etiquetas de emoción oficiales
                                         (desde config.yaml).

        Returns:
            str: Nombre normalizado de la emoción dominante o 'Neutral'.
        """
        if not emotion_probs or not isinstance(emotion_probs, dict):
            return "Neutral"

        try:
            # Crear mapa de normalización desde las etiquetas recibidas
            label_map = {label.strip().lower(): label for label in official_labels}

            top_emotion_raw = max(emotion_probs, key=emotion_probs.get)

            # Normalizar y buscar en el mapa oficial
            normalized_key = top_emotion_raw.strip().lower()
            top_emotion_normalized = label_map.get(normalized_key, top_emotion_raw.strip().capitalize())

            # Fallback final si la etiqueta no es válida o es desconocida
            if "Etiqueta_" in top_emotion_normalized or not top_emotion_normalized or top_emotion_normalized not in official_labels + ["Neutral"]:
                # Verificar si "Neutral" está explícitamente en las etiquetas oficiales
                if "Neutral" in official_labels:
                    return "Neutral"
                else:
                    # Si "Neutral" no está en config, usar la primera etiqueta oficial como fallback seguro
                    return official_labels[0] if official_labels else "Desconocida"

            return top_emotion_normalized

        except (ValueError, AttributeError, KeyError): # Capturar errores si emotion_probs es inválido
            logging.warning(f"Error procesando emotion_probs: {emotion_probs}. Devolviendo Neutral.")
            # Similar fallback que arriba
            if "Neutral" in official_labels: return "Neutral"
            return official_labels[0] if official_labels else "Desconocida"


    def _log_gating_event(self, tutor_name: str, intention: Dict, detected_emotion: str,
                          official_labels: List[str]) -> Dict[str, str]:
        """
        Calcula congruencia y registra evento de Gating Afectivo.
        (Refactorizado: ya no depende de st.session_state).

        Args:
            tutor_name (str): Nombre del tutor.
            intention (Dict): Intención seleccionada.
            detected_emotion (str): Emoción dominante detectada (normalizada).
            official_labels (List[str]): Lista de etiquetas de emoción oficiales.

        Returns:
            Dict[str, str]: Detalles del evento de gating registrado.
        """
        # Obtener emoción esperada, normalizar y validar contra etiquetas oficiales
        emo_esperada_raw = intention.get("contexto_emocional_esperado", "neutral").strip().capitalize()
        valid_labels = official_labels + ["Neutral"] # Incluir Neutral
        emo_esperada = emo_esperada_raw if emo_esperada_raw in valid_labels else "Neutral"

        # Determinar tipo de congruencia
        if detected_emotion == "Neutral" or emo_esperada == "Neutral":
            tipo_congruencia = "Neutral"
        elif detected_emotion == emo_esperada:
            tipo_congruencia = "Congruente"
        else:
            tipo_congruencia = "Incongruente"
            # (Placeholder para lógica futura de reacción a incongruencia)

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
    ) -> Tuple[str, str, Dict[str, float]]: # <-- MODIFICADO TIPO DE RETORNO
        """
        Orquesta la generación completa del plan de acción adaptativo.
        (MODIFICADO para devolver `final_weights`).

        Args:
            user_profile (pd.Series): Características fuzzificadas del usuario.
            emotion_probs (Dict): {emocion: prob} del clasificador.
            conversation_context (Dict): Historial ('emotional_trajectory').
            config (Dict): Configuración general (para constantes).
            user_prompt (str): Consulta actual del usuario.

        Returns:
            Tuple[str, str, Dict[str, float]]: Tupla con:
                - plan_string (str): Texto completo del plan.
                - predicted_archetype (str): Arquetipo cognitivo predicho.
                - final_weights (Dict[str, float]): Pesos finales de los expertos.
        """
        # --- 0. Extraer Constantes de Config (INYECCIÓN DE DEPENDENCIA) ---
        constants = config.get('constants', {})
        # Extraer etiquetas una sola vez para pasarlas a funciones hijas
        official_labels = constants.get('emotion_labels', [])
        negative_emotions = constants.get('negative_emotions', [])

        # --- 1. Predicción Cognitiva ---
        # Asegurar que las features estén en el orden correcto y manejar faltantes
        try:
             # Usar reindex para asegurar orden y columnas, llenar NaNs con 0.0
             profile_for_prediction = user_profile.reindex(self.feature_columns).fillna(0.0)
             # Convertir a DataFrame de 1 fila
             profile_df = pd.DataFrame([profile_for_prediction])

             predicted_archetype = self.cognitive_model.predict(profile_df)[0]
        except (KeyError, ValueError, AttributeError) as e:
             logging.error(f"Error preparando features para predicción: {e}. Usando default.")
             # Fallback a un arquetipo si falla la preparación o predicción
             predicted_archetype = list(self.expert_map.keys())[0]
        except Exception as e: # Captura otros errores de predicción
             logging.error(f"Error inesperado prediciendo arquetipo: {e}. Usando default.")
             predicted_archetype = list(self.expert_map.keys())[0]

        # --- 2. Pesos Base ---
        base_weights = {archetype: 0.0 for archetype in self.expert_map.keys()}
        if predicted_archetype in base_weights:
            base_weights[predicted_archetype] = 1.0

        # --- 3. Modulación Afectiva ---
        affective_weights = self._apply_affective_modulation(base_weights, emotion_probs)

        # --- 4. Modulación Contextual ---
        # (Las constantes ya se extrajeron en el paso 0)
        conversational_weights = self._apply_conversational_modulation(
            affective_weights, conversation_context, negative_emotions
        )

        # --- 5. Normalización Final ---
        final_weights = self._normalize_weights(conversational_weights)

        # --- 6. Construcción del Plan y Gating ---
        sorted_plan = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)
        final_recs = []
        
        # Obtener emoción dominante (pasando las etiquetas)
        top_detected_emotion = self._get_top_emotion(emotion_probs, official_labels)


        # --- Acción Proactiva CUD ---
        has_cud = user_profile.get('TIENE_CUD') == 'Si_Tiene_CUD'
        gating_log_for_ui = {} # Diccionario para logs específicos de esta respuesta
        if not has_cud:
            rec_str_cud, rec_int_cud = self.cud_expert.generate_recommendation(
                prompt=user_prompt, original_profile=user_profile, is_proactive_call=True
            )
            final_recs.append(rec_str_cud)
            # Loguear Gating (pasando las etiquetas)
            gating_cud = self._log_gating_event(
                "GestorCUD (Proactivo)", rec_int_cud, top_detected_emotion, official_labels
            )
            gating_log_for_ui["cud_expert"] = gating_cud # Guardar para posible uso futuro

        # --- Recomendaciones de Arquetipo ---
        final_recs.append("**[Plan de Acción Adaptativo (Arquetipo)]**")
        min_rec_weight = self.thresholds.get('min_recommendation_weight', 0.15)
        recommendations_added = 0

        for archetype, weight in sorted_plan:
            if weight > min_rec_weight and archetype in self.expert_map:
                expert = self.expert_map[archetype]
                rec_str_arch, rec_int_arch = expert.generate_recommendation(
                    prompt=user_prompt, original_profile=user_profile
                )
                is_cud_topic_arch = "cud" in rec_int_arch.get("tags", [])
                # Añadir si: (NO es CUD) O (SÍ es CUD PERO usuario tiene CUD)
                if not is_cud_topic_arch or has_cud:
                    # Formato mejorado con nombre del experto
                    final_recs.append(f"  - ({weight:.0%}) {rec_str_arch}") # Quitar 'Prioridad:'
                    recommendations_added += 1
                    # Loguear Gating Afectivo SOLO para el tutor principal (el 1ro añadido)
                    if recommendations_added == 1:
                        # Loguear Gating (pasando las etiquetas)
                        gating_archetype = self._log_gating_event(
                            expert.nombre, rec_int_arch, top_detected_emotion, official_labels
                        )
                        gating_log_for_ui["archetype_expert"] = gating_archetype

        # --- Mensaje Default si no hubo recomendaciones ---
        if recommendations_added == 0 and len(final_recs) <= 1: # Solo título
            default_expert_name = predicted_archetype if predicted_archetype in self.expert_map else list(self.expert_map.keys())[0]
            default_expert = self.expert_map[default_expert_name]
            _, default_intention = default_expert.generate_recommendation(prompt="default", original_profile=user_profile)
            default_response = default_intention.get("respuesta", "Analicemos tu situación.")
            # Formato consistente
            final_recs.append(f"  - [{default_expert.nombre} (Default)]: {default_response}")
            # Loguear Gating (pasando las etiquetas)
            gating_default = self._log_gating_event(
                default_expert.nombre + " (Default)", default_intention, top_detected_emotion, official_labels
            )
            gating_log_for_ui["default_expert"] = gating_default


        plan_string = "\n".join(final_recs)
        # Devolver el diccionario `final_weights` en lugar de `analysis_log_data`
        return plan_string, predicted_archetype, final_weights
