# cognitive_tutor.py
"""
Define la arquitectura del Tutor Cognitivo (Versión Avanzada).

Implementa una búsqueda semántica basada en embeddings y
un mecanismo de Gating Afectivo de Intención  para modular
las recomendaciones según la congruencia emocional detectada.
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
    SEMANTIC_SEARCH_ENABLED = True
except ImportError:
    warnings.warn(
        "`sentence-transformers` no está instalado. "
        "El tutor recurrirá a la búsqueda simple por palabras clave."
    )
    SEMANTIC_SEARCH_ENABLED = False
    # Definir Clases Dummy si falta la librería
    class SentenceTransformer: pass
    class util: pass
    class torch: pass

# --- 2. GESTOR DEL MODELO SEMÁNTICO (SINGLETON) ---

_SEMANTIC_MODEL = None
_MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es' # Modelo de embeddings multilingüe

def get_semantic_model() -> SentenceTransformer | None:
    """
    Carga y devuelve el modelo de SentenceTransformer como un singleton.

    Esto evita recargar el modelo pesado en memoria repetidamente.
    Si la librería `sentence-transformers` no está disponible o el modelo
    falla al cargar, devuelve None y desactiva la búsqueda semántica.

    Returns:
        Una instancia del modelo SentenceTransformer o None si falla la carga.
    """
    global _SEMANTIC_MODEL
    if not SEMANTIC_SEARCH_ENABLED:
        return None

    if _SEMANTIC_MODEL is None:
        print(f"› Cargando modelo semántico optimizado para español: {_MODEL_NAME} (esto ocurre solo una vez)...")
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
    intención más relevante a la consulta del usuario.

    Attributes:
        nombre (str): Nombre identificador del experto (ej. "TutorCarrera").
        knowledge_base (List[Dict[str, Any]]): Lista de diccionarios, donde
            cada diccionario representa una intención con claves como
            'pregunta_clave', 'respuesta', 'contexto_emocional_esperado', 'tags'.
        kb_keys (List[str]): Lista extraída de las 'pregunta_clave' para la búsqueda.
        kb_embeddings (torch.Tensor | None): Embeddings pre-calculados de las kb_keys.
        similarity_threshold (float): Umbral de similitud coseno para considerar
            una coincidencia semántica válida.
    """
    def __init__(self, nombre_experto: str):
        """
        Inicializa el experto, cargando su nombre y KB.

        Args:
            nombre_experto: El nombre clave del experto, debe coincidir con
                            una clave en `EXPERT_KB` del módulo `expert_kb`.
        """
        self.nombre = nombre_experto
        self.knowledge_base: List[Dict[str, Any]] = []
        self.kb_keys: List[str] = []
        self.kb_embeddings = None
        self.similarity_threshold = 0.40 # Umbral base

        self._load_kb() # Carga dinámica al instanciar

    def _load_kb(self):
        """
        Carga la Base de Conocimiento (KB) dinámicamente desde `src.expert_kb`.

        Utiliza `importlib` para mantener el desacoplamiento. Si el módulo
        o la clave del experto no se encuentran, la KB permanecerá vacía.
        """
        try:
            kb_module = import_module("src.expert_kb") #
            self.knowledge_base = kb_module.EXPERT_KB.get(self.nombre, []) #
            if not self.knowledge_base:
                warnings.warn(f"KB no encontrada o vacía para el experto: {self.nombre}")
        except ImportError:
            warnings.warn("No se pudo importar 'src.expert_kb'. La KB estará vacía.")
        except Exception as e:
            warnings.warn(f"Error cargando KB para {self.nombre}: {e}")

    def _initialize_knowledge_base(self):
        """
        Pre-calcula los embeddings de las 'pregunta_clave' de la KB.

        Debe llamarse explícitamente después de instanciar al experto.
        Si el modelo semántico no está disponible, los embeddings serán None.
        """
        model = get_semantic_model()
        if not model:
            self.kb_embeddings = None # Asegurar estado consistente
            return

        # Extraer todas las 'pregunta_clave' de la lista de diccionarios
        self.kb_keys = [item.get("pregunta_clave", "") for item in self.knowledge_base] #
        self.kb_keys = [key for key in self.kb_keys if key and key != "default"] # Filtrar vacías y 'default'

        if self.kb_keys:
            try:
                # Pre-calcular los embeddings para todas las claves
                self.kb_embeddings = model.encode(self.kb_keys, convert_to_tensor=True)
                # print(f"› Embeddings calculados para {self.nombre} (KB size: {len(self.kb_keys)})") # Debug opcional
            except Exception as e:
                 warnings.warn(
                     f"Error al codificar KB para {self.nombre}: {e}. "
                     "Búsqueda semántica puede fallar."
                 )
                 self.kb_embeddings = None
        else:
             self.kb_embeddings = None # KB vacía o sin claves válidas

    def _get_intention_by_key(self, key: str) -> Dict[str, Any]:
        """
        Busca y devuelve el diccionario completo de una intención por su 'pregunta_clave'.

        Args:
            key: La 'pregunta_clave' a buscar.

        Returns:
            El diccionario de la intención encontrada o un diccionario 'default'
            si no se encuentra la clave.
        """
        for item in self.knowledge_base:
            if item.get("pregunta_clave") == key: #
                return item
        # Fallback genérico si la clave (incluida 'default') no está explícitamente en la KB
        return {
            "pregunta_clave": "default",
            "respuesta": f"Continuemos analizando tu situación general relacionada con {self.nombre.lower()}.",
            "contexto_emocional_esperado": "neutral",
            "tags": []
        }

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Genera la recomendación buscando la intención más relevante al prompt.

        Prioriza la búsqueda semántica si está habilitada y los embeddings existen.
        Si falla o está deshabilitada, recurre a una búsqueda simple por
        palabra clave en las 'pregunta_clave'.

        Args:
            prompt: La consulta textual del usuario.
            **kwargs: Argumentos adicionales (no usados en la base, pero sí
                      en subclases como TutorCarrera).

        Returns:
            Una tupla conteniendo:
            - La respuesta formateada como string (ej. "[TutorX]: ...").
            - El diccionario completo de la intención encontrada ('default' si no hay match).
        """
        model = get_semantic_model()

        # --- Intento de Búsqueda Semántica ---
        if model and self.kb_embeddings is not None and len(self.kb_keys) > 0:
            try:
                prompt_embedding = model.encode(prompt, convert_to_tensor=True)
                cos_scores = util.cos_sim(prompt_embedding, self.kb_embeddings)[0]
                best_match_idx = torch.argmax(cos_scores).item()
                best_match_score = cos_scores[best_match_idx].item()

                if best_match_score > self.similarity_threshold:
                    best_match_key = self.kb_keys[best_match_idx]
                    matched_intention = self._get_intention_by_key(best_match_key)
                    response_str = f"[{self.nombre}]: {matched_intention['respuesta']}" #
                    return response_str, matched_intention
                # Si la similitud es baja, NO recurrir a default aún, probar fallback
            except Exception as e:
                warnings.warn(f"Error durante búsqueda semántica para {self.nombre}: {e}. Intentando fallback.")
                # Continúa al fallback si hay error

        # --- Fallback: Búsqueda por Palabra Clave (si semántica falló o score bajo) ---
        prompt_lower = prompt.lower()
        # Buscar coincidencias exactas o parciales (más robusto)
        found_item = None
        for item in self.knowledge_base:
             key = item.get("pregunta_clave", "") #
             if key.lower() in prompt_lower:
                 found_item = item
                 break # Usar la primera coincidencia por palabra clave

        if found_item:
             response_str = f"[{self.nombre}]: {found_item['respuesta']}" #
             return response_str, found_item

        # --- Último Recurso: Respuesta Default ---
        default_intention = self._get_intention_by_key("default")
        fallback_response = f"[{self.nombre}]: {default_intention['respuesta']}"
        return fallback_response, default_intention


# --- Implementaciones concretas de los Expertos ---

class GestorCUD(Experto):
    """
    Experto SEMÁNTICO y PROACTIVO en la gestión del CUD.

    Su KB cubre preguntas frecuentes sobre el CUD. Además, tiene una lógica
    especial en `generate_recommendation` para dar una respuesta proactiva
    si se detecta que el usuario no tiene CUD y la consulta no era sobre eso.
    """
    def __init__(self):
        """Inicializa el GestorCUD cargando su KB específica."""

        super().__init__("GestorCUD") # Usar nombre corto

        self._initialize_knowledge_base() # Pre-calcula embeddings

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Genera recomendación sobre el CUD. Incluye lógica proactiva.

        Args:
            prompt: Consulta del usuario.
            is_proactive_call (bool, opcional): Flag pasado por MoESystem si
                se detectó que el usuario no tiene CUD. Default False.
            **kwargs: Otros argumentos.

        Returns:
            Tupla (respuesta_string, intencion_dict).
        """
        base_response, matched_intention = super().generate_recommendation(prompt, **kwargs)

        is_proactive_call = kwargs.get("is_proactive_call", False)
        # Si la búsqueda semántica/keyword devolvió 'default' Y esta llamada fue proactiva
        if matched_intention.get("pregunta_clave") == "default" and is_proactive_call:
            # Usar una respuesta proactiva más informativa que el default genérico
            proactive_response = (
                "[Acción CUD]: He detectado que no posees el CUD. Se recomienda iniciar el trámite.\n"
                "  › **Qué es:** El CUD (Ley 22.431) es un documento gratuito que te da acceso a derechos " #
                "clave en salud (Ley 24.901) y transporte.\n"
                "  › **Consulta:** Podés preguntarme '¿Qué beneficios tengo con el CUD?' o '¿Dónde se tramita?' " #
                "para más detalles."
            )
            # Modificar la intención 'default' devuelta para reflejar esta respuesta
            proactive_intention = matched_intention.copy() # Evitar modificar el objeto original
            proactive_intention["respuesta"] = proactive_response
            proactive_intention["pregunta_clave"] = "accion_cud_proactiva" # Clave específica
            proactive_intention["tags"] = ["cud", "legal", "proactivo"]
            return proactive_response, proactive_intention

        return base_response, matched_intention


class TutorCarrera(Experto):
    """Experto en estrategia profesional, CV, entrevistas y negociación."""
    def __init__(self):
        """Inicializa TutorCarrera, define su umbral y carga KB/embeddings."""
 
        super().__init__("TutorCarrera") # Usar nombre corto

        self.similarity_threshold = 0.50 # Umbral más alto para este tutor
        self._initialize_knowledge_base()

    def generate_recommendation(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Genera recomendación de carrera, añadiendo consejo sobre CUD si aplica.

        Args:
            prompt: Consulta del usuario.
            original_profile (pd.Series, opcional): Perfil del usuario para
                verificar si tiene CUD.
            **kwargs: Otros argumentos.

        Returns:
            Tupla (respuesta_string, intencion_dict).
        """
        base_response, matched_intention = super().generate_recommendation(prompt, **kwargs)

        original_profile = kwargs.get('original_profile')
        # Añadir consejo CUD solo si el usuario TIENE CUD
        if original_profile is not None and original_profile.get('TIENE_CUD') == 'Si_Tiene_CUD':
             # Y si la intención principal NO era sobre CUD para evitar redundancia
             if "cud" not in matched_intention.get("tags", []): #
                base_response += (
                    "\n  [Consejo Legal Adicional]: Dado que posees el CUD, recuerda que postular a "
                    "concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)." #
                 )

        return base_response, matched_intention


class TutorInteraccion(Experto):
    """Experto en habilidades blandas, comunicación y manejo de conflictos."""
    def __init__(self):
        """Inicializa TutorInteraccion y carga KB/embeddings."""
        super().__init__("TutorInteraccion")
        self._initialize_knowledge_base()


class TutorCompetencias(Experto):
    """Experto en 'upskilling', cursos y aprendizaje de nuevas tecnologías."""
    def __init__(self):
        """Inicializa TutorCompetencias y carga KB/embeddings."""
        super().__init__("TutorCompetencias") 
        self._initialize_knowledge_base()


class TutorBienestar(Experto):
    """Experto en confianza, manejo del estrés y motivación."""
    def __init__(self):
        """Inicializa TutorBienestar y carga KB/embeddings."""
        super().__init__("TutorBienestar") 
        self._initialize_knowledge_base()


class TutorApoyos(Experto):
    """Experto en adaptaciones, derechos y 'ajustes razonables'."""
    def __init__(self):
        """Inicializa TutorApoyos y carga KB/embeddings."""
        super().__init__("TutorApoyos") 
        self._initialize_knowledge_base()


class TutorPrimerEmpleo(Experto):
    """Experto en guiar a jóvenes en su primera experiencia laboral."""
    def __init__(self):
        """Inicializa TutorPrimerEmpleo y carga KB/embeddings."""
        super().__init__("TutorPrimerEmpleo") 
        self._initialize_knowledge_base()


# -----------------------------------------------------------------
# --- 4. SISTEMA DE ORQUESTACIÓN MoE ---
# -----------------------------------------------------------------

# Mapeo central de Arquetipos a sus Tutores Expertos
# Se instancia cada tutor aquí. La inicialización de embeddings se hará en MoESystem.
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

    Utiliza un modelo cognitivo para predecir el arquetipo principal, modula
    los pesos de los expertos basados en la emoción detectada y el contexto
    conversacional, y finalmente selecciona las recomendaciones más relevantes
    usando la búsqueda semántica de cada experto. Implementa Gating Afectivo.

    Attributes:
        cognitive_model: Modelo entrenado para predecir arquetipos (ej. RandomForest).
        feature_columns (List[str]): Nombres de las columnas usadas por el cognitive_model.
        expert_map (Dict[str, Experto]): Mapeo de nombres de arquetipo a instancias de Experto.
        cud_expert (GestorCUD): Instancia del experto CUD.
        affective_rules (Dict): Reglas cargadas desde config.yaml para modulación afectiva.
        thresholds (Dict): Umbrales del sistema (ej. min_emotion_probability).
        affective_congruence_log (List[Dict[str, str]]): Log histórico de eventos de gating.
    """
    def __init__(self, cognitive_model: Any, feature_columns: List[str],
                 affective_rules: Dict, thresholds: Dict):
        """
        Inicializa el sistema MoE y pre-calcula embeddings de todos los expertos.

        Args:
            cognitive_model: El modelo predictivo de arquetipos.
            feature_columns: Lista de nombres de las características de entrada.
            affective_rules: Diccionario de reglas de modulación afectiva.
            thresholds: Diccionario de umbrales del sistema.
        """
        self.cognitive_model = cognitive_model
        self.feature_columns = feature_columns
        self.expert_map = EXPERT_MAP
        self.cud_expert = CUD_EXPERT # Guardar la instancia del experto CUD
        self.affective_rules = affective_rules or {} # Asegurar que sea un dict
        # Extraer sub-diccionario 'affective_engine' si existe, si no, usar thresholds directamente
        self.thresholds = thresholds.get('affective_engine', thresholds or {})
        self.affective_congruence_log: List[Dict[str, str]] = []

        # Inicializar modelo semántico y embeddings de TODOS los expertos al crear MoESystem
        print("› Inicializando modelo semántico y embeddings de expertos...")
        get_semantic_model() # Carga el modelo si no está cargado
        for expert in list(self.expert_map.values()) + [self.cud_expert]:
            expert._initialize_knowledge_base()
        print("› Embeddings de expertos listos.")


    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        """
        Modula los pesos base de los expertos según las emociones detectadas.

        Utiliza las reglas definidas en `affective_rules` (config.yaml).
        Aplica una modulación exponencial para dar más peso a probabilidades altas.

        Args:
            base_weights: Pesos iniciales (generalmente 1.0 para el arquetipo predicho).
            emotion_probs: Diccionario {emocion: probabilidad} del clasificador.

        Returns:
            Diccionario de pesos modulados.
        """
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated_weights = base_weights.copy()

        for emotion, prob in emotion_probs.items():
            emotion_norm = str(emotion).strip().capitalize()
            # Validar que la emoción esté en las reglas y supere el umbral
            if prob > min_prob and emotion_norm in self.affective_rules:
                rules_for_emotion = self.affective_rules[emotion_norm]
                for archetype, factor in rules_for_emotion.items():
                    if archetype in modulated_weights:
                        # Modulación exponencial: peso *= factor^probabilidad
                        modulated_weights[archetype] *= factor ** prob
                        # Logica anterior (lineal): modulated_weights[archetype] *= (1 + (factor - 1) * prob)
        return modulated_weights

    def _apply_conversational_modulation(self, weights: Dict, context: Dict,
                                         negative_emotions: List[str]) -> Dict:
        """
        Ajusta los pesos basados en el historial reciente de la conversación.

        Actualmente, potencia al TutorBienestar si detecta una racha de
        emociones negativas. Podría extenderse para manejar frustración.

        Args:
            weights: Pesos actuales de los expertos.
            context: Diccionario con el contexto (ej. 'emotional_trajectory').
            negative_emotions: Lista de nombres de emociones consideradas negativas.

        Returns:
            Diccionario de pesos ajustados contextualmente.
        """
        modulated_weights = weights.copy()
        emotional_trajectory = context.get("emotional_trajectory", [])[-2:] # Últimas 2

        # Potenciar Bienestar ante racha negativa
        if len(emotional_trajectory) == 2 and all(e in negative_emotions for e in emotional_trajectory):
            # 'Potencial_Latente' es el arquetipo asociado a TutorBienestar
            if 'Potencial_Latente' in modulated_weights:
                boost_factor = 1.5 # Factor de potenciación (podría ir a config.yaml)
                logging.info(f"Racha negativa detectada. Potenciando Tutor Bienestar x{boost_factor}.")
                modulated_weights['Potencial_Latente'] *= boost_factor

        # (Placeholder para lógica de frustración de Épica 3)

        return modulated_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """
        Normaliza un diccionario de pesos para que la suma total sea 1.0.

        Args:
            weights: Diccionario {nombre_experto: peso}.

        Returns:
            Diccionario con pesos normalizados. Devuelve el original si la suma es 0.
        """
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in weights.items()}
        return weights # Evitar división por cero

    def _get_top_emotion(self, emotion_probs: Dict) -> str:
        """
        Extrae la emoción con mayor probabilidad del diccionario.

        Normaliza el nombre de la emoción y devuelve 'Neutral' si no hay
        probabilidades o si la emoción no está en las reglas afectivas.

        Args:
            emotion_probs: Diccionario {emocion: probabilidad}.

        Returns:
            Nombre normalizado de la emoción dominante o 'Neutral'.
        """
        if not emotion_probs:
            return "Neutral"

        try:
             # Encontrar la clave (emoción) con el valor (probabilidad) máximo
             top_emotion_raw = max(emotion_probs, key=emotion_probs.get)
             # Limpiar y capitalizar (ej. " ira " -> "Ira")
             normalized = str(top_emotion_raw).strip().capitalize()
             # Lista de emociones válidas (las de las reglas + Neutral)
             valid_emotions = list(self.affective_rules.keys()) + ["Neutral"]
             # Si la emoción detectada es una de las conocidas, devolverla, si no 'Neutral'
             return normalized if normalized in valid_emotions else "Neutral"
        except ValueError: # Caso de diccionario vacío
             return "Neutral"


    def _log_gating_event(self, tutor_name: str, intention: Dict, detected_emotion: str) -> Dict[str, str]:
        """
        Calcula la (in)congruencia y registra el evento de Gating Afectivo.

        Compara la emoción detectada con la esperada por la intención y
        registra el evento en `self.affective_congruence_log` y en el log de Python.

        Args:
            tutor_name: Nombre del tutor que generó la intención.
            intention: Diccionario de la intención seleccionada.
            detected_emotion: Emoción dominante detectada en el prompt del usuario.

        Returns:
            Diccionario con los detalles del evento de gating registrado.
        """
        # Obtener emoción esperada, normalizar y validar
        emo_esperada = intention.get("contexto_emocional_esperado", "neutral").capitalize() #
        valid_emotions = list(self.affective_rules.keys()) + ["Neutral"]
        emo_esperada = emo_esperada if emo_esperada in valid_emotions else "Neutral"

        # Determinar tipo de congruencia
        if detected_emotion == "Neutral" or emo_esperada == "Neutral":
            tipo_congruencia = "Neutral"
        elif detected_emotion == emo_esperada:
            tipo_congruencia = "Congruente"
        else:
            tipo_congruencia = "Incongruente"
            # Aquí se podría añadir lógica futura para reaccionar a incongruencias

        # Crear registro del evento
        gating_entry = {
            "tutor": tutor_name,
            "intencion_mapeada": intention.get("pregunta_clave", "default"), #
            "emo_detectada": detected_emotion,
            "emo_esperada": emo_esperada,
            "tipo": tipo_congruencia
        }

        # Guardar en el log histórico y loguear en consola/archivo
        self.affective_congruence_log.append(gating_entry)
        logging.info(f"GATING AFECTIVO ({tutor_name}): {gating_entry}")

        return gating_entry # Devuelve la entrada para el log de análisis de la respuesta

    def get_cognitive_plan(
        self,
        user_profile: pd.Series,
        emotion_probs: Dict,
        conversation_context: Dict,
        config: Dict,
        user_prompt: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Orquesta la generación completa del plan de acción adaptativo.

        Combina la predicción cognitiva, la modulación afectiva y contextual,
        la selección semántica de intenciones y el Gating Afectivo.

        Args:
            user_profile: Serie de Pandas con las características fuzzificadas del usuario.
            emotion_probs: Diccionario {emocion: prob} del clasificador afectivo.
            conversation_context: Dict con historial ('emotional_trajectory').
            config: Diccionario de configuración general (para constantes).
            user_prompt: La consulta actual del usuario.

        Returns:
            Una tupla conteniendo:
            - plan_string (str): El texto completo del plan de acción.
            - predicted_archetype (str): El nombre del arquetipo cognitivo predicho.
            - analysis_log_data (dict): Datos de análisis (pesos, gating) para la UI.
        """
        # --- 1. Predicción Cognitiva ---
        profile_df = pd.DataFrame([user_profile])
        # Asegurar que el DataFrame tenga exactamente las columnas esperadas y en orden
        profile_for_prediction = profile_df.reindex(columns=self.feature_columns, fill_value=0.0)
        try:
             # Utiliza el modelo cognitivo (ej. RandomForest) para predecir
             predicted_archetype = self.cognitive_model.predict(profile_for_prediction)[0]
        except Exception as e:
             logging.error(f"Error prediciendo arquetipo: {e}. Usando default.")
             # Fallback a un arquetipo por defecto si falla la predicción
             predicted_archetype = list(self.expert_map.keys())[0]


        # --- 2. Pesos Base ---
        # Inicializa pesos a 0, asigna 1.0 al tutor del arquetipo predicho
        base_weights = {archetype: 0.0 for archetype in self.expert_map.keys()}
        if predicted_archetype in base_weights:
            base_weights[predicted_archetype] = 1.0

        # --- 3. Modulación Afectiva ---
        affective_weights = self._apply_affective_modulation(base_weights, emotion_probs)

        # --- 4. Modulación Contextual ---
        constants = config.get('constants', {}) #
        negative_emotions = constants.get('negative_emotions', []) #
        conversational_weights = self._apply_conversational_modulation(
            affective_weights, conversation_context, negative_emotions
        )

        # --- 5. Normalización Final ---
        final_weights = self._normalize_weights(conversational_weights)

        # --- 6. Construcción del Plan y Gating ---
        # Ordenar tutores por peso descendente
        sorted_plan = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)
        final_recs = [] # Lista para almacenar los strings de recomendación

        # Preparar datos de análisis
        top_detected_emotion = self._get_top_emotion(emotion_probs)
        analysis_log_data = {
            "predicted_archetype": predicted_archetype,
            "top_emotion_detected": top_detected_emotion,
            "final_expert_weights": {k: round(v, 3) for k, v in final_weights.items()},
            "gating_log": {} # Se llenará con los eventos de gating
        }

        # --- Acción Proactiva CUD ---
        has_cud = user_profile.get('TIENE_CUD') == 'Si_Tiene_CUD'
        if not has_cud:
            # Llama al experto semántico CUD_EXPERT, indicando que es proactivo
            rec_str_cud, rec_int_cud = self.cud_expert.generate_recommendation(
                prompt=user_prompt,
                original_profile=user_profile,
                is_proactive_call=True # Flag importante para la lógica interna del GestorCUD
            )
            final_recs.append(rec_str_cud)
            # Registrar Gating Afectivo para esta acción proactiva
            gating_cud = self._log_gating_event("GestorCUD (Proactivo)", rec_int_cud, top_detected_emotion)
            analysis_log_data["gating_log"]["cud_expert"] = gating_cud

        # --- Recomendaciones de Arquetipo ---
        final_recs.append("**[Plan de Acción Adaptativo (Arquetipo)]**")
        # Umbral mínimo de peso para incluir una recomendación
        min_rec_weight = self.thresholds.get('min_recommendation_weight', 0.15) #
        recommendations_added = 0

        # Iterar sobre los tutores ordenados por peso
        for archetype, weight in sorted_plan:
            # Solo añadir recomendaciones si el peso es significativo
            if weight > min_rec_weight and archetype in self.expert_map:
                expert = self.expert_map[archetype]
                # Cada experto busca la intención más relevante a la consulta del usuario
                rec_str_arch, rec_int_arch = expert.generate_recommendation(
                    prompt=user_prompt,
                    original_profile=user_profile
                )

                # Lógica para evitar duplicar info CUD si ya se dio proactivamente
                is_cud_topic_arch = "cud" in rec_int_arch.get("tags", []) #
                # Añadir si: (NO es tema CUD) O (SÍ es tema CUD PERO el usuario YA tiene CUD)
                if not is_cud_topic_arch or has_cud:
                    final_recs.append(f"  - (Prioridad: {weight:.0%}): {rec_str_arch}")
                    recommendations_added += 1

                    # Loguear Gating Afectivo SOLO para el tutor principal (el 1ro añadido)
                    if recommendations_added == 1:
                        gating_archetype = self._log_gating_event(expert.nombre, rec_int_arch, top_detected_emotion)
                        analysis_log_data["gating_log"]["archetype_expert"] = gating_archetype


        # --- Mensaje Default si no hubo recomendaciones de arquetipo ---
        if recommendations_added == 0:
             # Solo añadir mensaje default si TAMPOCO hubo recomendación proactiva CUD
             if len(final_recs) <= 1: # (Solo contiene el título "[Plan...]")
                 # Usar el default del tutor predicho como fallback
                 default_expert_name = predicted_archetype if predicted_archetype in self.expert_map else list(self.expert_map.keys())[0]
                 default_expert = self.expert_map[default_expert_name]
                 # Llamar con prompt "default" para obtener su respuesta default específica
                 _, default_intention = default_expert.generate_recommendation(prompt="default", original_profile=user_profile)
                 default_response = default_intention.get("respuesta", "Analicemos tu situación con más detalle.")
                 final_recs.append(f"  - [Sistema]: {default_response}")
                 # Loguear gating para esta respuesta default
                 gating_default = self._log_gating_event(default_expert.nombre + " (Default)", default_intention, top_detected_emotion)
                 analysis_log_data["gating_log"]["default_expert"] = gating_default


        # Unir todas las recomendaciones en un solo string
        plan_string = "\n".join(final_recs)

        return plan_string, predicted_archetype, analysis_log_data
