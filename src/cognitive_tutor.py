"""
Define la arquitectura del Tutor Cognitivo (Versión Avanzada).

Esta versión implementa una **búsqueda semántica** utilizando modelos de
embeddings (sentence-transformers) para que los expertos puedan entender
la intención del usuario, en lugar de depender de palabras clave exactas.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings

# --- 1. IMPORTACIONES PARA LA BÚSQUEDA SEMÁNTICA ---
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SEMANTIC_SEARCH_ENABLED = True
except ImportError:
    warnings.warn("`sentence-transformers` no está instalado. El tutor recurrirá a la búsqueda simple por palabras clave.")
    SEMANTIC_SEARCH_ENABLED = False

# --- 2. GESTOR DEL MODELO SEMÁNTICO (SINGLETON) ---

_SEMANTIC_MODEL = None
_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def get_semantic_model():
    """
    Carga el modelo de SentenceTransformer como un singleton (una sola vez)
    para ahorrar memoria y tiempo de carga.
    """
    global _SEMANTIC_MODEL
    if not SEMANTIC_SEARCH_ENABLED:
        return None
        
    if _SEMANTIC_MODEL is None:
        print("› Cargando modelo semántico (esto ocurre solo una vez)...")
        try:
            _SEMANTIC_MODEL = SentenceTransformer(_MODEL_NAME)
            print("› ✅ Modelo semántico cargado exitosamente.")
        except Exception as e:
            warnings.warn(f"Error al cargar el modelo semántico: {e}. Se desactivará la búsqueda semántica.")
            _SEMANTIC_MODEL = None # Marcar como fallido
            
    return _SEMANTIC_MODEL

# --- 3. ARQUITECTURA DE EXPERTOS CONVERSACIONALES (SEMÁNTICA) ---

class Experto:
    """
    Clase base para tutores con una base de conocimiento semántica.
    """
    def __init__(self, nombre_experto: str):
        self.nombre = nombre_experto
        self.knowledge_base: Dict[str, str] = {
            "default": f"Analicemos tu situación general. Mi consejo principal es sobre {self.nombre.lower()}."
        }
        self.kb_keys: List[str] = []
        self.kb_embeddings = None
        self.similarity_threshold = 0.5 # Umbral de confianza para la coincidencia semántica

    def _initialize_knowledge_base(self):
        """
        Procesa la base de conocimiento, extrae las claves (preguntas)
        y pre-calcula sus embeddings.
        """
        model = get_semantic_model()
        if not model: # Si el modelo no se pudo cargar
            return

        # Extraer todas las claves excepto 'default'
        self.kb_keys = [key for key in self.knowledge_base.keys() if key != "default"]
        
        if self.kb_keys:
            try:
                # Pre-calcular los embeddings para todas las claves
                self.kb_embeddings = model.encode(self.kb_keys, convert_to_tensor=True)
            except Exception as e:
                 warnings.warn(f"Error al codificar KB para {self.nombre}: {e}. La búsqueda semántica puede fallar.")
                 self.kb_embeddings = None

    def generate_recommendation(self, prompt: str, **kwargs) -> str:
        """
        Genera una recomendación usando búsqueda semántica.
        """
        model = get_semantic_model()
        
        # --- Fallback a búsqueda por palabra clave si el modelo semántico falló ---
        if not model or self.kb_embeddings is None:
            prompt_lower = prompt.lower()
            for key, response in self.knowledge_base.items():
                if key != "default" and key in prompt_lower:
                    return f"[{self.nombre}]: {response}"
            return f"[{self.nombre}]: {self.knowledge_base['default']}"
        
        # --- Búsqueda Semántica ---
        try:
            prompt_embedding = model.encode(prompt, convert_to_tensor=True)
            
            # Calcular similitud coseno
            cos_scores = util.cos_sim(prompt_embedding, self.kb_embeddings)[0]
            
            # Encontrar la mejor coincidencia
            best_match_idx = torch.argmax(cos_scores).item()
            best_match_score = cos_scores[best_match_idx].item()
            
            # Devolver la respuesta solo si supera el umbral de confianza
            if best_match_score > self.similarity_threshold:
                best_match_key = self.kb_keys[best_match_idx]
                return f"[{self.nombre}]: {self.knowledge_base[best_match_key]}"
            else:
                # Si la similitud es baja, usar la respuesta por defecto
                return f"[{self.nombre}]: {self.knowledge_base['default']}"
                
        except Exception as e:
            warnings.warn(f"Error durante la búsqueda semántica para {self.nombre}: {e}")
            return f"[{self.nombre}]: {self.knowledge_base['default']}" # Fallback

class GestorCUD(Experto):
    """Experto en la gestión del Certificado Único de Discapacidad."""
    def __init__(self): 
        super().__init__("Gestor de CUD")
    
    def generate_recommendation(self, **kwargs) -> str: 
        return "[Acción CUD]: Se ha detectado que no posee el CUD. Se recomienda iniciar el trámite (Ley 22.431)."

class TutorCarrera(Experto):
    """Experto en estrategia profesional, CV, entrevistas y negociación."""
    def __init__(self):
        super().__init__("Tutor de Estrategia de Carrera")
        self.knowledge_base = {
            "¿Cómo puedo mejorar mi CV?": "Para tu CV, enfócate en logros cuantificables. Por ejemplo, 'optimicé el proceso X, reduciendo el tiempo en un 15%'.",
            "Consejos para una entrevista de trabajo": "Durante una entrevista, prepara respuestas para 'háblame de ti' usando la estructura 'Presente-Pasado-Futuro'. Es muy efectiva.",
            "¿Debería actualizar mi perfil de LinkedIn?": "Tu perfil de LinkedIn debe tener una foto profesional y un titular que describa el valor que aportas, no solo tu puesto actual.",
            "¿Cómo negociar mi salario?": "Al negociar el salario, investiga el rango de mercado. Nunca des la primera cifra, pregunta por el presupuesto que manejan para el rol.",
            "Quiero crecer en mi carrera": "Para crecer profesionalmente, identifica a un mentor en tu campo y busca proyectos que te saquen de tu zona de confort.",
            "default": "El plan es relanzar tu perfil profesional, identificando tus fortalezas clave y alineándolas con las demandas del mercado."
        }
        self.similarity_threshold = 0.6 # Umbral más alto para este tutor
        self._initialize_knowledge_base()
    
    def generate_recommendation(self, prompt: str, **kwargs) -> str:
        base_response = super().generate_recommendation(prompt, **kwargs)
        original_profile = kwargs.get('original_profile')
        if original_profile is not None and original_profile.get('TIENE_CUD') == 'Si_Tiene_CUD':
            base_response += "\n  [Consejo Legal Adicional]: Dado que posees el CUD, recuerda que postular a concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)."
        return base_response

class TutorInteraccion(Experto):
    """Experto en habilidades blandas, comunicación y manejo de conflictos."""
    def __init__(self):
        super().__init__("Tutor de Habilidades de Interacción")
        self.knowledge_base = {
            "¿Cómo puedo comunicarme mejor?": "Una técnica clave es la 'escucha activa': reformula lo que dice la otra persona para asegurar que has entendido. Genera mucha confianza.",
            "Me cuesta hablar en reuniones": "En reuniones, si te cuesta intervenir, prepara una o dos preguntas de antemano. Es una forma fácil de participar.",
            "¿Cómo manejar un conflicto con un colega?": "Para manejar un conflicto, enfócate en el problema, no en la persona. Usa frases como 'Cuando ocurre X, siento Y'.",
            "Tengo que dar feedback a alguien": "Al dar feedback, usa el método 'sandwich': empieza con algo positivo, luego el área de mejora, y cierra con otra nota positiva.",
            "Me da miedo presentar en público": "Al presentar en público, estructura tu discurso con una introducción clara (el problema), un desarrollo (tu solución) y una conclusión (el llamado a la acción).",
            "default": "Focalicémonos en desarrollar estrategias de comunicación adaptadas a tu perfil y objetivos."
        }
        self._initialize_knowledge_base()

class TutorCompetencias(Experto):
    """Experto en 'upskilling', cursos y aprendizaje de nuevas tecnologías."""
    def __init__(self):
        super().__init__("Tutor de Competencias Técnicas")
        self.knowledge_base = {
            "¿Qué cursos online me recomiendas?": "Plataformas como Coursera, edX o Platzi ofrecen rutas de aprendizaje de alta calidad, no solo cursos aislados.",
            "¿Vale la pena sacar una certificación?": "Una certificación oficial (ej. de Google, Microsoft, AWS) puede validar tus habilidades y dar un gran impulso a tu CV.",
            "¿Cómo me mantengo actualizado en tecnología?": "Para mantenerte al día, te recomiendo seguir newsletters especializados en tu área y dedicar 30 minutos al día a leer sobre nuevas tendencias.",
            "¿Qué es mejor, aprender solo o en un bootcamp?": "La mejor forma de aprender una nueva tecnología es construir un pequeño proyecto con ella. La práctica es clave.",
            "¿Qué habilidades son importantes además de programar?": "Además de lo técnico, las 'power skills' como la resolución de problemas complejos y el pensamiento crítico son muy demandadas.",
            "default": "Sugiero que identifiquemos tus habilidades actuales y busquemos cursos cortos y de alto impacto (Upskilling)."
        }
        self._initialize_knowledge_base()

class TutorBienestar(Experto):
    """Experto en confianza, manejo del estrés y motivación."""
    def __init__(self):
        super().__init__("Tutor de Bienestar y Activación")
        self.knowledge_base = {
            "Estoy muy estresado con el trabajo": "Para manejar el estrés, prueba la técnica 'Pomodoro': 25 minutos de trabajo enfocado y 5 de descanso. Ayuda a mantener la mente fresca.",
            "Siento mucha ansiedad antes de las reuniones": "Frente a la ansiedad, practica la 'respiración de caja': inhala contando 4, sostén 4, exhala 4, espera 4. Repite.",
            "Perdí la motivación, no tengo ganas de hacer nada": "Para recuperar la motivación, divide una tarea grande en pasos muy pequeños y celebra cada vez que completas uno.",
            "No tengo confianza en mí mismo": "La autoconfianza se construye con pequeñas victorias. Fija un objetivo muy pequeño para hoy y cúmplelo.",
            "Tengo miedo de fracasar en este nuevo rol": "El miedo al fracaso es normal. Intenta reencuadrarlo: cada error es un dato valioso sobre cómo no hacer algo.",
            "default": "Nuestro primer paso es fortalecer tu autoconfianza e identificar tus intereses principales."
        }
        self._initialize_knowledge_base()

class TutorApoyos(Experto):
    """Experto en adaptaciones, derechos y 'ajustes razonables'."""
    def __init__(self):
        super().__init__("Tutor de Apoyos y Adaptaciones")
        self.knowledge_base = {
            "¿Qué son los ajustes razonables?": "Los 'ajustes razonables' son modificaciones que no imponen una carga desproporcionada al empleador. Tienes derecho a solicitarlos (Ley 26.378).",
            "¿Cuáles son mis derechos laborales?": "Tu derecho principal es la no discriminación. Esto incluye el acceso a entrevistas y adaptaciones en el puesto de trabajo.",
            "¿Cómo debo pedir una adaptación en mi trabajo?": "Al solicitar una adaptación, sé específico: 'Necesito un software de lectura de pantalla para realizar mis tareas de análisis' es mejor que 'necesito ayuda'.",
            "¿Qué beneficios de transporte tengo con el CUD?": "Recuerda que el CUD te da derecho al 'Pase Libre' en transporte público nacional.",
            "¿Cómo sé si un sitio web es accesible?": "La accesibilidad web (ej. sitios que funcionen con lectores de pantalla) es un derecho. Herramientas como 'WAVE' pueden ayudarte a evaluar sitios.",
            "default": "Exploremos juntos las adaptaciones y tecnologías de apoyo que necesitas. La Ley 26.378 garantiza tu derecho a 'ajustes razonables'."
        }
        self._initialize_knowledge_base()

class TutorPrimerEmpleo(Experto):
    """Experto en guiar a jóvenes en su primera experiencia laboral."""
    def __init__(self):
        super().__init__("Tutor de Primer Empleo")
        self.knowledge_base = {
            "No tengo experiencia, ¿qué pongo en mi CV?": "Para tu primer CV, si no tienes experiencia laboral, enfócate en proyectos académicos, voluntariados y habilidades que hayas desarrollado.",
            "Ayuda con mi primera entrevista": "En tu primera entrevista, demuestra entusiasmo y ganas de aprender. Es más importante tu actitud que tu experiencia previa.",
            "¿Dónde busco mi primer trabajo?": "Para buscar tu primer empleo, activa las alertas en LinkedIn y Bumeran, y no subestimes el poder de las redes de contactos de tu universidad.",
            "¿Debería hacer una pasantía?": "Una pasantía es una excelente forma de ganar experiencia. Búcalas activamente, ya que son la mejor puerta de entrada.",
            "¿Qué habilidades blandas son importantes?": "Destaca habilidades blandas: trabajo en equipo, adaptabilidad y buena comunicación. A menudo son más valoradas que las técnicas en un perfil junior.",
            "default": "Te guiaré en la creación de tu primer CV y en cómo prepararte para las entrevistas. ¡Conseguir el primer empleo es un hito importante!"
        }
        self._initialize_knowledge_base()

# Mapeo central de Arquetipos a sus Tutores Expertos
EXPERT_MAP = {
    'Com_Desafiado': TutorInteraccion(),
    'Nav_Informal': TutorCompetencias(),
    'Prof_Subutil': TutorCarrera(),
    'Potencial_Latente': TutorBienestar(),
    'Cand_Nec_Sig': TutorApoyos(),
    'Joven_Transicion': TutorPrimerEmpleo()
}

# --- 4. SISTEMA DE ORQUESTACIÓN MoE (SEMÁNTICO) ---

class MoESystem:
    """Sistema de Mezcla de Expertos (MoE) que orquesta la respuesta del tutor."""
    def __init__(self, cognitive_model, feature_columns: List[str], 
                 affective_rules: Dict, thresholds: Dict):
        """Inicializa el sistema MoE."""
        self.cognitive_model = cognitive_model
        self.feature_columns = feature_columns
        self.expert_map = EXPERT_MAP
        self.affective_rules = affective_rules
        self.thresholds = thresholds.get('affective_engine', thresholds)
        # Inicializar el modelo semántico al crear el sistema
        get_semantic_model()

    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        """Aplica la modulación afectiva a los pesos base de los expertos."""
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated_weights = base_weights.copy()
        
        for emotion, prob in emotion_probs.items():
            emotion_norm = str(emotion).strip().capitalize()
            if prob > min_prob and emotion_norm in self.affective_rules:
                rules = self.affective_rules[emotion_norm]
                for archetype, factor in rules.items():
                    if archetype in modulated_weights:
                        modulated_weights[archetype] *= (1 + (factor - 1) * prob)
        return modulated_weights
        
    def _apply_conversational_modulation(self, weights: Dict, context: Dict, 
                                         negative_emotions: List[str]) -> Dict:
        """Aplica modulación basada en la memoria de la conversación."""
        modulated_weights = weights.copy()
        emotional_trajectory = context.get("emotional_trajectory", [])
        
        if len(emotional_trajectory) >= 2:
            last_two_emotions = emotional_trajectory[-2:]
            if all(emotion in negative_emotions for emotion in last_two_emotions):
                if 'Potencial_Latente' in modulated_weights:
                    print("› DEBUG: Racha negativa detectada. Potenciando Tutor de Bienestar.")
                    modulated_weights['Potencial_Latente'] *= 1.5
        return modulated_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """Normaliza los pesos de los expertos para que sumen 1."""
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else weights

    def get_cognitive_plan(
        self, 
        user_profile: pd.Series, 
        emotion_probs: Dict,
        conversation_context: Dict,
        config: Dict,
        user_prompt: str
    ) -> Tuple[str, str]:
        """
        Genera el plan de acción final del tutor usando búsqueda semántica.
        """
        # 1. Predicción Cognitiva
        profile_df = pd.DataFrame([user_profile])
        profile_for_prediction = profile_df[self.feature_columns]
        predicted_archetype = self.cognitive_model.predict(profile_for_prediction)[0]

        # 2. Asignación de Pesos Base
        base_weights = {archetype: 0.0 for archetype in self.expert_map.keys()}
        if predicted_archetype in base_weights:
            base_weights[predicted_archetype] = 1.0

        # 3. Modulación Afectiva
        affective_weights = self._apply_affective_modulation(base_weights, emotion_probs)

        # 4. Modulación Contextual (Memoria)
        constants = config.get('constants', {})
        negative_emotions = constants.get('negative_emotions', [])
        conversational_weights = self._apply_conversational_modulation(
            affective_weights, conversation_context, negative_emotions
        )
        
        # 5. Normalización Final
        final_weights = self._normalize_weights(conversational_weights)
        
        # 6. Construcción del Plan de Recomendaciones
        sorted_plan = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)
        final_recs = []
        
        if user_profile.get('TIENE_CUD') != 'Si_Tiene_CUD':
            final_recs.append(GestorCUD().generate_recommendation(prompt=user_prompt))

        final_recs.append("**[Plan de Acción Adaptativo]**")
        
        min_rec_weight = self.thresholds.get('min_recommendation_weight', 0.15)
        recommendations_added = 0
        
        for archetype, weight in sorted_plan:
            if weight > min_rec_weight and archetype in self.expert_map:
                expert = self.expert_map[archetype]
                # ¡Pasamos el prompt al experto para que realice la búsqueda semántica!
                rec = expert.generate_recommendation(
                    prompt=user_prompt, 
                    original_profile=user_profile
                )
                final_recs.append(f"  - (Prioridad: {weight:.0%}): {rec}")
                recommendations_added += 1

        if recommendations_added == 0 and len(final_recs) <= 1: 
             final_recs.append("  - [Sistema]: No hay recomendaciones adicionales de tutores para esta consulta específica.")

        plan_string = "\n".join(final_recs)
        
        return plan_string, predicted_archetype

