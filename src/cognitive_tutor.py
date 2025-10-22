"""
Define la arquitectura del Tutor Cognitivo, incluyendo el Sistema de Mezcla
de Expertos (MoE) y las clases de Tutores Expertos individuales.

Esta versión incluye:
- Bases de conocimiento (KB) para cada experto para dar respuestas específicas.
- Lógica de modulación basada en el contexto (memoria) de la conversación.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any

# --- 1. ARQUITECTURA DE EXPERTOS CONVERSACIONALES ---

class Experto:
    """Clase base para tutores con una base de conocimiento interna."""
    def __init__(self, nombre_experto: str):
        self.nombre = nombre_experto
        # Cada experto tiene su propia base de conocimiento (KB)
        self.knowledge_base: Dict[str, str] = {
            # Respuesta genérica si no se encuentran palabras clave
            "default": f"Analicemos tu situación general. Mi consejo principal es sobre {self.nombre.lower()}."
        }

    def generate_recommendation(self, prompt: str, **kwargs) -> str:
        """
        Genera una recomendación buscando palabras clave del prompt en su KB.

        Args:
            prompt: El texto escrito por el usuario.
            **kwargs: Argumentos adicionales (ej. original_profile).

        Returns:
            La respuesta específica del experto o una respuesta por defecto.
        """
        prompt_lower = prompt.lower()
        # Busca una respuesta específica basada en palabras clave
        for keyword, response in self.knowledge_base.items():
            if keyword != "default" and keyword in prompt_lower:
                return f"[{self.nombre}]: {response}"
        
        # Si no hay coincidencia, devuelve la respuesta genérica
        return f"[{self.nombre}]: {self.knowledge_base['default']}"

class GestorCUD(Experto):
    """Experto en la gestión del Certificado Único de Discapacidad."""
    def __init__(self): 
        super().__init__("Gestor de CUD")
    
    def generate_recommendation(self, **kwargs) -> str: 
        # Este experto tiene una recomendación fija y no depende del prompt
        return "[Acción CUD]: Se ha detectado que no posee el CUD. Se recomienda iniciar el trámite (Ley 22.431)."

class TutorCarrera(Experto):
    """Experto en estrategia profesional, CV, entrevistas y negociación."""
    def __init__(self):
        super().__init__("Tutor de Estrategia de Carrera")
        self.knowledge_base = {
            "cv": "Para tu CV, enfócate en logros cuantificables, no solo en responsabilidades. Por ejemplo, 'optimicé el proceso X, reduciendo el tiempo en un 15%'.",
            "entrevista": "Durante una entrevista, prepara respuestas para 'háblame de ti' usando la estructura 'Presente-Pasado-Futuro'. Es muy efectiva.",
            "linkedin": "Tu perfil de LinkedIn debe tener una foto profesional y un titular que describa el valor que aportas, no solo tu puesto actual.",
            "salario": "Al negociar el salario, investiga el rango de mercado. Nunca des la primera cifra, pregunta por el presupuesto que manejan para el rol.",
            "crecer": "Para crecer profesionalmente, identifica a un mentor en tu campo y busca proyectos que te saquen de tu zona de confort.",
            "default": "El plan es relanzar tu perfil profesional, identificando tus fortalezas clave y alineándolas con las demandas del mercado."
        }
    
    def generate_recommendation(self, prompt: str, **kwargs) -> str:
        """Genera la recomendación y añade consejos legales si aplica."""
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
            "comunicar": "Una técnica clave es la 'escucha activa': reformula lo que dice la otra persona para asegurar que has entendido. Genera mucha confianza.",
            "reuniones": "En reuniones, si te cuesta intervenir, prepara una o dos preguntas de antemano. Es una forma fácil de participar.",
            "conflicto": "Para manejar un conflicto, enfócate en el problema, no en la persona. Usa frases como 'Cuando ocurre X, siento Y'.",
            "feedback": "Al dar feedback, usa el método 'sandwich': empieza con algo positivo, luego el área de mejora, y cierra con otra nota positiva.",
            "presentar": "Al presentar en público, estructura tu discurso con una introducción clara (el problema), un desarrollo (tu solución) y una conclusión (el llamado a la acción).",
            "default": "Focalicémonos en desarrollar estrategias de comunicación adaptadas a tu perfil y objetivos."
        }

class TutorCompetencias(Experto):
    """Experto en 'upskilling', cursos y aprendizaje de nuevas tecnologías."""
    def __init__(self):
        super().__init__("Tutor de Competencias Técnicas")
        self.knowledge_base = {
            "cursos": "Plataformas como Coursera, edX o Platzi ofrecen rutas de aprendizaje de alta calidad, no solo cursos aislados.",
            "certificación": "Una certificación oficial (ej. de Google, Microsoft, AWS) puede validar tus habilidades y dar un gran impulso a tu CV.",
            "tecnología": "Para mantenerte al día, te recomiendo seguir newsletters especializados en tu área y dedicar 30 minutos al día a leer sobre nuevas tendencias.",
            "aprender": "La mejor forma de aprender una nueva tecnología es construir un pequeño proyecto con ella. La práctica es clave.",
            "habilidades": "Además de lo técnico, las 'power skills' como la resolución de problemas complejos y el pensamiento crítico son muy demandadas.",
            "default": "Sugiero que identifiquemos tus habilidades actuales y busquemos cursos cortos y de alto impacto (Upskilling)."
        }

class TutorBienestar(Experto):
    """Experto en confianza, manejo del estrés y motivación."""
    def __init__(self):
        super().__init__("Tutor de Bienestar y Activación")
        self.knowledge_base = {
            "estrés": "Para manejar el estrés, prueba la técnica 'Pomodoro': 25 minutos de trabajo enfocado y 5 de descanso. Ayuda a mantener la mente fresca.",
            "ansiedad": "Frente a la ansiedad, practica la 'respiración de caja': inhala contando 4, sostén 4, exhala 4, espera 4. Repite.",
            "motivación": "Para recuperar la motivación, divide una tarea grande en pasos muy pequeños y celebra cada vez que completas uno.",
            "confianza": "La autoconfianza se construye con pequeñas victorias. Fija un objetivo muy pequeño para hoy y cúmplelo.",
            "miedo": "El miedo al fracaso es normal. Intenta reencuadrarlo: cada error es un dato valioso sobre cómo no hacer algo.",
            "default": "Nuestro primer paso es fortalecer tu autoconfianza e identificar tus intereses principales."
        }

class TutorApoyos(Experto):
    """Experto en adaptaciones, derechos y 'ajustes razonables'."""
    def __init__(self):
        super().__init__("Tutor de Apoyos y Adaptaciones")
        self.knowledge_base = {
            "ajustes": "Los 'ajustes razonables' son modificaciones que no imponen una carga desproporcionada al empleador. Tienes derecho a solicitarlos (Ley 26.378).",
            "derechos": "Tu derecho principal es la no discriminación. Esto incluye el acceso a entrevistas y adaptaciones en el puesto de trabajo.",
            "solicitar": "Al solicitar una adaptación, sé específico: 'Necesito un software de lectura de pantalla para realizar mis tareas de análisis' es mejor que 'necesito ayuda'.",
            "transporte": "Recuerda que el CUD te da derecho al 'Pase Libre' en transporte público nacional.",
            "accesibilidad": "La accesibilidad web (ej. sitios que funcionen con lectores de pantalla) es un derecho. Herramientas como 'WAVE' pueden ayudarte a evaluar sitios.",
            "default": "Exploremos juntos las adaptaciones y tecnologías de apoyo que necesitas. La Ley 26.378 garantiza tu derecho a 'ajustes razonables'."
        }

class TutorPrimerEmpleo(Experto):
    """Experto en guiar a jóvenes en su primera experiencia laboral."""
    def __init__(self):
        super().__init__("Tutor de Primer Empleo")
        self.knowledge_base = {
            "cv": "Para tu primer CV, si no tienes experiencia laboral, enfócate en proyectos académicos, voluntariados y habilidades que hayas desarrollado.",
            "entrevista": "En tu primera entrevista, demuestra entusiasmo y ganas de aprender. Es más importante tu actitud que tu experiencia previa.",
            "buscar": "Para buscar tu primer empleo, activa las alertas en LinkedIn y Bumeran, y no subestimes el poder de las redes de contactos de tu universidad.",
            "pasantía": "Una pasantía es una excelente forma de ganar experiencia. Búcalas activamente, ya que son la mejor puerta de entrada.",
            "habilidades": "Destaca habilidades blandas: trabajo en equipo, adaptabilidad y buena comunicación. A menudo son más valoradas que las técnicas en un perfil junior.",
            "default": "Te guiaré en la creación de tu primer CV y en cómo prepararte para las entrevistas. ¡Conseguir el primer empleo es un hito importante!"
        }

# Mapeo central de Arquetipos a sus Tutores Expertos
EXPERT_MAP = {
    'Com_Desafiado': TutorInteraccion(),
    'Nav_Informal': TutorCompetencias(),
    'Prof_Subutil': TutorCarrera(),
    'Potencial_Latente': TutorBienestar(),
    'Cand_Nec_Sig': TutorApoyos(),
    'Joven_Transicion': TutorPrimerEmpleo()
}

# --- 2. SISTEMA DE ORQUESTACIÓN MoE (ACTUALIZADO) ---

class MoESystem:
    """Sistema de Mezcla de Expertos (MoE) que orquesta la respuesta del tutor.
    
    Integra el análisis cognitivo (arquetipo), afectivo (emoción) y
    contextual (memoria de chat) para seleccionar y modular las respuestas
    de una red de tutores expertos.
    """
    def __init__(self, cognitive_model, feature_columns: List[str], 
                 affective_rules: Dict, thresholds: Dict):
        """Inicializa el sistema MoE.

        Args:
            cognitive_model: El clasificador de arquetipos (RandomForest) entrenado.
            feature_columns: Lista de columnas de características (las '_memb').
            affective_rules: Diccionario de reglas para modular por emoción.
            thresholds: Diccionario con umbrales de activación del sistema.
        """
        self.cognitive_model = cognitive_model
        self.feature_columns = feature_columns
        self.expert_map = EXPERT_MAP
        self.affective_rules = affective_rules
        # Asegura que los thresholds se lean de la sub-clave correcta si existe
        self.thresholds = thresholds.get('affective_engine', thresholds)

    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        """Aplica la modulación afectiva a los pesos base de los expertos."""
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated_weights = base_weights.copy()
        
        for emotion, prob in emotion_probs.items():
            # Normalizar emoción (quitar espacios, capitalizar) por si acaso
            emotion_norm = str(emotion).strip().capitalize()
            if prob > min_prob and emotion_norm in self.affective_rules:
                rules = self.affective_rules[emotion_norm]
                for archetype, factor in rules.items():
                    if archetype in modulated_weights:
                        # Aplicar modulación difusa (escalada por la probabilidad)
                        modulated_weights[archetype] *= (1 + (factor - 1) * prob)
        return modulated_weights
        
    def _apply_conversational_modulation(self, weights: Dict, context: Dict, 
                                         negative_emotions: List[str]) -> Dict:
        """Aplica modulación basada en la memoria de la conversación."""
        modulated_weights = weights.copy()
        emotional_trajectory = context.get("emotional_trajectory", [])
        
        # Regla de ejemplo: Si las últimas 2 emociones fueron negativas,
        # potenciar al Tutor de Bienestar.
        if len(emotional_trajectory) >= 2:
            last_two_emotions = emotional_trajectory[-2:]
            if all(emotion in negative_emotions for emotion in last_two_emotions):
                if 'Potencial_Latente' in modulated_weights:
                    print("› DEBUG: Racha negativa detectada. Potenciando Tutor de Bienestar.")
                    modulated_weights['Potencial_Latente'] *= 1.5 # Boost de 50%
        return modulated_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """Normaliza los pesos de los expertos para que sumen 1."""
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else weights

    # --- ¡FIRMA DE FUNCIÓN ACTUALIZADA! ---
    def get_cognitive_plan(
        self, 
        user_profile: pd.Series, 
        emotion_probs: Dict,
        conversation_context: Dict,
        config: Dict,
        user_prompt: str
    ) -> Tuple[str, str]:
        """
        Genera el plan de acción final del tutor.

        Args:
            user_profile: Serie de Pandas con los datos del perfil de demo.
            emotion_probs: Diccionario de {emoción: probabilidad} del clasificador.
            conversation_context: Diccionario con el historial de la sesión.
            config: El diccionario de configuración global.
            user_prompt: El texto exacto que escribió el usuario.

        Returns:
            Una tupla (plan_de_accion_string, arquetipo_predicho).
        """
        # 1. Predicción Cognitiva
        profile_df = pd.DataFrame([user_profile])
        # Asegurarse de que las columnas están en el orden correcto
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
        
        # Regla Fija: GestorCUD siempre se activa si no tiene CUD
        if user_profile.get('TIENE_CUD') != 'Si_Tiene_CUD':
            final_recs.append(GestorCUD().generate_recommendation())

        final_recs.append("**[Plan de Acción Adaptativo]**")
        
        min_rec_weight = self.thresholds.get('min_recommendation_weight', 0.15)
        recommendations_added = 0
        
        for archetype, weight in sorted_plan:
            if weight > min_rec_weight and archetype in self.expert_map:
                expert = self.expert_map[archetype]
                # ¡Pasamos el prompt al experto para que busque en su KB!
                rec = expert.generate_recommendation(
                    prompt=user_prompt, 
                    original_profile=user_profile
                )
                final_recs.append(f"  - (Prioridad: {weight:.0%}): {rec}")
                recommendations_added += 1

        if recommendations_added == 0 and len(final_recs) <= 1: # Si solo está el título (o CUD + título)
             final_recs.append("  - [Sistema]: No hay recomendaciones adicionales de tutores para esta consulta específica.")


        plan_string = "\n".join(final_recs)
        
        return plan_string, predicted_archetype

