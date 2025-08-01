# src/cognitive_tutor.py (Versión Pulida y Documentada)

import pandas as pd
import warnings

# --- 1. DEFINICIÓN DE LAS CLASES DE EXPERTOS ---

class Experto:
    """Clase base abstracta para todos los tutores expertos."""
    def __init__(self, nombre_experto: str):
        """
        Inicializa un experto con un nombre específico.

        Args:
            nombre_experto (str): El nombre identificador del tutor experto.
        """
        self.nombre = nombre_experto
    
    def generate_recommendation(self, **kwargs) -> str:
        """
        Genera una recomendación genérica. Debe ser sobreescrita por las clases hijas.

        Returns:
            str: Un texto con la recomendación.
        """
        return f"[{self.nombre}]: Recomendación genérica."

class GestorCUD(Experto):
    """Experto especializado en el Certificado Único de Discapacidad (CUD)."""
    def __init__(self):
        super().__init__("Gestor de CUD")
    
    def generate_recommendation(self, **kwargs) -> str:
        """Genera una recomendación para iniciar el trámite del CUD."""
        return "[Acción CUD]: Se ha detectado que no posee el CUD. Se recomienda iniciar el trámite (Ley 22.431)."

class TutorCarrera(Experto):
    """Tutor experto en estrategia de carrera y reinserción profesional."""
    def __init__(self):
        super().__init__("Tutor de Estrategia de Carrera")
        
    def generate_recommendation(self, original_profile: pd.Series = None, **kwargs) -> str:
        """
        Genera un plan para relanzar un perfil profesional, con un consejo
        específico si el usuario ya posee el CUD.
        """
        rec = f"[{self.nombre}]: El plan es relanzar tu perfil profesional."
        if original_profile is not None and original_profile.get('TIENE_CUD') == 'Si_Tiene_CUD':
            rec += "\n  [Consejo Legal]: Dado que posees el CUD, postular a concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)."
        return rec

class TutorInteraccion(Experto):
    """Tutor experto en el desarrollo de habilidades de comunicación e interacción."""
    def __init__(self):
        super().__init__("Tutor de Habilidades de Interacción")
        
    def generate_recommendation(self, **kwargs) -> str:
        """Recomienda enfocarse en estrategias de comunicación."""
        return f"[{self.nombre}]: Focalicémonos en desarrollar estrategias de comunicación adaptadas."

class TutorCompetencias(Experto):
    """Tutor experto en la actualización de habilidades técnicas (Upskilling)."""
    def __init__(self):
        super().__init__("Tutor de Competencias Técnicas")
        
    def generate_recommendation(self, **kwargs) -> str:
        """Sugiere la identificación de habilidades y la búsqueda de cursos cortos."""
        return f"[{self.nombre}]: Sugiero que identifiquemos tus habilidades actuales y busquemos cursos cortos (Upskilling)."

class TutorBienestar(Experto):
    """Tutor experto en el fortalecimiento de la autoconfianza y la motivación."""
    def __init__(self):
        super().__init__("Tutor de Bienestar y Activación")
        
    def generate_recommendation(self, **kwargs) -> str:
        """Propone trabajar en la autoconfianza como primer paso."""
        return f"[{self.nombre}]: Nuestro primer paso es fortalecer tu autoconfianza e identificar intereses."

class TutorApoyos(Experto):
    """Tutor experto en la identificación de apoyos y adaptaciones necesarias."""
    def __init__(self):
        super().__init__("Tutor de Apoyos y Adaptaciones")
        
    def generate_recommendation(self, **kwargs) -> str:
        """Recomienda explorar 'ajustes razonables' garantizados por ley."""
        return f"[{self.nombre}]: Exploremos las adaptaciones que necesitas. La Ley 26.378 garantiza tu derecho a 'ajustes razonables'."

class TutorPrimerEmpleo(Experto):
    """Tutor experto en guiar a los usuarios en la búsqueda de su primer empleo."""
    def __init__(self):
        super().__init__("Tutor de Primer Empleo")
        
    def generate_recommendation(self, **kwargs) -> str:
        """Ofrece guía para la creación de CV y preparación de entrevistas."""
        return f"[{self.nombre}]: Te guiaré en la creación de tu primer CV y preparación para entrevistas."

# --- 2. MAPEO DE ARQUETIPOS A EXPERTOS ---

# Diccionario que conecta cada arquetipo predicho por el modelo cognitivo
# con una instancia de la clase de experto correspondiente.
EXPERT_MAP = {
    'Com_Desafiado': TutorInteraccion(),
    'Nav_Informal': TutorCompetencias(),
    'Prof_Subutil': TutorCarrera(),
    'Potencial_Latente': TutorBienestar(),
    'Cand_Nec_Sig': TutorApoyos(),
    'Joven_Transicion': TutorPrimerEmpleo()
}

# --- 3. SISTEMA DE ORQUESTACIÓN (Mixture of Experts) ---

class MoESystem:
    """
    Sistema de Mezcla de Expertos (Mixture of Experts) que integra el análisis
    cognitivo y afectivo para generar un plan de acción adaptativo.
    """
    def __init__(self, cognitive_model, feature_columns: list, affective_rules: dict):
        """
        Inicializa el sistema MoE.

        Args:
            cognitive_model: El modelo de clasificación (ej. RandomForest) ya entrenado.
            feature_columns (list): La lista de nombres de columnas que el modelo cognitivo espera.
            affective_rules (dict): El diccionario de reglas que mapea emociones a factores de modulación.
        """
        self.cognitive_model = cognitive_model
        self.feature_columns = feature_columns
        self.expert_map = EXPERT_MAP
        self.affective_rules = affective_rules

    def get_cognitive_plan(self, user_profile: pd.Series, emotion_probs: dict) -> str:
        """
        Genera el plan de acción final, modulado por la emoción.

        Este es el método principal del sistema. Predice el arquetipo del usuario,
        selecciona un experto principal, y luego usa el vector de probabilidades
        de emoción para modular las recomendaciones, generando un plan final priorizado.

        Args:
            user_profile (pd.Series): Una fila del DataFrame que representa el perfil de un usuario.
            emotion_probs (dict): Un diccionario con las probabilidades de cada emoción.

        Returns:
            str: Un texto formateado con el plan de acción completo y adaptativo.
        """
        # A. Predicción Cognitiva
        profile_df = pd.DataFrame([user_profile])
        profile_for_prediction = profile_df[self.feature_columns]
        predicted_archetype = self.cognitive_model.predict(profile_for_prediction)[0]
        
        # B. Inicialización de Pesos de Expertos
        expert_weights = {archetype: 0.0 for archetype in self.expert_map.keys()}
        if predicted_archetype in expert_weights:
            expert_weights[predicted_archetype] = 1.0

        # C. Lógica de Recomendaciones Fijas (no modulables)
        final_recs = []
        if user_profile.get('TIENE_CUD') != 'Si_Tiene_CUD':
            final_recs.append(GestorCUD().generate_recommendation())

        # D. Modulación Afectiva Difusa
        final_recs.append("**[Adaptación Afectiva Difusa Activada]**")
        modulation_factors = {archetype: 1.0 for archetype in self.expert_map.keys()}
        
        for emotion, prob in emotion_probs.items():
            if prob > 0.1 and emotion in self.affective_rules:
                rules = self.affective_rules[emotion]
                for archetype_from_rule, factor in rules.items():
                    # Solo se modula si el arquetipo de la regla es un experto válido.
                    if archetype_from_rule in modulation_factors:
                        modulation_factors[archetype_from_rule] *= (1 + (factor - 1) * prob)
                    else:
                        warnings.warn(f"Regla afectiva para '{archetype_from_rule}' ignorada: no es un arquetipo válido.")
        
        # Aplicar los factores de modulación a los pesos iniciales
        for archetype, factor in modulation_factors.items():
            if archetype in expert_weights:
                expert_weights[archetype] *= factor
        
        # E. Normalización y Priorización del Plan Final
        total_weight = sum(expert_weights.values())
        normalized_weights = expert_weights
        if total_weight > 0:
            normalized_weights = {key: value / total_weight for key, value in expert_weights.items()}

        # Ordenar los expertos por su peso final
        sorted_plan = sorted(normalized_weights.items(), key=lambda item: item[1], reverse=True)

        # F. Construcción del Texto de Salida
        for archetype, weight in sorted_plan:
            if weight > 0.15: # Umbral de relevancia
                expert = self.expert_map.get(archetype)
                if expert:
                    rec = expert.generate_recommendation(original_profile=user_profile)
                    final_recs.append(f"  - (Prioridad: {weight:.0%}): {rec}")
        
        if len(final_recs) <= 1:
             final_recs.append("[Sistema]: No se pudo determinar un plan de acción de tutores adicional para este perfil.")

        return "\n".join(final_recs)
