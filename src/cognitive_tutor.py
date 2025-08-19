# src/cognitive_tutor.py (Versión Refactorizada y Optimizada)

import pandas as pd
import warnings
from typing import Dict, List, Tuple

# --- 1. DEFINICIÓN DE LAS CLASES DE EXPERTOS ---

class Experto:
    """Clase base abstracta para todos los tutores expertos."""
    def __init__(self, nombre_experto: str):
        self.nombre = nombre_experto
    def generate_recommendation(self, **kwargs) -> str:
        return f"[{self.nombre}]: Recomendación genérica."

class GestorCUD(Experto):
    def __init__(self): super().__init__("Gestor de CUD")
    def generate_recommendation(self, **kwargs) -> str:
        return "[Acción CUD]: Se ha detectado que no posee el CUD. Se recomienda iniciar el trámite (Ley 22.431)."

class TutorCarrera(Experto):
    def __init__(self): super().__init__("Tutor de Estrategia de Carrera")
    def generate_recommendation(self, original_profile: pd.Series = None, **kwargs) -> str:
        rec = f"[{self.nombre}]: El plan es relanzar tu perfil profesional."
        if original_profile is not None and original_profile.get('TIENE_CUD') == 'Si_Tiene_CUD':
            rec += "\n  [Consejo Legal]: Dado que posees el CUD, postular a concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)."
        return rec

class TutorInteraccion(Experto):
    def __init__(self): super().__init__("Tutor de Habilidades de Interacción")
    def generate_recommendation(self, **kwargs) -> str:
        return f"[{self.nombre}]: Focalicémonos en desarrollar estrategias de comunicación adaptadas."

class TutorCompetencias(Experto):
    def __init__(self): super().__init__("Tutor de Competencias Técnicas")
    def generate_recommendation(self, **kwargs) -> str:
        return f"[{self.nombre}]: Sugiero que identifiquemos tus habilidades actuales y busquemos cursos cortos (Upskilling)."

class TutorBienestar(Experto):
    def __init__(self): super().__init__("Tutor de Bienestar y Activación")
    def generate_recommendation(self, **kwargs) -> str:
        return f"[{self.nombre}]: Nuestro primer paso es fortalecer tu autoconfianza e identificar intereses."

class TutorApoyos(Experto):
    def __init__(self): super().__init__("Tutor de Apoyos y Adaptaciones")
    def generate_recommendation(self, **kwargs) -> str:
        return f"[{self.nombre}]: Exploremos las adaptaciones que necesitas. La Ley 26.378 garantiza tu derecho a 'ajustes razonables'."

class TutorPrimerEmpleo(Experto):
    def __init__(self): super().__init__("Tutor de Primer Empleo")
    def generate_recommendation(self, **kwargs) -> str:
        return f"[{self.nombre}]: Te guiaré en la creación de tu primer CV y preparación para entrevistas."

# --- 2. MAPEO DE ARQUETIPOS A EXPERTOS (Sin cambios) ---
EXPERT_MAP = {
    'Com_Desafiado': TutorInteraccion(), 'Nav_Informal': TutorCompetencias(),
    'Prof_Subutil': TutorCarrera(), 'Potencial_Latente': TutorBienestar(),
    'Cand_Nec_Sig': TutorApoyos(), 'Joven_Transicion': TutorPrimerEmpleo()
}

# --- 3. SISTEMA DE ORQUESTACIÓN (Mixture of Experts) ---

class MoESystem:
    """
    Sistema de Mezcla de Expertos (MoE) que integra el análisis
    cognitivo y afectivo para generar un plan de acción adaptativo.
    """
    def __init__(self, cognitive_model, feature_columns: list, affective_rules: dict, thresholds: dict):
        """
        Inicializa el sistema MoE.
        
        Args:
            cognitive_model: El modelo cognitivo entrenado (RandomForestClassifier).
            feature_columns (list): Lista de columnas usadas para la predicción.
            affective_rules (dict): Reglas para modular pesos según la emoción.
            thresholds (dict): Umbrales de configuración del sistema (ej. probabilidad mínima).
        """
        self.cognitive_model = cognitive_model
        self.feature_columns = feature_columns
        self.expert_map = EXPERT_MAP
        self.affective_rules = affective_rules
        self.thresholds = thresholds

    def _apply_affective_modulation(self, base_weights: Dict, emotion_probs: Dict) -> Dict:
        """Aplica la modulación afectiva a los pesos base de los expertos."""
        min_prob = self.thresholds.get('min_emotion_probability', 0.1)
        modulated_weights = base_weights.copy()
        
        for emotion, prob in emotion_probs.items():
            if prob > min_prob and emotion in self.affective_rules:
                rules = self.affective_rules[emotion]
                for archetype, factor in rules.items():
                    if archetype in modulated_weights:
                        modulated_weights[archetype] *= (1 + (factor - 1) * prob)
        return modulated_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """Normaliza los pesos de los expertos para que sumen 1."""
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {key: value / total_weight for key, value in weights.items()}
        return weights

    def get_cognitive_plan(self, user_profile: pd.Series, emotion_probs: Dict) -> Tuple[str, str]:
        """
        Genera el plan de acción final y devuelve el arquetipo predicho.

        Args:
            user_profile (pd.Series): Perfil de usuario completo.
            emotion_probs (dict): Distribución de probabilidad emocional del usuario.

        Returns:
            Tuple[str, str]: Una tupla conteniendo (plan_de_accion, arquetipo_predicho).
        """
        # 1. Predecir el arquetipo cognitivo. 
        profile_df = pd.DataFrame([user_profile])
        profile_for_prediction = profile_df[self.feature_columns]
        predicted_archetype = self.cognitive_model.predict(profile_for_prediction)[0]

        # 2. Establecer el peso base: 1.0 para el experto principal, 0.0 para los demás.
        base_weights = {archetype: 0.0 for archetype in self.expert_map.keys()}
        if predicted_archetype in base_weights:
            base_weights[predicted_archetype] = 1.0

        # 3. Aplicar la modulación afectiva difusa.
        modulated_weights = self._apply_affective_modulation(base_weights, emotion_probs)

        # 4. Normalizar los pesos finales.
        final_weights = self._normalize_weights(modulated_weights)
        
        # 5. Construir el plan de recomendaciones.
        sorted_plan = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)
        
        final_recs = []
        if user_profile.get('TIENE_CUD') != 'Si_Tiene_CUD':
            final_recs.append(GestorCUD().generate_recommendation())

        final_recs.append("**[Adaptación Afectiva Difusa Activada]**")
        
        min_rec_weight = self.thresholds.get('min_recommendation_weight', 0.15)
        recommendations_added = 0
        for archetype, weight in sorted_plan:
            if weight > min_rec_weight:
                expert = self.expert_map.get(archetype)
                if expert:
                    rec = expert.generate_recommendation(original_profile=user_profile)
                    final_recs.append(f"  - (Prioridad: {weight:.0%}): {rec}")
                    recommendations_added += 1

        if recommendations_added == 0:
            final_recs.append("[Sistema]: No se pudo determinar un plan de acción de tutores adicional para este perfil.")

        plan_string = "\n".join(final_recs)
        
        # 6. Devolver tanto el plan como el arquetipo para que la app los use.
        return plan_string, predicted_archetype
