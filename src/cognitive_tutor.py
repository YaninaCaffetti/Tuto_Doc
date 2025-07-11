# src/cognitive_tutor.py

import pandas as pd

class Experto:
    def __init__(self, nombre_experto): self.nombre = nombre_experto
    def generate_recommendation(self, **kwargs): return f"[{self.nombre}]: Recomendación genérica."

class GestorCUD(Experto):
    def __init__(self):super().__init__("Gestor de CUD")
    def generate_recommendation(self,**kwargs):return "[Acción CUD]: Se ha detectado que no posee el CUD. Se recomienda iniciar el trámite (Ley 22.431)."

class TutorCarrera(Experto):
    def __init__(self):super().__init__("Tutor de Estrategia de Carrera")
    def generate_recommendation(self,original_profile=None,**kwargs):
        rec=f"[{self.nombre}]: El plan es relanzar tu perfil profesional.";
        if original_profile is not None and original_profile.get('TIENE_CUD')=='Si_Tiene_CUD':rec+="\n  [Consejo Legal]: Dado que posees el CUD, postular a concursos del Estado es una estrategia efectiva (cupo 4%, Ley 22.431)."
        return rec

class TutorInteraccion(Experto):
    def __init__(self):super().__init__("Tutor de Habilidades de Interacción")
    def generate_recommendation(self,**kwargs):return f"[{self.nombre}]: Focalicémonos en desarrollar estrategias de comunicación adaptadas."

class TutorCompetencias(Experto):
    def __init__(self):super().__init__("Tutor de Competencias Técnicas")
    def generate_recommendation(self,**kwargs):return f"[{self.nombre}]: Sugiero que identifiquemos tus habilidades actuales y busquemos cursos cortos (Upskilling)."

class TutorBienestar(Experto):
    def __init__(self):super().__init__("Tutor de Bienestar y Activación")
    def generate_recommendation(self,**kwargs):return f"[{self.nombre}]: Nuestro primer paso es fortalecer tu autoconfianza e identificar intereses."

class TutorApoyos(Experto):
    def __init__(self):super().__init__("Tutor de Apoyos y Adaptaciones")
    def generate_recommendation(self,**kwargs):return f"[{self.nombre}]: Exploremos las adaptaciones que necesitas. La Ley 26.378 garantiza tu derecho a 'ajustes razonables'."

class TutorPrimerEmpleo(Experto):
    def __init__(self):super().__init__("Tutor de Primer Empleo")
    def generate_recommendation(self,**kwargs):return f"[{self.nombre}]: Te guiaré en la creación de tu primer CV y preparación para entrevistas."


EXPERT_MAP={'Com_Desafiado':TutorInteraccion(),'Nav_Informal':TutorCompetencias(),'Prof_Subutil':TutorCarrera(),'Potencial_Latente':TutorBienestar(),'Cand_Nec_Sig':TutorApoyos(),'Joven_Transicion':TutorPrimerEmpleo()}

class MoESystem:
    def __init__(self, cognitive_model, feature_columns, affective_rules):
        self.cognitive_model = cognitive_model
        self.feature_columns = feature_columns
        self.expert_map = EXPERT_MAP
        self.affective_rules = affective_rules

    def get_cognitive_plan(self, user_profile, emotion_probs: dict):
        profile_df = pd.DataFrame([user_profile])
        profile_for_prediction = profile_df[self.feature_columns]
        
        arquetipo_predominante = self.cognitive_model.predict(profile_for_prediction)[0]
        
        expert_weights = { arquetipo: 0.0 for arquetipo in self.expert_map.keys() }
        if arquetipo_predominante in expert_weights:
            expert_weights[arquetipo_predominante] = 1.0

        final_recs = []
        if user_profile.get('TIENE_CUD') != 'Si_Tiene_CUD':
            cud_rec = GestorCUD().generate_recommendation()
            if 'Ira' in emotion_probs and emotion_probs['Ira'] > 0.5:
                cud_rec += "\n  [Consejo Adicional]: Entendemos tu frustración. Iniciar este trámite puede ser un paso concreto para resolver la situación."
            final_recs.append(cud_rec)

        final_recs.append(f"**[Adaptación Afectiva Difusa Activada]**")
        modulation_factors = {arq: 1.0 for arq in self.expert_map.keys()}
        
        for emotion, prob in emotion_probs.items():
            if prob > 0.1 and emotion in self.affective_rules:
                rules = self.affective_rules[emotion]
                for arquetipo, factor in rules.items():
                    modulation_factors[arquetipo] *= (1 + (factor - 1) * prob)
        
        for arquetipo, factor in modulation_factors.items():
            if arquetipo in expert_weights:
                expert_weights[arquetipo] *= factor
        
        total_weight = sum(expert_weights.values())
        if total_weight > 0:
            normalized_weights = {key: value / total_weight for key, value in expert_weights.items()}
        else:
            normalized_weights = expert_weights

        sorted_plan = sorted(normalized_weights.items(), key=lambda item: item[1], reverse=True)

        for arquetipo, peso in sorted_plan:
            if peso > 0.15:
                experto = self.expert_map.get(arquetipo)
                if experto:
                    rec = experto.generate_recommendation(original_profile=user_profile)
                    final_recs.append(f"  - (Prioridad: {peso:.0%}): {rec}")
        
        if len(final_recs) == 0 or (len(final_recs) == 1 and final_recs[0].startswith("[Acción CUD]")):
             final_recs.append("[Sistema]: No se pudo determinar un plan de acción de tutores para este perfil.")

        return "\n".join(final_recs)
