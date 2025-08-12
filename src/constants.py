# src/constants.py

# Este archivo centraliza las constantes del proyecto y facilitar el mantenimiento del código.

# --- Nombres de Arquetipos ---
# Usados en data_processing, cognitive_tutor y config.yaml
ARCHETYPE_COM_DESAFIADO = 'Com_Desafiado'
ARCHETYPE_NAV_INFORMAL = 'Nav_Informal'
ARCHETYPE_PROF_SUBUTIL = 'Prof_Subutil'
ARCHETYPE_POTENCIAL_LATENTE = 'Potencial_Latente'
ARCHETYPE_CAND_NEC_SIG = 'Cand_Nec_Sig'
ARCHETYPE_JOVEN_TRANSICION = 'Joven_Transicion'

# Lista ordenada de arquetipos para consistencia
ALL_ARCHETYPES = [
    ARCHETYPE_COM_DESAFIADO,
    ARCHETYPE_NAV_INFORMAL,
    ARCHETYPE_PROF_SUBUTIL,
    ARCHETYPE_POTENCIAL_LATENTE,
    ARCHETYPE_CAND_NEC_SIG,
    ARCHETYPE_JOVEN_TRANSICION,
]

# --- Columnas de Datos ---
# Columna objetivo final que se creará en el preprocesamiento
TARGET_COLUMN = 'ARQUETIPO_PRED'
