# tests/test_data_processing.py

import pandas as pd
import numpy as np
import pytest
import sys
import os

# --- Configuración del Path ---
# Ajusta la ruta si tu estructura de carpetas es diferente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# --- Módulos a Probar ---
from data_processing import run_archetype_engineering
from constants import ARCHETYPE_PROF_SUBUTIL, ARCHETYPE_JOVEN_TRANSICION

# --- Fixtures de Datos de Prueba ---

@pytest.fixture
def profile_subutilizado_ok() -> pd.DataFrame:
    """Perfil ideal para 'Profesional Subutilizado' (debe dar score alto)."""
    data = {
        'CAPITAL_HUMANO': ['3_Alto'], 'Perfil_Dificultad_Agrupado': ['1A_Motora_Unica'],
        'Espectro_Inclusion_Laboral': ['2_Busqueda_Sin_Exito'], 'GRUPO_ETARIO_INDEC': ['2_Adulto_Medio (40-64)'],
        'tipo_hogar': [2], 'pc03': [1], 'TIENE_CUD': ['Si_Tiene_CUD']
    }
    return pd.DataFrame(data)

@pytest.fixture
def profile_subutilizado_mal() -> pd.DataFrame:
    """Perfil no relacionado con 'Profesional Subutilizado' (debe dar score bajo)."""
    data = {
        'CAPITAL_HUMANO': ['1_Bajo'], 'Perfil_Dificultad_Agrupado': ['0_Sin_Dificultad_Registrada'],
        'Espectro_Inclusion_Laboral': ['4_Inclusion_Plena_Aprox'], 'GRUPO_ETARIO_INDEC': ['1_Joven_Adulto_Temprano (14-39)'],
        'tipo_hogar': [1], 'pc03': [9], 'TIENE_CUD': ['No_Tiene_CUD']
    }
    return pd.DataFrame(data)

@pytest.fixture
def profile_joven_transicion_ok() -> pd.DataFrame:
    """Perfil ideal para 'Joven en Transición' (debe dar score alto)."""
    data = {
        'CAPITAL_HUMANO': ['2_Medio'], 'Perfil_Dificultad_Agrupado': ['1B_Visual_Unica'],
        'Espectro_Inclusion_Laboral': ['1_Exclusion_del_Mercado'], 'GRUPO_ETARIO_INDEC': ['1_Joven_Adulto_Temprano (14-39)'],
        'PC08': [1], 'tipo_hogar': [3], 'pc03': [2], 'TIENE_CUD': ['No_Tiene_CUD']
    }
    return pd.DataFrame(data)

@pytest.fixture
def profile_datos_faltantes() -> pd.DataFrame:
    """Perfil con datos faltantes para probar robustez."""
    data = {
        'CAPITAL_HUMANO': [None], 'Perfil_Dificultad_Agrupado': ['1A_Motora_Unica'],
        'Espectro_Inclusion_Laboral': [None], 'GRUPO_ETARIO_INDEC': ['2_Adulto_Medio (40-64)'],
        'PC08': [np.nan], 'tipo_hogar': [2], 'pc03': [1], 'TIENE_CUD': ['Si_Tiene_CUD']
    }
    return pd.DataFrame(data)


# --- Pruebas Unitarias ---

def test_profesional_subutilizado_score_alto(profile_subutilizado_ok: pd.DataFrame):
    """Prueba el caso feliz: un perfil de 'Profesional Subutilizado' debe tener un score alto."""
    processed_df = run_archetype_engineering(profile_subutilizado_ok)
    columna_pertenencia = f'Pertenencia_{ARCHETYPE_PROF_SUBUTIL}'
    
    assert columna_pertenencia in processed_df.columns
    pertenencia_score = processed_df[columna_pertenencia].iloc[0]
    assert 0.7 < pertenencia_score <= 1.0, "El score para un perfil ideal debería ser alto."

def test_profesional_subutilizado_score_bajo(profile_subutilizado_mal: pd.DataFrame):
    """Prueba el caso negativo: un perfil no relacionado debe tener un score bajo."""
    processed_df = run_archetype_engineering(profile_subutilizado_mal)
    columna_pertenencia = f'Pertenencia_{ARCHETYPE_PROF_SUBUTIL}'
    
    assert columna_pertenencia in processed_df.columns
    pertenencia_score = processed_df[columna_pertenencia].iloc[0]
    assert 0.0 <= pertenencia_score < 0.3, "El score para un perfil no relacionado debería ser bajo."

def test_joven_transicion_score_alto(profile_joven_transicion_ok: pd.DataFrame):
    """Prueba un arquetipo diferente: 'Joven en Transición'."""
    processed_df = run_archetype_engineering(profile_joven_transicion_ok)
    columna_pertenencia = f'Pertenencia_{ARCHETYPE_JOVEN_TRANSICION}'
    
    assert columna_pertenencia in processed_df.columns
    pertenencia_score = processed_df[columna_pertenencia].iloc[0]
    assert 0.7 < pertenencia_score <= 1.0, "El score para un perfil ideal de joven en transición debería ser alto."

def test_manejo_de_datos_faltantes(profile_datos_faltantes: pd.DataFrame):
    """Prueba de robustez: el script no debe fallar con datos faltantes."""
    try:
        processed_df = run_archetype_engineering(profile_datos_faltantes)
        # La prueba pasa si la función se ejecuta sin errores.
        # Adicionalmente, verificamos que todas las columnas de pertenencia se hayan creado.
        for archetype in ALL_ARCHETYPES:
            assert f'Pertenencia_{archetype}' in processed_df.columns
    except Exception as e:
        pytest.fail(f"La función run_archetype_engineering falló con datos faltantes: {e}")
