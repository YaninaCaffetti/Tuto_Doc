# tests/test_data_processing.py

import pandas as pd
import pytest
import sys
import os

# --- Configuración del Path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Tuto_Doc/src')))

# --- Módulos a Probar ---
# Importamos la función que queremos validar y las constantes que utiliza.
from data_processing import run_archetype_engineering
from constants import ARCHETYPE_PROF_SUBUTIL

# --- Datos de Prueba (Fixtures) ---
# Usamos @pytest.fixture para crear datos de prueba reutilizables.

@pytest.fixture
def sample_profile_subutilizado() -> pd.DataFrame:
    """
    Crea un DataFrame con un perfil de usuario de ejemplo que debería clasificar
    fuertemente como 'Profesional Subutilizado'.
    """
    data = {
        'CAPITAL_HUMANO': ['3_Alto'],
        'Perfil_Dificultad_Agrupado': ['1A_Motora_Unica'],
        'Espectro_Inclusion_Laboral': ['2_Busqueda_Sin_Exito'],
        'GRUPO_ETARIO_INDEC': ['2_Adulto_Medio (40-64)'],
        'tipo_hogar': [2], 
        'pc03': [1], 
        'TIENE_CUD': ['Si_Tiene_CUD']
    }
    return pd.DataFrame(data)

# --- Pruebas Unitarias ---

def test_archetype_engineering_calculates_scores(sample_profile_subutilizado: pd.DataFrame):
    """
    Prueba que la Fase 2 de ingeniería de arquetipos se ejecuta y
    calcula correctamente la columna de pertenencia esperada para un perfil específico.
    
    Args:
        sample_profile_subutilizado (pd.DataFrame): El perfil de prueba generado por el fixture.
    """
    # 1. EJECUTAR: Llama a la función que estamos probando con los datos de ejemplo.
    processed_df = run_archetype_engineering(sample_profile_subutilizado)

    # 2. VERIFICAR: Usamos 'assert' para comprobar que los resultados son los esperados.
    
    # Verificar que la columna de pertenencia fue creada correctamente.
    columna_pertenencia = f'Pertenencia_{ARCHETYPE_PROF_SUBUTIL}'
    assert columna_pertenencia in processed_df.columns

    # Verificar que el valor de pertenencia es un número válido (entre 0 y 1).
    pertenencia_score = processed_df[columna_pertenencia].iloc[0]
    assert isinstance(pertenencia_score, (float, np.floating))
    assert 0.0 <= pertenencia_score <= 1.0
    
    # Verificar que la probabilidad es alta, como se espera para este perfil de prueba.
    # (El umbral 0.7 es un ejemplo, se puede ajustar según la lógica de negocio).
    assert pertenencia_score > 0.7

# Para ejecutar las pruebas, instala pytest (`pip install pytest`) y luego,
# desde la carpeta raíz de tu proyecto, corre en la terminal:
# pytest
