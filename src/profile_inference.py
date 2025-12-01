"""
Módulo de Inferencia de Perfil de Usuario.

Este módulo centraliza la lógica de Feature Engineering, Simulación de Rasgos (MBTI)
y Fuzzificación. Su diseño permite su uso dual:
1. En 'data_processing.py' para generar datos de entrenamiento offline.
2. En 'app.py' para la inferencia en tiempo real de nuevos usuarios (Onboarding).

Funciones Principales:
- infer_profile_features: Orquestador principal para un usuario individual (dict -> Series).
- run_feature_engineering: Transforma variables crudas en características de alto nivel.
- _simulate_mbti_scores: Infiere rasgos de personalidad a partir de datos sociodemográficos.
- run_fuzzification: Convierte características en variables de membresía difusa (0-1).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Union

# Configuración del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# FASE 1: INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma las variables crudas del dataset (encuesta) en características compuestas de alto nivel.

    Realiza la limpieza de tipos de datos, manejo de nulos y la creación de variables
    sintéticas como 'Perfil_Dificultad_Agrupado' y 'CAPITAL_HUMANO'.

    Args:
        df (pd.DataFrame): DataFrame con columnas crudas (ej. 'dificultad_total', 'MNEA').

    Returns:
        pd.DataFrame: DataFrame enriquecido con nuevas columnas de características.
    """
    # logging.info("Ejecutando Fase 1: Ingeniería de Características...") # Reducir verbosidad en inferencia
    df_p = df.copy()

    # 1. Conversión numérica segura y manejo de nulos
    cols_to_numeric = [
        'dificultad_total', 'dificultades', 'tipo_dificultad', 'MNEA', 'edad_agrupada',
        'Estado_ocup', 'cat_ocup', 'certificado', 'PC08', 'pc03', 'tipo_hogar'
    ]
    for col in cols_to_numeric:
        if col in df_p.columns:
            # Coerce errors to NaN, then fill with 0 (o valor neutro seguro)
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce').fillna(0)

    # 2. Creación de 'Perfil_Dificultad_Agrupado'
    # Lógica de agrupación basada en reglas de negocio del dominio de discapacidad
    conditions_dificultad = [
        df_p['dificultad_total'] == 0,
        df_p['tipo_dificultad'] == 1,
        df_p['tipo_dificultad'] == 2,
        df_p['tipo_dificultad'] == 3,
        df_p['tipo_dificultad'] == 4,
        df_p['tipo_dificultad'] == 5,
        df_p['tipo_dificultad'] == 6,
        df_p['tipo_dificultad'].isin([7, 8]), # 7 y 8 son múltiples o combinadas
        (df_p['tipo_dificultad'] == 9) | (df_p['dificultades'] == 4)
    ]
    choices_dificultad = [
        '0_Sin_Dificultad_Registrada', '1A_Motora_Unica', '1B_Visual_Unica',
        '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica', '1E_Autocuidado_Unica',
        '1F_Habla_Comunicacion_Unica', '2_Dos_Dificultades', '3_Tres_o_Mas_Dificultades'
    ]
    
    # np.select requiere que conditions y choices tengan la misma longitud.
    # El default maneja casos no cubiertos (ej. nulos originales).
    df_p['Perfil_Dificultad_Agrupado'] = np.select(
        conditions_dificultad, choices_dificultad, default='3_Tres_o_Mas_Dificultades' # Default conservador
    )

    # 3. Creación de 'CAPITAL_HUMANO'
    # Basado en MNEA (Nivel Educativo): 5=Terciario, 4=Terciario Inc, 1-3=Bajo
    conditions_capital = [df_p['MNEA'] == 5, df_p['MNEA'] == 4, df_p['MNEA'].isin([1, 2, 3])]
    choices_capital = ['3_Alto', '2_Medio', '1_Bajo']
    df_p['CAPITAL_HUMANO'] = np.select(conditions_capital, choices_capital, default='1_Bajo')

    # 4. Creación de 'GRUPO_ETARIO_INDEC'
    edad_map = {
        1: '0A_0_a_5_anios', 2: '0B_6_a_13_anios', 3: '1_Joven_Adulto_Temprano (14-39)',
        4: '2_Adulto_Medio (40-64)', 5: '3_Adulto_Mayor (65+)'
    }
    df_p['GRUPO_ETARIO_INDEC'] = df_p['edad_agrupada'].map(edad_map).fillna('1_Joven_Adulto_Temprano (14-39)')

    # 5. Creación de 'TIENE_CUD'
    cud_map = {1: 'Si_Tiene_CUD', 2: 'No_Tiene_CUD', 9: 'Ignorado_CUD'}
    df_p['TIENE_CUD'] = df_p['certificado'].map(cud_map).fillna('No_Tiene_CUD')

    # 6. Creación de 'Espectro_Inclusion_Laboral'
    # Combina Estado Ocupacional y Categoría Ocupacional
    conditions_inclusion = [
        df_p['Estado_ocup'] == 3, # Inactivo
        df_p['Estado_ocup'] == 2, # Desocupado
        (df_p['Estado_ocup'] == 1) & (df_p['cat_ocup'].isin([1, 3])), # Ocupado Pleno
        (df_p['Estado_ocup'] == 1) & (df_p['cat_ocup'].isin([2, 4]))  # Ocupado Precario
    ]
    choices_inclusion = [
        '1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito',
        '4_Inclusion_Plena_Aprox', '3_Inclusion_Precaria_Aprox'
    ]
    df_p['Espectro_Inclusion_Laboral'] = np.select(conditions_inclusion, choices_inclusion, default='1_Exclusion_del_Mercado')

    return df_p


# ==============================================================================
# FASE 2: SIMULACIÓN DE RASGOS (MBTI)
# ==============================================================================

def _simulate_mbti_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula scores de personalidad (MBTI) basados en proxies sociodemográficos.
    
    NOTA: Esto NO es un test psicométrico real. Es una inferencia estadística de
    tendencias basada en la literatura para enriquecer el perfil del usuario.

    Args:
        df (pd.DataFrame): DataFrame con características de la Fase 1.

    Returns:
        pd.DataFrame: DataFrame con 4 nuevas columnas de scores MBTI (-1.0 a 1.0).
    """
    df_out = df.copy()
    
    # 1. Extroversión (E) vs. Introversión (I)
    # Hipótesis: Barreras de comunicación y exclusión correlacionan con Introversión (score negativo).
    score_ei = pd.Series(0.0, index=df_out.index)
    mask_intro = df_out['Perfil_Dificultad_Agrupado'].isin(['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
    score_ei.loc[mask_intro] -= 0.4
    score_ei.loc[df_out['Espectro_Inclusion_Laboral'] == '1_Exclusion_del_Mercado'] -= 0.3
    # Clip para mantener rango [-1, 1]
    df_out['MBTI_EI_score_sim'] = score_ei.clip(-1., 1.).round(2)

    # 2. Sensación (S) vs. Intuición (N)
    # Hipótesis: Capital Humano Bajo -> Sensación (Concreto); Alto -> Intuición (Abstracto).
    df_out['MBTI_SN_score_sim'] = np.select(
        [df_out['CAPITAL_HUMANO'] == '1_Bajo', df_out['CAPITAL_HUMANO'] == '3_Alto'],
        [-0.5, 0.5], default=0.0
    )

    # 3. Pensamiento (T) vs. Sentimiento (F)
    # Sin lógica fuerte en demo, default neutro (0.0).
    df_out['MBTI_TF_score_sim'] = 0.0

    # 4. Juicio (J) vs. Percepción (P)
    # Hipótesis: Inclusión precaria requiere flexibilidad (Percepción). Tener CUD implica estructura (Juicio).
    score_jp = pd.Series(0.0, index=df_out.index)
    score_jp.loc[df_out['Espectro_Inclusion_Laboral'] == '3_Inclusion_Precaria_Aprox'] += 0.5
    df_out['MBTI_JP_score_sim'] = score_jp.clip(-1., 1.).round(2)

    return df_out


# ==============================================================================
# FASE 3: FUZZIFICACIÓN
# ==============================================================================

def run_fuzzification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas y scores numéricos en grados de membresía difusa (0-1).
    
    Genera las columnas `_memb` que serán las entradas (features) del modelo de ML.

    Args:
        df (pd.DataFrame): DataFrame con características de Fase 1 y scores MBTI.

    Returns:
        pd.DataFrame: DataFrame final con todas las columnas de membresía.
    """
    # logging.info("Ejecutando Fase 3: Fuzzificación...")
    df_out = df.copy()

    # --- A. Fuzzificación de Capital Humano ---
    map_ch = {
        '1_Bajo': [1., .2, 0.],   # [Bajo, Medio, Alto]
        '2_Medio': [.2, 1., .2],
        '3_Alto': [0., .2, 1.]
    }
    def get_ch(x): return map_ch.get(x, [.33, .33, .33]) # Default incierto
    
    ch_vals = df_out['CAPITAL_HUMANO'].apply(get_ch).tolist()
    df_out[['CH_Bajo_memb', 'CH_Medio_memb', 'CH_Alto_memb']] = pd.DataFrame(ch_vals, index=df_out.index)

    # --- B. Fuzzificación de Perfil de Dificultad ---
    # Simplificado para robustez en inferencia
    def get_pd(x):
        # Retorna: [Motora, Sensorial, ComCog, Autocuidado, Multiple]
        if 'Motora' in x: return [1., 0., 0., .1, 0.]
        if 'Visual' in x or 'Auditiva' in x: return [0., 1., .1, 0., 0.]
        if 'Mental' in x or 'Habla' in x: return [.1, .1, 1., .1, 0.]
        if 'Autocuidado' in x: return [.2, 0., .1, 1., 0.]
        if 'Dos' in x or 'Tres' in x: return [.4, .4, .4, .4, 1.]
        return [.2, .2, .2, .2, .2] # Default equilibrado

    pd_vals = df_out['Perfil_Dificultad_Agrupado'].apply(get_pd).tolist()
    cols_pd = ['PD_Motora_memb', 'PD_Sensorial_memb', 'PD_ComCog_memb', 'PD_Autocuidado_memb', 'PD_Multiple_memb']
    df_out[cols_pd] = pd.DataFrame(pd_vals, index=df_out.index)

    # --- C. Fuzzificación de Grupo Etario ---
    def get_age(x):
        # Retorna: [Infanto, Joven, Adulto, Mayor]
        if 'Joven' in x: return [.1, 1., .2, 0.]
        if 'Medio' in x: return [0., .2, 1., .2]
        if 'Mayor' in x: return [0., 0., .2, 1.]
        return [0., 0., 0., 0.] # Default
    
    age_vals = df_out['GRUPO_ETARIO_INDEC'].apply(get_age).tolist()
    cols_age = ['Edad_Infanto_Juvenil_memb', 'Edad_Joven_memb', 'Edad_Adulta_memb', 'Edad_Mayor_memb']
    df_out[cols_age] = pd.DataFrame(age_vals, index=df_out.index)

    # --- D. Fuzzificación de Scores MBTI (Funciones Sigmoides/Lineales) ---
    
    # EI: Introvertido vs Equilibrado (Extrovertido implícito)
    s_ei = df_out['MBTI_EI_score_sim']
    df_out['MBTI_EI_Introvertido_memb'] = ((-s_ei - 0.2) * 1.25).clip(0, 1).round(2)
    df_out['MBTI_EI_Equilibrado_memb'] = ((s_ei + 0.8) * 1.25).clip(0, 1).round(2)

    # SN: Sensing vs Intuition
    s_sn = df_out['MBTI_SN_score_sim']
    df_out['MBTI_SN_Sensing_memb'] = (-s_sn + 0.5).clip(0, 1).round(2)
    df_out['MBTI_SN_Intuition_memb'] = (s_sn + 0.5).clip(0, 1).round(2)

    # TF: Thinking vs Feeling
    s_tf = df_out['MBTI_TF_score_sim']
    sn_tf = (s_tf + 0.25) / 0.75 # Normalización
    df_out['MBTI_TF_Thinking_memb'] = (1 - sn_tf).clip(0, 1).round(2)
    df_out['MBTI_TF_Feeling_memb'] = sn_tf.clip(0, 1).round(2)

    # JP: Judging vs Perceiving
    s_jp = df_out['MBTI_JP_score_sim']
    df_out['MBTI_JP_Judging_memb'] = (-s_jp + 0.5).clip(0, 1).round(2)
    df_out['MBTI_JP_Perceiving_memb'] = (s_jp + 0.5).clip(0, 1).round(2)

    return df_out


# ==============================================================================
# FUNCIÓN PÚBLICA DE INFERENCIA (ENTRY POINT)
# ==============================================================================

def infer_profile_features(raw_user_data: Dict) -> pd.Series:
    """
    Ejecuta el pipeline completo de inferencia para un usuario individual.

    Esta es la función que debe llamar `app.py` cuando se envía el formulario de onboarding.
    
    Args:
        raw_user_data (Dict): Diccionario con los datos del formulario. 
                              Claves esperadas: 'edad_agrupada', 'MNEA', 'Estado_ocup', etc.

    Returns:
        pd.Series: Una Serie de Pandas conteniendo TODAS las características generadas
                   (tanto las de alto nivel como las '_memb' difusas).
                   Retorna una Serie vacía si ocurre un error crítico.
    """
    try:
        # 1. Convertir dict a DataFrame (1 fila)
        user_df = pd.DataFrame([raw_user_data])
        
        # 2. Pipeline secuencial
        user_featured = run_feature_engineering(user_df)
        user_mbti = _simulate_mbti_scores(user_featured)
        user_fuzzified = run_fuzzification(user_mbti)
        
        # 3. Retornar la fila como Serie
        return user_fuzzified.iloc[0]
        
    except Exception as e:
        logging.error(f"Error crítico en infer_profile_features: {e}")
        # Retorna serie vacía para manejo de errores aguas arriba
        return pd.Series(dtype=object)
