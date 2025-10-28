"""
Módulo de Inferencia de Perfil de Usuario.

Este módulo extrae la lógica de ingeniería de características y fuzzificación
del pipeline de data_processing.py. Su objetivo es permitir la inferencia
en tiempo real de nuevos perfiles de usuario, tomando datos crudos
(en un diccionario) y transformándolos en el pd.Series de características
difusas que espera el modelo cognitivo.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

# Configuración del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# FASE 1: INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma las variables crudas del dataset en características compuestas.

    Args:
        df: El DataFrame crudo (de 1 o más filas) con datos de la encuesta.

    Returns:
        El DataFrame con nuevas columnas de características de alto nivel.
    """
    logging.info("Ejecutando Fase 1: Ingeniería de Características...")
    df_p = df.copy()

    # Define las columnas que deben ser numéricas
    cols_to_numeric = [
        'dificultad_total', 'dificultades', 'tipo_dificultad', 'MNEA', 'edad_agrupada',
        'Estado_ocup', 'cat_ocup', 'certificado', 'PC08', 'pc03', 'tipo_hogar'
    ]
    for col in cols_to_numeric:
        if col in df_p.columns:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce')

    # 1. Perfil de Dificultad Agrupado
    conditions_dificultad = [
        df_p['dificultad_total'] == 0, df_p['tipo_dificultad'] == 1,
        df_p['tipo_dificultad'] == 2, df_p['tipo_dificultad'] == 3,
        df_p['tipo_dificultad'] == 4, df_p['tipo_dificultad'] == 5,
        df_p['tipo_dificultad'] == 6, df_p['tipo_dificultad'] == 7,
        df_p['tipo_dificultad'] == 8, (df_p['tipo_dificultad'] == 9) | (df_p['dificultades'] == 4),
        df_p['dificultad_total'] == 1
    ]
    choices_dificultad = [
        '0_Sin_Dificultad_Registrada', '1A_Motora_Unica', '1B_Visual_Unica', '1C_Auditiva_Unica',
        '1D_Mental_Cognitiva_Unica', '1E_Autocuidado_Unica', '1F_Habla_Comunicacion_Unica',
        '2_Dos_Dificultades', '3_Tres_o_Mas_Dificultades', '4_Solo_Certificado',
        '5_Dificultad_General_No_Detallada'
    ]
    df_p['Perfil_Dificultad_Agrupado'] = np.select(
        conditions_dificultad, choices_dificultad, default='9_Ignorado_o_No_Clasificado'
    )

    # 2. Capital Humano
    conditions_capital = [df_p['MNEA'] == 5, df_p['MNEA'] == 4, df_p['MNEA'].isin([1, 2, 3])]
    choices_capital = ['3_Alto', '2_Medio', '1_Bajo']
    df_p['CAPITAL_HUMANO'] = np.select(conditions_capital, choices_capital, default='9_No_Sabe_o_NC')

    # 3. Grupo Etario
    edad_map = {
        1: '0A_0_a_5_anios', 2: '0B_6_a_13_anios', 3: '1_Joven_Adulto_Temprano (14-39)',
        4: '2_Adulto_Medio (40-64)', 5: '3_Adulto_Mayor (65+)'
    }
    df_p['GRUPO_ETARIO_INDEC'] = df_p['edad_agrupada'].map(edad_map).fillna('No Especificado_Edad')

    # 4. Tenencia de CUD
    cud_map = {1: 'Si_Tiene_CUD', 2: 'No_Tiene_CUD', 9: 'Ignorado_CUD'}
    df_p['TIENE_CUD'] = df_p['certificado'].map(cud_map).fillna('Desconocido_CUD')

    # 5. Espectro de Inclusión Laboral
    conditions_inclusion = [
        df_p['Estado_ocup'] == 3, df_p['Estado_ocup'] == 2,
        (df_p['Estado_ocup'] == 1) & (df_p['cat_ocup'].isin([1, 3])),
        (df_p['Estado_ocup'] == 1) & (df_p['cat_ocup'].isin([2, 4]))
    ]
    choices_inclusion = [
        '1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito',
        '4_Inclusion_Plena_Aprox', '3_Inclusion_Precaria_Aprox'
    ]
    base_inclusion = pd.Series(
        np.select(conditions_inclusion, choices_inclusion, default='No_Clasificado_Laboral'),
        index=df_p.index
    )
    # Solo aplica a personas en edad de trabajar y con discapacidad
    df_p['Espectro_Inclusion_Laboral'] = base_inclusion.where(
        (df_p['edad_agrupada'] >= 3) & (df_p['dificultad_total'] == 1)
    )

    return df_p


# ==============================================================================
# FASE 2: SIMULACIÓN DE RASGOS (MBTI)
# ==============================================================================

def _simulate_mbti_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Simula scores de personalidad tipo MBTI basados en características existentes.

    Args:
        df: DataFrame con características de la Fase 1.

    Returns:
        El DataFrame con 4 nuevas columnas de scores de personalidad simulados.
    """
    df_out = df.copy()
    
    # 1. Extroversión (E) vs. Introversión (I)
    score_ei = pd.Series(0.0, index=df_out.index)
    score_ei.loc[df_out['Perfil_Dificultad_Agrupado'].isin(['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica'])] -= 0.4
    score_ei.loc[df_out['Espectro_Inclusion_Laboral'] == '1_Exclusion_del_Mercado'] -= 0.3
    score_ei.loc[df_out['tipo_hogar'] == 1] -= 0.3 # Hogar unipersonal
    df_out['MBTI_EI_score_sim'] = score_ei.clip(-1., 1.).round(2)

    # 2. Sensación (S) vs. Intuición (N)
    df_out['MBTI_SN_score_sim'] = np.select(
        [df_out['CAPITAL_HUMANO'] == '1_Bajo', df_out['CAPITAL_HUMANO'] == '3_Alto'],
        [-0.5, 0.5], default=0.0
    )

    # 3. Pensamiento (T) vs. Sentimiento (F)
    df_out['MBTI_TF_score_sim'] = np.select(
        [df_out['pc03'] == 4, (df_out['pc03'].notna()) & (df_out['pc03'] != 9) & (df_out['pc03'] != 4)],
        [0.5, -0.25], default=0.0
    )

    # 4. Juicio (J) vs. Percepción (P)
    score_jp = pd.Series(0.0, index=df_out.index)
    score_jp.loc[df_out['Espectro_Inclusion_Laboral'] == '3_Inclusion_Precaria_Aprox'] += 0.5
    score_jp.loc[df_out['TIENE_CUD'] == 'Si_Tiene_CUD'] -= 0.5
    df_out['MBTI_JP_score_sim'] = score_jp.clip(-1., 1.).round(2)

    return df_out


# ==============================================================================
# FASE 3: FUZZIFICACIÓN
# ==============================================================================

def run_fuzzification(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte características categóricas y scores numéricos en variables difusas.

    Toma las características de alto nivel (Fase 1) y los scores simulados (Fase 2)
    y los convierte en variables de membresía difusa (entre 0 y 1) que
    serán las entradas finales para el modelo de ML.

    Args:
        df: DataFrame con características de Fase 1 y scores MBTI.

    Returns:
        El DataFrame con todas las nuevas columnas de membresía (`_memb`).
    """
    logging.info("Ejecutando Fase 3: Fuzzificación...")
    df_out = df.copy()

    # --- Funciones de membresía difusa ---

    def _fuzzificar_capital_humano(r):
        v = r.get('CAPITAL_HUMANO')
        m = {
            '1_Bajo': {'CH_Bajo_memb': 1., 'CH_Medio_memb': .2, 'CH_Alto_memb': 0.},
            '2_Medio': {'CH_Bajo_memb': .2, 'CH_Medio_memb': 1., 'CH_Alto_memb': .2},
            '3_Alto': {'CH_Bajo_memb': 0., 'CH_Medio_memb': .2, 'CH_Alto_memb': 1.}
        }
        return pd.Series(m.get(v, {'CH_Bajo_memb': .33, 'CH_Medio_memb': .33, 'CH_Alto_memb': .33}))

    def _fuzzificar_perfil_dificultad(r):
        v = r.get('Perfil_Dificultad_Agrupado')
        m = {
            '1A_Motora_Unica': {'PD_Motora_memb': 1., 'PD_Sensorial_memb': 0., 'PD_ComCog_memb': 0., 'PD_Autocuidado_memb': .1, 'PD_Multiple_memb': 0.},
            '1B_Visual_Unica': {'PD_Motora_memb': 0., 'PD_Sensorial_memb': 1., 'PD_ComCog_memb': .1, 'PD_Autocuidado_memb': 0., 'PD_Multiple_memb': 0.},
            '1C_Auditiva_Unica': {'PD_Motora_memb': 0., 'PD_Sensorial_memb': 1., 'PD_ComCog_memb': .2, 'PD_Autocuidado_memb': 0., 'PD_Multiple_memb': 0.},
            '1D_Mental_Cognitiva_Unica': {'PD_Motora_memb': .1, 'PD_Sensorial_memb': .1, 'PD_ComCog_memb': 1., 'PD_Autocuidado_memb': .2, 'PD_Multiple_memb': 0.},
            '1E_Autocuidado_Unica': {'PD_Motora_memb': .2, 'PD_Sensorial_memb': 0., 'PD_ComCog_memb': .2, 'PD_Autocuidado_memb': 1., 'PD_Multiple_memb': 0.},
            '1F_Habla_Comunicacion_Unica': {'PD_Motora_memb': .1, 'PD_Sensorial_memb': .2, 'PD_ComCog_memb': 1., 'PD_Autocuidado_memb': 0., 'PD_Multiple_memb': 0.},
            '2_Dos_Dificultades': {'PD_Motora_memb': .4, 'PD_Sensorial_memb': .4, 'PD_ComCog_memb': .4, 'PD_Autocuidado_memb': .4, 'PD_Multiple_memb': 1.},
            '3_Tres_o_Mas_Dificultades': {'PD_Motora_memb': .6, 'PD_Sensorial_memb': .6, 'PD_ComCog_memb': .6, 'PD_Autocuidado_memb': .6, 'PD_Multiple_memb': 1.}
        }
        return pd.Series(m.get(v, {'PD_Motora_memb': .2, 'PD_Sensorial_memb': .2, 'PD_ComCog_memb': .2, 'PD_Autocuidado_memb': .2, 'PD_Multiple_memb': .2}))

    def _fuzzificar_grupo_etario(r):
        v = r.get('GRUPO_ETARIO_INDEC')
        m = {
            '1_Joven_Adulto_Temprano (14-39)': {'Edad_Infanto_Juvenil_memb': .1, 'Edad_Joven_memb': 1., 'Edad_Adulta_memb': .2, 'Edad_Mayor_memb': 0.},
            '2_Adulto_Medio (40-64)': {'Edad_Infanto_Juvenil_memb': 0., 'Edad_Joven_memb': .2, 'Edad_Adulta_memb': 1., 'Edad_Mayor_memb': .2},
            '3_Adulto_Mayor (65+)': {'Edad_Infanto_Juvenil_memb': 0., 'Edad_Joven_memb': 0., 'Edad_Adulta_memb': .2, 'Edad_Mayor_memb': 1.},
            '0B_6_a_13_anios': {'Edad_Infanto_Juvenil_memb': 1., 'Edad_Joven_memb': .2, 'Edad_Adulta_memb': 0., 'Edad_Mayor_memb': 0.},
            '0A_0_a_5_anios': {'Edad_Infanto_Juvenil_memb': 1., 'Edad_Joven_memb': 0., 'Edad_Adulta_memb': 0., 'Edad_Mayor_memb': 0.}
        }
        return pd.Series(m.get(v, {'Edad_Infanto_Juvenil_memb': .25, 'Edad_Joven_memb': .25, 'Edad_Adulta_memb': .25, 'Edad_Mayor_memb': .25}))

    def _fuzzificar_ei_score(r):
        s = r.get('MBTI_EI_score_sim', 0.0)
        s = 0.0 if pd.isna(s) else s
        return pd.Series({
            'MBTI_EI_Introvertido_memb': round(max(0, min(1, 1.25 * (-s - 0.2))), 2),
            'MBTI_EI_Equilibrado_memb': round(max(0, min(1, 1.25 * (s + 0.8))), 2)
        })

    def _fuzzificar_sn_score(r):
        s = r.get('MBTI_SN_score_sim', 0.0)
        s = 0.0 if pd.isna(s) else s
        return pd.Series({
            'MBTI_SN_Sensing_memb': round(max(0, -s + 0.5), 2),
            'MBTI_SN_Intuition_memb': round(max(0, s + 0.5), 2)
        })

    def _fuzzificar_tf_score(r):
        s = r.get('MBTI_TF_score_sim', 0.0)
        s = 0.0 if pd.isna(s) else s
        sn = (s + 0.25) / 0.75
        return pd.Series({
            'MBTI_TF_Thinking_memb': round(max(0, 1 - sn), 2),
            'MBTI_TF_Feeling_memb': round(max(0, sn), 2)
        })

    def _fuzzificar_jp_score(r):
        s = r.get('MBTI_JP_score_sim', 0.0)
        s = 0.0 if pd.isna(s) else s
        return pd.Series({
            'MBTI_JP_Judging_memb': round(max(0, -s + 0.5), 2),
            'MBTI_JP_Perceiving_memb': round(max(0, s + 0.5), 2)
        })
    
    # --- Aplicación de las funciones ---
    fuzz_funcs = [
        _fuzzificar_capital_humano, _fuzzificar_perfil_dificultad, _fuzzificar_grupo_etario,
        _fuzzificar_ei_score, _fuzzificar_sn_score, _fuzzificar_tf_score, _fuzzificar_jp_score
    ]
    for func in fuzz_funcs:
        fuzz_cols_df = df_out.apply(func, axis=1)
        df_out = pd.concat([df_out, fuzz_cols_df], axis=1)
    
    logging.info("Fase 3 completada.")
    return df_out


# ==============================================================================
# FUNCIÓN PRINCIPAL DE INFERENCIA
# ==============================================================================

def infer_profile_features(raw_user_data: Dict) -> pd.Series:
    """
    Función principal de inferencia para un solo usuario.

    Toma un diccionario de datos crudos de un nuevo usuario, ejecuta el pipeline
    completo de ingeniería de características y fuzzificación, y devuelve
    un pd.Series con todas las características generadas, listo para ser usado
    por el modelo cognitivo y el motor de reglas.

    Args:
        raw_user_data: Un diccionario donde las claves son los nombres de
                       las columnas crudas (ej. 'edad_agrupada', 'MNEA') y
                       los valores son las respuestas del usuario.

    Returns:
        Un pd.Series que contiene todas las características generadas
        (features de alto nivel y features '_memb' difusas).
        Devuelve un pd.Series vacío si ocurre un error.
    """
    try:
        # 1. Convertir el dict a un DataFrame de 1 fila
        user_df = pd.DataFrame([raw_user_data])
        
        # 2. Ejecutar la ingeniería de características
        user_featured = run_feature_engineering(user_df)
        
        # 3. Simular scores MBTI
        user_mbti = _simulate_mbti_scores(user_featured)
        
        # 4. Ejecutar la fuzzificación
        user_fuzzified = run_fuzzification(user_mbti)
        
        # 5. Devolver la primera (y única) fila como una Serie
        #    Se devuelven todas las columnas, ya que las de alto nivel
        #    (ej. 'TIENE_CUD') pueden ser útiles para reglas en la app.
        return user_fuzzified.iloc[0]
        
    except Exception as e:
        logging.error(f"Error fatal durante la inferencia del perfil: {e}")
        # Devuelve una Serie vacía en caso de error
        return pd.Series(dtype=object)
