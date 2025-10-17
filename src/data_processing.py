"""Pipeline de procesamiento de datos para el Tutor Cognitivo.

Este script ejecuta un pipeline ETL (Extracción, Transformación, Carga) completo:
1.  Carga los datos crudos de la encuesta.
2.  Realiza la ingeniería de características para crear variables de alto nivel.
3.  Aplica un conjunto de reglas expertas ("receta dorada") para calcular la
    pertenencia de cada perfil a 6 arquetipos, generando una base de conocimiento
    con alta separabilidad estadística.
4.  Convierte las características a un formato de lógica difusa (fuzzification).
5.  Limpia y valida los datos para asegurar que estén listos para el entrenamiento.
6.  Genera dos archivos de salida:
    a) `cognitive_profiles.csv`: El dataset completo para entrenar el modelo.
    b) `demo_profiles.csv`: Un subconjunto curado y representativo para la
       demostración en la aplicación Streamlit.
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
from collections import Counter

# Importar las constantes centralizadas
try:
    from src.constants import ALL_ARCHETYPES, TARGET_COLUMN
except ImportError:
    # Fallback para permitir la ejecución directa del script si es necesario
    ALL_ARCHETYPES = ['Com_Desafiado', 'Nav_Informal', 'Prof_Subutil', 'Potencial_Latente', 'Cand_Nec_Sig', 'Joven_Transicion']
    TARGET_COLUMN = 'ARQUETIPO_PRED'

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Funciones del Pipeline de Procesamiento ---

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma las variables crudas del dataset en características compuestas."""
    logging.info("Ejecutando Fase 1: Ingeniería de Características...")
    df_p = df.copy()

    cols_to_numeric = [
        'dificultad_total', 'dificultades', 'tipo_dificultad', 'MNEA', 'edad_agrupada',
        'Estado_ocup', 'cat_ocup', 'certificado', 'PC08', 'pc03', 'tipo_hogar'
    ]
    for col in cols_to_numeric:
        if col in df_p.columns:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce')

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

    conditions_capital = [df_p['MNEA'] == 5, df_p['MNEA'] == 4, df_p['MNEA'].isin([1, 2, 3])]
    choices_capital = ['3_Alto', '2_Medio', '1_Bajo']
    df_p['CAPITAL_HUMANO'] = np.select(conditions_capital, choices_capital, default='9_No_Sabe_o_NC')

    edad_map = {
        1: '0A_0_a_5_anios', 2: '0B_6_a_13_anios', 3: '1_Joven_Adulto_Temprano (14-39)',
        4: '2_Adulto_Medio (40-64)', 5: '3_Adulto_Mayor (65+)'
    }
    df_p['GRUPO_ETARIO_INDEC'] = df_p['edad_agrupada'].map(edad_map).fillna('No Especificado_Edad')

    cud_map = {1: 'Si_Tiene_CUD', 2: 'No_Tiene_CUD', 9: 'Ignorado_CUD'}
    df_p['TIENE_CUD'] = df_p['certificado'].map(cud_map).fillna('Desconocido_CUD')

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
    df_p['Espectro_Inclusion_Laboral'] = base_inclusion.where(
        (df_p['edad_agrupada'] >= 3) & (df_p['dificultad_total'] == 1)
    )

    return df_p


def _simulate_mbti_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Simula scores de personalidad tipo MBTI basados en características existentes."""
    df_out = df.copy()
    
    # Simulación de Extraversión/Introversión (E/I)
    score_ei = pd.Series(0.0, index=df_out.index)
    score_ei.loc[df_out['Perfil_Dificultad_Agrupado'].isin(['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica'])] -= 0.4
    score_ei.loc[df_out['Espectro_Inclusion_Laboral'] == '1_Exclusion_del_Mercado'] -= 0.3
    score_ei.loc[df_out['tipo_hogar'] == 1] -= 0.3
    df_out['MBTI_EI_score_sim'] = score_ei.clip(-1., 1.).round(2)

    # Simulación de Sensación/Intuición (S/N)
    df_out['MBTI_SN_score_sim'] = np.select(
        [df_out['CAPITAL_HUMANO'] == '1_Bajo', df_out['CAPITAL_HUMANO'] == '3_Alto'],
        [-0.5, 0.5], default=0.0
    )

    # Simulación de Pensamiento/Sentimiento (T/F)
    df_out['MBTI_TF_score_sim'] = np.select(
        [df_out['pc03'] == 4, (df_out['pc03'].notna()) & (df_out['pc03'] != 9) & (df_out['pc03'] != 4)],
        [0.5, -0.25], default=0.0
    )

    # Simulación de Juicio/Percepción (J/P)
    score_jp = pd.Series(0.0, index=df_out.index)
    score_jp.loc[df_out['Espectro_Inclusion_Laboral'] == '3_Inclusion_Precaria_Aprox'] += 0.5
    score_jp.loc[df_out['TIENE_CUD'] == 'Si_Tiene_CUD'] -= 0.5
    df_out['MBTI_JP_score_sim'] = score_jp.clip(-1., 1.).round(2)

    return df_out


def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas expertas de la versión "dorada" para calcular la pertenencia a cada arquetipo."""
    df_out = df.copy()

    # --- INICIO DE LAS REGLAS DE ALTO RENDIMIENTO (Refactorizadas para claridad) ---
    def _clasificar_comunicador_desafiado(r):
        ch, pdif, slab, get = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral'), r.get('GRUPO_ETARIO_INDEC')
        ei, sn = r.get('MBTI_EI_score_sim'), r.get('MBTI_SN_score_sim')
        
        es_capital_alto = (ch == '3_Alto')
        es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        es_dificultad_multiple = (pdif in ['2_Dos_Dificultades', '3_Tres_o_Mas_Dificultades']) and not es_dificultad_com
        
        prob_base = 0.0
        if es_capital_alto and es_dificultad_com and es_inclusion_deficiente:
            prob_base = 0.9 if get == '1_Joven_Adulto_Temprano (14-39)' else 0.75 if get == '2_Adulto_Medio (40-64)' else 0.65
        elif es_capital_alto and es_inclusion_deficiente and es_dificultad_multiple:
            prob_base = 0.2
            
        if prob_base == 0.0: return 0.0
        
        factor_ei = 1.0 - (0.2 * ei) if pd.notna(ei) else 1.0
        factor_sn = 1.1 if pd.notna(sn) and sn == 0.5 else (0.9 if pd.notna(sn) and sn == -0.5 else 1.0)
        
        prob_final = prob_base * max(0.8, min(factor_ei, 1.2)) * factor_sn
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_navegante_informal(r):
        ch, pdif, slab = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        sn, jp = r.get('MBTI_SN_score_sim'), r.get('MBTI_JP_score_sim')

        es_capital_bajo = (ch == '1_Bajo')
        es_inclusion_deficiente = (slab in ['3_Inclusion_Precaria_Aprox', '2_Busqueda_Sin_Exito'])
        
        prob_base = 0.0
        if es_capital_bajo and es_inclusion_deficiente:
            if pdif == '2_Dos_Dificultades': prob_base = 0.65
            elif pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica', '3_Tres_o_Mas_Dificultades']: prob_base = 0.1
            else: prob_base = 0.9
            
        if prob_base == 0.0: return 0.0
        
        factor_sn = 1.1 if pd.notna(sn) and sn == -0.5 else (0.9 if pd.notna(sn) and sn == 0.5 else 1.0)
        factor_jp = 1.0 + (0.2 * jp) if pd.notna(jp) else 1.0
        
        prob_final = prob_base * factor_sn * max(0.8, min(factor_jp, 1.2))
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_profesional_subutilizado(r):
        ch, pdif, slab, get = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral'), r.get('GRUPO_ETARIO_INDEC')
        ei, sn, tf, jp = r.get('MBTI_EI_score_sim'), r.get('MBTI_SN_score_sim'), r.get('MBTI_TF_score_sim'), r.get('MBTI_JP_score_sim')

        es_capital_alto = (ch == '3_Alto')
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        es_dificultad_menor = pdif in ['0_Sin_Dificultad_Registrada', '4_Solo_Certificado', '1A_Motora_Unica', '1B_Visual_Unica', '1E_Autocuidado_Unica']
        es_edad_avanzada = (get in ['2_Adulto_Medio (40-64)', '3_Adulto_Mayor (65+)'])
        
        prob_base = 0.0
        if es_capital_alto and es_inclusion_deficiente:
            if es_dificultad_menor: prob_base = 0.9 if es_edad_avanzada else 0.7
            elif pdif == '2_Dos_Dificultades': prob_base = 0.6 if es_edad_avanzada else 0.4
            elif pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica', '3_Tres_o_Mas_Dificultades']: prob_base = 0.15
            
        if prob_base == 0.0: return 0.0
        
        factor_ei = max(0.9, min(1.0 - (0.1 * ei), 1.1)) if pd.notna(ei) else 1.0
        factor_sn = 1.05 if pd.notna(sn) and sn == 0.5 else (0.95 if pd.notna(sn) and sn == -0.5 else 1.0)
        factor_tf = max(0.8, min(1.0 - (0.4 * tf), 1.2)) if pd.notna(tf) else 1.0
        factor_jp = max(0.8, min(1.0 - (0.2 * jp), 1.2)) if pd.notna(jp) else 1.0
        
        prob_final = prob_base * factor_ei * factor_sn * factor_tf * factor_jp
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_potencial_latente(r):
        slab, pdif = r.get('Espectro_Inclusion_Laboral'), r.get('Perfil_Dificultad_Agrupado')
        ei, tf, jp = r.get('MBTI_EI_score_sim'), r.get('MBTI_TF_score_sim'), r.get('MBTI_JP_score_sim')

        prob_base = 0.0
        if slab == '1_Exclusion_del_Mercado':
            if pdif in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']: prob_base = 0.95
            elif pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica']: prob_base = 0.8
            elif pdif in ['1A_Motora_Unica', '1B_Visual_Unica', '2_Dos_Dificultades']: prob_base = 0.6
            else: prob_base = 0.4
            
        if prob_base == 0.0: return 0.0
        
        factor_ei = max(0.7, min(1.0 - (0.3 * ei), 1.3)) if pd.notna(ei) else 1.0
        factor_tf = max(0.9, min(1.0 + (0.1 * tf), 1.1)) if pd.notna(tf) else 1.0
        factor_jp = max(0.8, min(1.0 + (0.2 * jp), 1.2)) if pd.notna(jp) else 1.0
        
        prob_final = prob_base * factor_ei * factor_tf * factor_jp
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_candidato_necesidades_sig(r):
        pdif, slab = r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        ei, tf, jp = r.get('MBTI_EI_score_sim'), r.get('MBTI_TF_score_sim'), r.get('MBTI_JP_score_sim')

        prob_base = 0.0
        if pdif == '3_Tres_o_Mas_Dificultades': prob_base = 0.95
        elif pdif == '1E_Autocuidado_Unica': prob_base = 0.85
        elif pdif == '2_Dos_Dificultades': prob_base = 0.75
        elif pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica', '1D_Mental_Cognitiva_Unica']: prob_base = 0.4
        elif pdif in ['1A_Motora_Unica', '1B_Visual_Unica']: prob_base = 0.2
        elif pd.notna(pdif) and pdif != '0_Sin_Dificultad_Registrada': prob_base = 0.2
            
        if prob_base == 0.0: return 0.0
        
        factor_lab = 0.7 if slab == '4_Inclusion_Plena_Aprox' else (0.9 if slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'] else 1.0)
        prob_mod_lab = prob_base * factor_lab
        
        factor_ei = max(0.9, min(1.0 - (0.1 * ei), 1.1)) if pd.notna(ei) else 1.0
        factor_tf = max(0.9, min(1.0 + (0.1 * tf), 1.1)) if pd.notna(tf) else 1.0
        factor_jp = max(0.9, min(1.0 - (0.1 * jp), 1.1)) if pd.notna(jp) else 1.0
        
        prob_final = prob_mod_lab * factor_ei * factor_tf * factor_jp
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_joven_transicion(r):
        get, ch, slab, asiste = r.get('GRUPO_ETARIO_INDEC'), r.get('CAPITAL_HUMANO'), r.get('Espectro_Inclusion_Laboral'), r.get('PC08')
        ei, tf, jp = r.get('MBTI_EI_score_sim'), r.get('MBTI_TF_score_sim'), r.get('MBTI_JP_score_sim')

        prob_base = 0.0
        if get == '1_Joven_Adulto_Temprano (14-39)':
            if asiste == 1: prob_base = 0.85
            elif ch in ['2_Medio', '3_Alto'] and slab in ['1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito']: prob_base = 0.95
            elif ch == '1_Bajo' and slab in ['1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito']: prob_base = 0.65
        
        if prob_base == 0.0: return 0.0
        
        factor_ei = max(0.9, min(1.0 - (0.1 * ei), 1.1)) if pd.notna(ei) else 1.0
        factor_tf = max(0.9, min(1.0 + (0.1 * tf), 1.1)) if pd.notna(tf) else 1.0
        factor_jp = max(0.9, min(1.0 + (0.1 * jp), 1.1)) if pd.notna(jp) else 1.0

        prob_final = prob_base * factor_ei * factor_tf * factor_jp
        return round(max(0.0, min(prob_final, 1.0)), 2)
    # --- FIN DE LAS REGLAS DE ALTO RENDIMIENTO ---

    arch_funcs = {
        ALL_ARCHETYPES[0]: _clasificar_comunicador_desafiado,
        ALL_ARCHETYPES[1]: _clasificar_navegante_informal,
        ALL_ARCHETYPES[2]: _clasificar_profesional_subutilizado,
        ALL_ARCHETYPES[3]: _clasificar_potencial_latente,
        ALL_ARCHETYPES[4]: _clasificar_candidato_necesidades_sig,
        ALL_ARCHETYPES[5]: _clasificar_joven_transicion,
    }
    for name, func in arch_funcs.items():
        df_out[f'Pertenencia_{name}'] = df_out.apply(func, axis=1)
    return df_out


def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el grado de pertenencia a arquetipos usando la lógica "dorada"."""
    logging.info("Ejecutando Fase 2: Ingeniería de Arquetipos (con reglas de alto rendimiento)...")
    df_mbti = _simulate_mbti_scores(df)
    df_archetyped = _calculate_archetype_membership(df_mbti)
    logging.info("Fase 2 completada.")
    return df_archetyped


def run_fuzzification(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte características y scores en variables difusas."""
    logging.info("Ejecutando Fase 3: Fuzzificación...")
    df_out = df.copy()

    def _fuzzificar_capital_humano(r):
        v=r.get('CAPITAL_HUMANO'); m={'1_Bajo':{'CH_Bajo_memb':1.,'CH_Medio_memb':.2,'CH_Alto_memb':0.},'2_Medio':{'CH_Bajo_memb':.2,'CH_Medio_memb':1.,'CH_Alto_memb':.2},'3_Alto':{'CH_Bajo_memb':0.,'CH_Medio_memb':.2,'CH_Alto_memb':1.}}; return pd.Series(m.get(v,{'CH_Bajo_memb':.33,'CH_Medio_memb':.33,'CH_Alto_memb':.33}))
    def _fuzzificar_perfil_dificultad(r):
        v=r.get('Perfil_Dificultad_Agrupado'); m={'1A_Motora_Unica':{'PD_Motora_memb':1.,'PD_Sensorial_memb':0.,'PD_ComCog_memb':0.,'PD_Autocuidado_memb':.1,'PD_Multiple_memb':0.},'1B_Visual_Unica':{'PD_Motora_memb':0.,'PD_Sensorial_memb':1.,'PD_ComCog_memb':.1,'PD_Autocuidado_memb':0.,'PD_Multiple_memb':0.},'1C_Auditiva_Unica':{'PD_Motora_memb':0.,'PD_Sensorial_memb':1.,'PD_ComCog_memb':.2,'PD_Autocuidado_memb':0.,'PD_Multiple_memb':0.},'1D_Mental_Cognitiva_Unica':{'PD_Motora_memb':.1,'PD_Sensorial_memb':.1,'PD_ComCog_memb':1.,'PD_Autocuidado_memb':.2,'PD_Multiple_memb':0.},'1E_Autocuidado_Unica':{'PD_Motora_memb':.2,'PD_Sensorial_memb':0.,'PD_ComCog_memb':.2,'PD_Autocuidado_memb':1.,'PD_Multiple_memb':0.},'1F_Habla_Comunicacion_Unica':{'PD_Motora_memb':.1,'PD_Sensorial_memb':.2,'PD_ComCog_memb':1.,'PD_Autocuidado_memb':0.,'PD_Multiple_memb':0.},'2_Dos_Dificultades':{'PD_Motora_memb':.4,'PD_Sensorial_memb':.4,'PD_ComCog_memb':.4,'PD_Autocuidado_memb':.4,'PD_Multiple_memb':1.},'3_Tres_o_Mas_Dificultades':{'PD_Motora_memb':.6,'PD_Sensorial_memb':.6,'PD_ComCog_memb':.6,'PD_Autocuidado_memb':.6,'PD_Multiple_memb':1.}}; return pd.Series(m.get(v,{'PD_Motora_memb':.2,'PD_Sensorial_memb':.2,'PD_ComCog_memb':.2,'PD_Autocuidado_memb':.2,'PD_Multiple_memb':.2}))
    def _fuzzificar_grupo_etario(r):
        v=r.get('GRUPO_ETARIO_INDEC'); m={'1_Joven_Adulto_Temprano (14-39)':{'Edad_Infanto_Juvenil_memb':.1,'Edad_Joven_memb':1.,'Edad_Adulta_memb':.2,'Edad_Mayor_memb':0.},'2_Adulto_Medio (40-64)':{'Edad_Infanto_Juvenil_memb':0.,'Edad_Joven_memb':.2,'Edad_Adulta_memb':1.,'Edad_Mayor_memb':.2},'3_Adulto_Mayor (65+)':{'Edad_Infanto_Juvenil_memb':0.,'Edad_Joven_memb':0.,'Edad_Adulta_memb':.2,'Edad_Mayor_memb':1.},'0B_6_a_13_anios':{'Edad_Infanto_Juvenil_memb':1.,'Edad_Joven_memb':.2,'Edad_Adulta_memb':0.,'Edad_Mayor_memb':0.},'0A_0_a_5_anios':{'Edad_Infanto_Juvenil_memb':1.,'Edad_Joven_memb':0.,'Edad_Adulta_memb':0.,'Edad_Mayor_memb':0.}}; return pd.Series(m.get(v,{'Edad_Infanto_Juvenil_memb':.25,'Edad_Joven_memb':.25,'Edad_Adulta_memb':.25,'Edad_Mayor_memb':.25}))
    def _fuzzificar_ei_score(r):
        s=r.get('MBTI_EI_score_sim',0.);s=0. if pd.isna(s)else s;return pd.Series({'MBTI_EI_Introvertido_memb':round(max(0,min(1,1.25*(-s-.2))),2),'MBTI_EI_Equilibrado_memb':round(max(0,min(1,1.25*(s+.8))),2)})
    def _fuzzificar_sn_score(r):
        s=r.get('MBTI_SN_score_sim',0.);s=0. if pd.isna(s)else s;return pd.Series({'MBTI_SN_Sensing_memb':round(max(0,-s+.5),2),'MBTI_SN_Intuition_memb':round(max(0,s+.5),2)})
    def _fuzzificar_tf_score(r):
        s=r.get('MBTI_TF_score_sim',0.);s=0. if pd.isna(s)else s;sn=(s+.25)/.75;return pd.Series({'MBTI_TF_Thinking_memb':round(max(0,1-sn),2),'MBTI_TF_Feeling_memb':round(max(0,sn),2)})
    def _fuzzificar_jp_score(r):
        s=r.get('MBTI_JP_score_sim',0.);s=0. if pd.isna(s)else s;return pd.Series({'MBTI_JP_Judging_memb':round(max(0,-s+.5),2),'MBTI_JP_Perceiving_memb':round(max(0,s+.5),2)})
    
    fuzz_funcs=[_fuzzificar_capital_humano,_fuzzificar_perfil_dificultad,_fuzzificar_grupo_etario,_fuzzificar_ei_score,_fuzzificar_sn_score,_fuzzificar_tf_score,_fuzzificar_jp_score]
    for func in fuzz_funcs:
        fuzz_cols_df=df_out.apply(func,axis=1)
        df_out=pd.concat([df_out,fuzz_cols_df],axis=1)
    
    logging.info("Fase 3 completada.")
    return df_out


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    RAW_DATA_PATH = config['data_paths']['raw_data']
    
    logging.info("--- ⚙️ Iniciando Pipeline de Procesamiento de Datos ---")
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"No se encontró el archivo de datos crudos en '{RAW_DATA_PATH}'.")
    else:
        logging.info(f"Cargando datos crudos desde: {RAW_DATA_PATH}")
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';')
        
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_archetyped)
        
        logging.info("Ejecutando Fase 4: Limpieza y Validación de Datos...")
        feature_cols = [col for col in df_fuzzified.columns if '_memb' in col]
        
        for col in feature_cols:
            df_fuzzified[col] = pd.to_numeric(df_fuzzified[col], errors='coerce')
        
        df_fuzzified[feature_cols] = df_fuzzified[feature_cols].fillna(0.0)
        logging.info("Fase 4 completada. Todas las columnas de características son numéricas.")
        
        logging.info("Preparando archivo para el entrenamiento del modelo cognitivo...")
        archetype_cols = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
        df_fuzzified[TARGET_COLUMN] = df_fuzzified[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')
        
        # --- Verificación de Distribución de Clases ---
        logging.info("--- Distribución Final de Arquetipos en el Dataset de Entrenamiento ---")
        logging.info("\n" + df_fuzzified[TARGET_COLUMN].value_counts().to_string())
        logging.info("-" * 60)
        
        df_cognitive_training = df_fuzzified[feature_cols + [TARGET_COLUMN]]
        
        OUTPUT_TRAINING_PATH = config['data_paths']['cognitive_training_data']
        os.makedirs(os.path.dirname(OUTPUT_TRAINING_PATH), exist_ok=True)
        df_cognitive_training.to_csv(OUTPUT_TRAINING_PATH, index=False)
        logging.info(f"✅ Archivo para entrenamiento guardado en: {OUTPUT_TRAINING_PATH}")

        # --- Creación de un Dataset de Demostración CURADO ---
        logging.info("Creando un archivo de demostración curado y representativo...")
        learned_archetypes = df_fuzzified[TARGET_COLUMN].unique().tolist()
        demo_profiles_list = []
        for archetype in learned_archetypes:
            subset = df_fuzzified[df_fuzzified[TARGET_COLUMN] == archetype]
            sample_size = min(2, len(subset))
            if sample_size > 0:
                demo_profiles_list.append(subset.sample(n=sample_size, random_state=42))
        
        if demo_profiles_list:
            df_demo = pd.concat(demo_profiles_list)
            logging.info(f"Se seleccionaron {len(df_demo)} perfiles para la demo, representando a los arquetipos: {learned_archetypes}")
        else:
            logging.warning("No se pudieron seleccionar perfiles para la demo. Usando una muestra aleatoria como fallback.")
            df_demo = df_fuzzified.sample(n=5, random_state=42)

        df_demo['ID'] = [f'Perfil_Demo_{i+1}' for i in range(len(df_demo))]
        df_demo.set_index('ID', inplace=True)
        
        OUTPUT_DEMO_PATH = config['data_paths']['demo_profiles']
        os.makedirs(os.path.dirname(OUTPUT_DEMO_PATH), exist_ok=True)
        df_demo.to_csv(OUTPUT_DEMO_PATH)
        logging.info(f"✅ Archivo para demostración CURADO guardado en: {OUTPUT_DEMO_PATH}")

        logging.info("--- 🎉 Pipeline de Procesamiento de Datos Finalizado ---")

