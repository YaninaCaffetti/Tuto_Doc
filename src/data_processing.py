"""Pipeline de procesamiento de datos (Versión Corregida: Rescate de Universitarios Inactivos).

Este script ajusta las reglas de arquetipos para capturar correctamente a los
profesionales que se han retirado del mercado (Inactivos), clasificándolos como
'Potencial Latente' en lugar de descartarlos como huérfanos o confundirlos.

Correcciones:
1.  **Potencial Latente:** Se elimina la exclusión de Capital Alto. Ahora acepta
    universitarios si están en 'Exclusión del Mercado' (Inactivos).
2.  **Inyección Sintética:** Mantenemos la vacunación para 'Comunicador Desafiado'.
3.  **Relabeling:** Mantenemos la corrección de 'Joven Transición'.
4.  **Balanceo:** Mantenemos el upsampling a 1000.
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
import traceback
from sklearn.utils import resample

try:
    from .constants import ALL_ARCHETYPES, TARGET_COLUMN
    from .profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )
except ImportError:
    from constants import ALL_ARCHETYPES, TARGET_COLUMN
    from profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 0. INYECCIÓN DE DATOS SINTÉTICOS
# ==============================================================================
def _inject_synthetic_data() -> pd.DataFrame:
    logging.info("💉 Inyectando datos sintéticos...")
    synthetic_rows = []
    
    # CASO 1: COMUNICADOR DESAFIADO (Joven Universitario + Habla + BÚSQUEDA)
    for i in range(300):
        row = {
            'dificultad_total': 1, 'tipo_dificultad': 6, 'dificultades': 1,
            'MNEA': 5, # Universitario
            'edad_agrupada': 3, # Joven
            'Estado_ocup': 2, # Desocupado (Busca)
            'cat_ocup': 9, 'certificado': 1, 'PC08': 9, 'pc03': 1, 'tipo_hogar': 2,
            'ID': f'SYN_COM_DES_{i}'
        }
        synthetic_rows.append(row)

    # CASO 2: POTENCIAL LATENTE CALIFICADO (Universitario + INACTIVO)
    # Para enseñar la diferencia entre "Busca" (Com/Prof) y "No Busca" (Latente)
    for i in range(100):
        row = {
            'dificultad_total': 1, 'tipo_dificultad': 1, 'dificultades': 1,
            'MNEA': 5, # Universitario
            'edad_agrupada': 3,
            'Estado_ocup': 3, # Inactivo (No busca) -> CLAVE
            'cat_ocup': 9, 'certificado': 1, 'PC08': 9, 'pc03': 1, 'tipo_hogar': 2,
            'ID': f'SYN_POT_LAT_{i}'
        }
        synthetic_rows.append(row)

    return pd.DataFrame(synthetic_rows)

# ==============================================================================
# FASE 2: INGENIERÍA DE ARQUETIPOS (REGLAS AJUSTADAS)
# ==============================================================================

def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    def _clasificar_comunicador_desafiado(r):
        # Requiere Capital Alto
        if r.get('CAPITAL_HUMANO') != '3_Alto': return 0.0
        
        pdif, slab = r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        ei = r.get('MBTI_EI_score_sim')
        
        # Debe estar buscando o en empleo precario (NO Inactivo)
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        if not es_inclusion_deficiente: return 0.0

        es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        
        prob_base = 0.0
        if es_dificultad_com: prob_base = 0.95
        elif pdif == '3_Tres_o_Mas_Dificultades': prob_base = 0.3
            
        if prob_base == 0.0: return 0.0
        
        factor_ei = 1.0 - (0.2 * ei) if pd.notna(ei) else 1.0
        return round(max(0.0, min(prob_base * factor_ei, 1.0)), 2)

    def _clasificar_navegante_informal(r):
        if r.get('CAPITAL_HUMANO') != '1_Bajo': return 0.0
        slab, jp = r.get('Espectro_Inclusion_Laboral'), r.get('MBTI_JP_score_sim')
        if slab not in ['3_Inclusion_Precaria_Aprox', '2_Busqueda_Sin_Exito']: return 0.0
        prob_base = 0.9
        factor_jp = 1.0 + (0.2 * jp) if pd.notna(jp) else 1.0
        return round(max(0.0, min(prob_base * factor_jp, 1.0)), 2)

    def _clasificar_profesional_subutilizado(r):
        if r.get('CAPITAL_HUMANO') == '1_Bajo': return 0.0
        pdif, slab = r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        
        # Debe buscar o empleo precario (NO Inactivo)
        es_inclusion_mala = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        if not es_inclusion_mala: return 0.0
        
        es_dificultad_menor = pdif in ['0_Sin_Dificultad_Registrada', '4_Solo_Certificado', '1A_Motora_Unica', '1B_Visual_Unica']
        if es_dificultad_menor: return 0.85
        return 0.0

    def _clasificar_potencial_latente(r):
        # REGLA AJUSTADA: Acepta Capital Alto si es Inactivo
        slab = r.get('Espectro_Inclusion_Laboral')
        pdif = r.get('Perfil_Dificultad_Agrupado')
        ei = r.get('MBTI_EI_score_sim')
        
        # Condición Sine Qua Non: Exclusión Total (Inactividad)
        if slab != '1_Exclusion_del_Mercado': return 0.0
        
        prob_base = 0.6
        # Si tiene barreras severas -> Muy probable
        if pdif in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']: 
            prob_base = 0.95
        # Si tiene Capital Alto (Universitario Inactivo) -> Probable (Desaliento)
        elif r.get('CAPITAL_HUMANO') == '3_Alto':
            prob_base = 0.85
        
        factor_ei = 1.0 - (0.3 * ei) if pd.notna(ei) else 1.0
        return round(max(0.0, min(prob_base * factor_ei, 1.0)), 2)

    def _clasificar_candidato_necesidades_sig(r):
        pdif = r.get('Perfil_Dificultad_Agrupado')
        # Exclusión suave para Capital Alto (salvo barrera extrema)
        if r.get('CAPITAL_HUMANO') == '3_Alto' and pdif not in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']:
             return 0.0

        if pdif in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']: return 0.9
        if pdif == '2_Dos_Dificultades': return 0.6
        return 0.0

    def _clasificar_joven_transicion(r):
        get, ch, slab, asiste = r.get('GRUPO_ETARIO_INDEC'), r.get('CAPITAL_HUMANO'), r.get('Espectro_Inclusion_Laboral'), r.get('PC08')
        
        # EXCLUSIÓN: Si es Capital Alto, NO es Joven Transición
        if ch == '3_Alto': return 0.0
        
        if get != '1_Joven_Adulto_Temprano (14-39)': return 0.0
        
        prob_base = 0.0
        if asiste == 1: prob_base = 0.85 
        elif slab == '2_Busqueda_Sin_Exito': prob_base = 0.8
            
        return prob_base

    arch_funcs = {
        ALL_ARCHETYPES[0]: _clasificar_comunicador_desafiado,
        ALL_ARCHETYPES[1]: _clasificar_navegante_informal,
        ALL_ARCHETYPES[2]: _clasificar_profesional_subutilizado,
        ALL_ARCHETYPES[3]: _clasificar_potencial_latente,
        ALL_ARCHETYPES[4]: _clasificar_candidato_necesidades_sig,
        ALL_ARCHETYPES[5]: _clasificar_joven_transicion,
    }

    for name, func in arch_funcs.items():
        try:
            df_out[f'Pertenencia_{name}'] = df_out.apply(func, axis=1)
        except Exception as e:
            logging.error(f"Error regla {name}: {e}")
            df_out[f'Pertenencia_{name}'] = 0.0

    return df_out

# ==============================================================================
# MAIN
# ==============================================================================

def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Ejecutando Fase 2...")
    df_mbti = _simulate_mbti_scores(df)
    df_archetyped = _calculate_archetype_membership(df_mbti)
    return df_archetyped

if __name__ == '__main__':
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    except Exception as e: logging.error(f"Config: {e}"); exit()

    RAW_DATA_PATH = config.get('data_paths', {}).get('raw_data')
    if not RAW_DATA_PATH: logging.error("Path error"); exit()

    logging.info("--- ⚙️ Iniciando Pipeline (RESCATE DE UNIVERSITARIOS) ---")
    
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
    except Exception as e: logging.error(f"CSV error: {e}"); exit()

    # 1. Inyección
    df_synthetic = _inject_synthetic_data()
    df_combined = pd.concat([df_raw, df_synthetic], ignore_index=True)

    # 2. Pipeline
    df_featured = run_feature_engineering(df_combined)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)

    # 3. Limpieza Huérfanos
    archetype_cols = [f'Pertenencia_{name}' for name in ALL_ARCHETYPES]
    df_fuzzified['MAX_SCORE'] = df_fuzzified[archetype_cols].max(axis=1)
    df_clean = df_fuzzified[df_fuzzified['MAX_SCORE'] > 0.1].copy()
    
    if len(df_fuzzified) - len(df_clean) > 0:
        logging.warning(f"⚠️ Se eliminaron {len(df_fuzzified) - len(df_clean)} filas huérfanas.")

    # 4. Target
    df_clean[TARGET_COLUMN] = df_clean[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')
    
    # 5. Relabeling (Limpieza de Etiquetas Inconsistentes)
    mask_error = (df_clean[TARGET_COLUMN] == 'Joven_Transicion') & (df_clean['CH_Alto_memb'] > 0.5)
    if mask_error.sum() > 0:
        logging.warning(f"🔄 Reasignando {mask_error.sum()} 'Joven_Transicion' universitarios.")
        mask_com = mask_error & ((df_clean['PD_ComCog_memb'] > 0.5) | (df_clean['PD_Sensorial_memb'] > 0.5))
        df_clean.loc[mask_com, TARGET_COLUMN] = 'Com_Desafiado'
        df_clean.loc[mask_error & (~mask_com), TARGET_COLUMN] = 'Prof_Subutil'

    # 6. Balanceo
    logging.info("--- Balanceo ---")
    target_counts = df_clean[TARGET_COLUMN].value_counts()
    logging.info(f"Distribución:\n{target_counts}")

    MIN_SAMPLES = 1000 
    dfs_to_concat = [df_clean]

    for archetype in ALL_ARCHETYPES:
        count = target_counts.get(archetype, 0)
        if 0 < count < MIN_SAMPLES:
            n_needed = MIN_SAMPLES - count
            df_minority = df_clean[df_clean[TARGET_COLUMN] == archetype]
            if not df_minority.empty:
                df_upsampled = resample(df_minority, replace=True, n_samples=n_needed, random_state=42)
                dfs_to_concat.append(df_upsampled)
                logging.info(f"  › {archetype}: +{n_needed} copias.")

    df_balanced = pd.concat(dfs_to_concat).sample(frac=1, random_state=42).reset_index(drop=True)

    # 7. Guardado
    feature_cols = [col for col in df_balanced.columns if '_memb' in col]
    df_training = df_balanced[feature_cols + [TARGET_COLUMN]].fillna(0.0)
    
    base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    df_training.to_csv(os.path.join(out_dir, 'cognitive_profiles.csv'), index=False)
    
    # Demo
    demo_list = []
    for arch in df_balanced[TARGET_COLUMN].unique():
        subset = df_balanced[df_balanced[TARGET_COLUMN] == arch]
        if len(subset) > 0:
            demo_list.append(subset.sample(min(2, len(subset)), random_state=42))
    
    if demo_list:
        pd.concat(demo_list).to_csv(os.path.join(out_dir, 'demo_profiles.csv'))
        logging.info("✅ Perfiles demo generados.")
