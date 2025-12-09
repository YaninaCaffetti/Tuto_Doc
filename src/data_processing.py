"""Pipeline de procesamiento de datos (Versión Híbrida: Lógica Backup + Fixes SHAP/Balanceo).

Este script restaura la lógica compleja de reglas heurísticas (con multiplicadores MBTI)
del diseño original, pero incorpora las correcciones críticas de auditoría:
1. Filtrado de huérfanos.
2. Balanceo de clases (Upsampling).
3. Regla de exclusión en 'Joven en Transición' para evitar conflictos con profesionales.
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
import traceback
from sklearn.utils import resample

# Importación robusta
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
# FASE 2: INGENIERÍA DE ARQUETIPOS (REGLAS DE EXPERTO - LÓGICA RESTAURADA)
# ==============================================================================

def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula pertenencia usando la lógica detallada del diseño original."""
    df_out = df.copy()
    logging.info("Calculando pertenencia a arquetipos (Lógica Restaurada + Ajustes)...")

    # --- REGLAS DETALLADAS DEL BACKUP (CON AJUSTES FINOS) ---

    def _clasificar_comunicador_desafiado(r):
        ch, pdif, slab, get = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral'), r.get('GRUPO_ETARIO_INDEC')
        ei, sn = r.get('MBTI_EI_score_sim'), r.get('MBTI_SN_score_sim')
        
        es_capital_alto = (ch == '3_Alto')
        es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        es_dificultad_multiple = (pdif in ['2_Dos_Dificultades', '3_Tres_o_Mas_Dificultades']) and not es_dificultad_com
        
        prob_base = 0.0
        if es_capital_alto and es_dificultad_com and es_inclusion_deficiente: 
            # Ajuste: Igualar probabilidad para jóvenes y adultos para no perderlos ante Joven_Transicion
            prob_base = 0.95 
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
        
        # --- AJUSTE FINO (NUEVO): Penalizar Capital Alto ---
        if r.get('CAPITAL_HUMANO') == '3_Alto' and pdif not in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']:
            return 0.0
        # -------------------------------------------------

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
        
        # --- AJUSTE FINO (NUEVO): Penalizar Capital Alto ---
        if r.get('CAPITAL_HUMANO') == '3_Alto' and pdif not in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']:
             return 0.0
        # -------------------------------------------------

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
            # --- NUEVA REGLA DE EXCLUSIÓN (FIX LÓGICO) ---
            # Si tiene Capital Humano ALTO (Universitario), NO es transición simple.
            if ch == '3_Alto':
                return 0.0
            # ---------------------------------------------
            
            if asiste == 1: prob_base = 0.85 # Asiste a establecimiento educativo
            elif ch in ['2_Medio', '3_Alto'] and slab in ['1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito']: prob_base = 0.95
            elif ch == '1_Bajo' and slab in ['1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito']: prob_base = 0.65
            
        if prob_base == 0.0: return 0.0
        
        factor_ei = max(0.9, min(1.0 - (0.1 * ei), 1.1)) if pd.notna(ei) else 1.0
        factor_tf = max(0.9, min(1.0 + (0.1 * tf), 1.1)) if pd.notna(tf) else 1.0
        factor_jp = max(0.9, min(1.0 + (0.1 * jp), 1.1)) if pd.notna(jp) else 1.0
        prob_final = prob_base * factor_ei * factor_tf * factor_jp
        return round(max(0.0, min(prob_final, 1.0)), 2)

    # --- Aplicación de Reglas ---
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
            logging.error(f"Error aplicando regla para {name}: {e}")
            df_out[f'Pertenencia_{name}'] = 0.0

    return df_out


def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper orquestador para la Fase 2 completa."""
    logging.info("Ejecutando Fase 2: Ingeniería de Arquetipos...")
    df_mbti = _simulate_mbti_scores(df)
    df_archetyped = _calculate_archetype_membership(df_mbti)
    return df_archetyped


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error cargando config: {e}")
        exit()

    RAW_DATA_PATH = config.get('data_paths', {}).get('raw_data')
    if not RAW_DATA_PATH or not os.path.exists(RAW_DATA_PATH):
        logging.error(f"Error crítico: No se encontró archivo en '{RAW_DATA_PATH}'.")
        exit()

    logging.info("--- ⚙️ Iniciando Pipeline de Procesamiento (HÍBRIDO: BACKUP + FIXES) ---")
    
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
    except Exception as e:
        logging.error(f"Error leyendo CSV: {e}")
        exit()

    # --- Pipeline ---
    df_featured = run_feature_engineering(df_raw)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)

    # --- 1. Limpieza de Filas Huérfanas (SHAP FIX) ---
    archetype_cols = [f'Pertenencia_{name}' for name in ALL_ARCHETYPES]
    df_fuzzified['MAX_SCORE'] = df_fuzzified[archetype_cols].max(axis=1)
    
    # Filtrar ruido (Score < 0.1)
    df_clean = df_fuzzified[df_fuzzified['MAX_SCORE'] > 0.1].copy()
    dropped_count = len(df_fuzzified) - len(df_clean)
    if dropped_count > 0:
        logging.warning(f"⚠️ Se eliminaron {dropped_count} filas huérfanas (Score < 0.1).")

    if df_clean.empty:
        logging.error("Error crítico: Dataset vacío tras limpieza.")
        exit()

    # --- Generación Target Base ---
    df_clean[TARGET_COLUMN] = df_clean[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')
    
    # --- 2. Balanceo de Clases (OVERSAMPLING) ---
    logging.info("--- Aplicando Balanceo de Clases (Oversampling) ---")
    target_counts = df_clean[TARGET_COLUMN].value_counts()
    logging.info(f"Distribución Original:\n{target_counts}")

    MIN_SAMPLES = 1000 
    dfs_to_concat = [df_clean]

    for archetype in ALL_ARCHETYPES:
        count = target_counts.get(archetype, 0)
        
        if 0 < count < MIN_SAMPLES:
            df_minority = df_clean[df_clean[TARGET_COLUMN] == archetype]
            if not df_minority.empty:
                n_samples_needed = MIN_SAMPLES - count
                df_upsampled = resample(
                    df_minority, 
                    replace=True,     
                    n_samples=n_samples_needed,    
                    random_state=42
                )
                dfs_to_concat.append(df_upsampled)
                logging.info(f"  › {archetype}: Se añadieron +{n_samples_needed} copias.")
        elif count == 0:
            logging.warning(f"  ⚠️ {archetype}: 0 muestras encontradas.")

    df_balanced = pd.concat(dfs_to_concat).sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info("--- Distribución Final Balanceada ---")
    logging.info("\n" + df_balanced[TARGET_COLUMN].value_counts().to_string())

    # --- Guardado ---
    feature_cols = [col for col in df_balanced.columns if '_memb' in col]
    df_training = df_balanced[feature_cols + [TARGET_COLUMN]].fillna(0.0)
    
    base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, 'cognitive_profiles.csv')
    df_training.to_csv(out_path, index=False)
    logging.info(f"✅ Archivo de entrenamiento BALANCEADO guardado: {out_path}")
    
    # Generar Demo
    demo_list = []
    for arch in df_balanced[TARGET_COLUMN].unique():
        subset = df_balanced[df_balanced[TARGET_COLUMN] == arch]
        if len(subset) > 0:
            unique_subset = subset.drop_duplicates()
            sample_source = unique_subset if len(unique_subset) >= 2 else subset
            demo_list.append(sample_source.sample(min(2, len(sample_source)), random_state=42))
    
    if demo_list:
        df_demo = pd.concat(demo_list)
        df_demo['ID'] = [f'Demo_{i}' for i in range(len(df_demo))]
        df_demo.set_index('ID', inplace=True)
        df_demo.to_csv(os.path.join(out_dir, 'demo_profiles.csv'))
        logging.info("✅ Perfiles demo generados.")
