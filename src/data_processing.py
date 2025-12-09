"""Pipeline de procesamiento de datos (Versión Híbrida Final: Lógica Detallada + Fixes Críticos).

Este script combina la lógica heurística detallada del diseño original (con multiplicadores MBTI)
con las correcciones críticas de auditoría para eliminar sesgos:
1. Inyección de datos sintéticos para cubrir huecos de representatividad (ej. Comunicador Desafiado).
2. Filtrado de filas huérfanas (ruido).
3. Corrección de etiquetas (Relabeling) para eliminar el sesgo de 'Joven en Transición' en perfiles profesionales.
4. Balanceo de clases (Upsampling).
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
# 0. INYECCIÓN DE DATOS SINTÉTICOS (LA VACUNA)
# ==============================================================================
def _inject_synthetic_data() -> pd.DataFrame:
    """Genera casos de libro para asegurar que el modelo vea ejemplos perfectos."""
    logging.info("💉 Inyectando datos sintéticos para reforzar reglas...")
    synthetic_rows = []
    
    # CASO CLAVE: Comunicador Desafiado (Joven Universitario + Habla)
    # Generamos 300 casos para que el peso sea significativo
    for i in range(300):
        row = {
            'dificultad_total': 1,
            'tipo_dificultad': 6, # Habla
            'dificultades': 1,
            'MNEA': 5, # Universitario Completo (Capital Alto)
            'edad_agrupada': 3, # Joven (El caso difícil)
            'Estado_ocup': 2, # Desocupado
            'cat_ocup': 9, 'certificado': 1, 'PC08': 9, 'pc03': 1, 'tipo_hogar': 2,
            'ID': f'SYN_COM_DES_{i}'
        }
        synthetic_rows.append(row)
        
    # CASO DE REFUERZO: Profesional Subutilizado (Joven Universitario sin Discapacidad Habla)
    for i in range(100):
        row = {
            'dificultad_total': 1,
            'tipo_dificultad': 1, # Motora (Dificultad menor)
            'dificultades': 1,
            'MNEA': 5, # Universitario
            'edad_agrupada': 3, # Joven
            'Estado_ocup': 2,
            'cat_ocup': 9, 'certificado': 1, 'PC08': 9, 'pc03': 1, 'tipo_hogar': 2,
            'ID': f'SYN_PROF_SUB_{i}'
        }
        synthetic_rows.append(row)

    return pd.DataFrame(synthetic_rows)

# ==============================================================================
# FASE 2: INGENIERÍA DE ARQUETIPOS (REGLAS DETALLADAS RESTAURADAS)
# ==============================================================================

def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula pertenencia usando la lógica detallada con multiplicadores MBTI."""
    df_out = df.copy()
    logging.info("Calculando pertenencia a arquetipos (Lógica Detallada)...")

    # --- REGLAS DETALLADAS ---

    def _clasificar_comunicador_desafiado(r):
        ch, pdif, slab, get = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral'), r.get('GRUPO_ETARIO_INDEC')
        ei, sn = r.get('MBTI_EI_score_sim'), r.get('MBTI_SN_score_sim')
        
        es_capital_alto = (ch == '3_Alto')
        es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        es_dificultad_multiple = (pdif in ['2_Dos_Dificultades', '3_Tres_o_Mas_Dificultades']) and not es_dificultad_com
        
        prob_base = 0.0
        if es_capital_alto and es_dificultad_com and es_inclusion_deficiente: 
            # Regla original: 0.9 si es joven. Esto está bien, el problema era que Joven_Transicion competía.
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
        
        # --- EXCLUSIÓN DE SEGURIDAD ---
        if r.get('CAPITAL_HUMANO') == '3_Alto' and pdif not in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']:
            return 0.0

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
        
        # --- EXCLUSIÓN DE SEGURIDAD ---
        if r.get('CAPITAL_HUMANO') == '3_Alto' and pdif not in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']:
             return 0.0

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
            # --- EXCLUSIÓN CRÍTICA: Joven Universitario NO es transición simple ---
            if ch == '3_Alto':
                return 0.0
            
            if asiste == 1: prob_base = 0.85 
            elif ch in ['2_Medio', '3_Alto'] and slab in ['1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito']: prob_base = 0.95
            elif ch == '1_Bajo' and slab in ['1_Exclusion_del_Mercado', '2_Busqueda_Sin_Exito']: prob_base = 0.65
            
        if prob_base == 0.0: return 0.0
        
        factor_ei = max(0.9, min(1.0 - (0.1 * ei), 1.1)) if pd.notna(ei) else 1.0
        factor_tf = max(0.9, min(1.0 + (0.1 * tf), 1.1)) if pd.notna(tf) else 1.0
        factor_jp = max(0.9, min(1.0 + (0.1 * jp), 1.1)) if pd.notna(jp) else 1.0
        prob_final = prob_base * factor_ei * factor_tf * factor_jp
        return round(max(0.0, min(prob_final, 1.0)), 2)

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


# ==============================================================================
# FASE 4: LIMPIEZA DE ETIQUETAS (CORRECCIÓN POST-TARGET)
# ==============================================================================

def _fix_inconsistent_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina el sesgo residual de los datos generados corrigiendo etiquetas
    lógicamente inconsistentes antes del entrenamiento.
    """
    df_clean = df.copy()
    
    # Condición de Conflicto: Etiquetado como 'Joven_Transicion' PERO tiene Capital Alto
    # (El sesgo que descubrimos en la prueba de inferencia)
    mask_error = (df_clean[TARGET_COLUMN] == 'Joven_Transicion') & (df_clean['CH_Alto_memb'] > 0.5)
    n_errors = mask_error.sum()
    
    if n_errors > 0:
        logging.warning(f"🔄 CORRIGIENDO {n_errors} etiquetas inconsistentes (Joven Universitario != Transición).")
        
        # Estrategia de Reasignación:
        # 1. Si tiene dificultad de comunicación o sensorial -> Com_Desafiado
        mask_com = mask_error & ((df_clean['PD_ComCog_memb'] > 0.5) | (df_clean['PD_Sensorial_memb'] > 0.5))
        df_clean.loc[mask_com, TARGET_COLUMN] = 'Com_Desafiado'
        
        # 2. El resto (probablemente motora o leve) -> Profesional Subutilizado
        mask_prof = mask_error & (~mask_com)
        df_clean.loc[mask_prof, TARGET_COLUMN] = 'Prof_Subutil'
        
        logging.info(f"   › {mask_com.sum()} reasignados a 'Com_Desafiado'")
        logging.info(f"   › {mask_prof.sum()} reasignados a 'Prof_Subutil'")
        
    return df_clean

# ==============================================================================
# MAIN
# ==============================================================================

def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Ejecutando Fase 2: Ingeniería de Arquetipos...")
    df_mbti = _simulate_mbti_scores(df)
    df_archetyped = _calculate_archetype_membership(df_mbti)
    return df_archetyped

if __name__ == '__main__':
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    except Exception as e: logging.error(f"Config error: {e}"); exit()

    RAW_DATA_PATH = config.get('data_paths', {}).get('raw_data')
    if not RAW_DATA_PATH: logging.error("Path error"); exit()

    logging.info("--- ⚙️ Iniciando Pipeline (HÍBRIDO + INYECCIÓN + RELABELING) ---")
    
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
    except Exception as e: logging.error(f"CSV error: {e}"); exit()

    # 1. Inyección de Datos Sintéticos (Para que el modelo aprenda lo que no está en la encuesta)
    df_synthetic = _inject_synthetic_data()
    df_combined = pd.concat([df_raw, df_synthetic], ignore_index=True)
    logging.info(f"Datos Reales: {len(df_raw)} | Sintéticos: {len(df_synthetic)}")

    # 2. Pipeline
    df_featured = run_feature_engineering(df_combined)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)

    # 3. Limpieza de Filas Huérfanas
    archetype_cols = [f'Pertenencia_{name}' for name in ALL_ARCHETYPES]
    df_fuzzified['MAX_SCORE'] = df_fuzzified[archetype_cols].max(axis=1)
    df_clean = df_fuzzified[df_fuzzified['MAX_SCORE'] > 0.1].copy()
    
    # 4. Generación Target Inicial
    df_clean[TARGET_COLUMN] = df_clean[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')

    # 5. CORRECCIÓN DE ETIQUETAS (El paso clave para eliminar el sesgo de edad)
    df_corrected = _fix_inconsistent_labels(df_clean)

    # 6. Balanceo de Clases
    logging.info("--- Balanceo de Clases (Upsampling) ---")
    target_counts = df_corrected[TARGET_COLUMN].value_counts()
    logging.info(f"Distribución Pre-Balanceo:\n{target_counts}")

    MIN_SAMPLES = 1000 
    dfs_to_concat = [df_corrected]

    for archetype in ALL_ARCHETYPES:
        count = target_counts.get(archetype, 0)
        if 0 < count < MIN_SAMPLES:
            n_needed = MIN_SAMPLES - count
            df_minority = df_corrected[df_corrected[TARGET_COLUMN] == archetype]
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
    logging.info("✅ Archivo de entrenamiento guardado.")
    
    # Demo
    demo_list = []
    for arch in df_balanced[TARGET_COLUMN].unique():
        subset = df_balanced[df_balanced[TARGET_COLUMN] == arch]
        if len(subset) > 0:
            demo_list.append(subset.sample(min(2, len(subset)), random_state=42))
    
    if demo_list:
        pd.concat(demo_list).to_csv(os.path.join(out_dir, 'demo_profiles.csv'))
        logging.info("✅ Perfiles demo generados.")
