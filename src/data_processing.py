"""Pipeline de procesamiento de datos para el Tutor Cognitivo. (FIX SHAP & Épica 2)

CORRECCIÓN CRÍTICA (Dic 2025):
- Se implementó un filtrado estricto para filas 'huérfanas' (todas las pertenencias = 0).
- Anteriormente, idxmax() asignaba el primer arquetipo ('Com_Desafiado') a los casos de score 0,
  contaminando el dataset de entrenamiento con perfiles de Bajo Capital Humano.
- Se refina la lógica para evitar ambigüedades.
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
import traceback

# Importar las constantes centralizadas
# Asegúrate de que constants.py exista y tenga estas variables
try:
    from .constants import ALL_ARCHETYPES, TARGET_COLUMN
    from .profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )
except ImportError:
    # Fallback para ejecución directa como script
    from constants import ALL_ARCHETYPES, TARGET_COLUMN
    from profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# FASE 2: INGENIERÍA DE ARQUETIPOS (REGLAS DE EXPERTO / RECETA DORADA)
# ==============================================================================

def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas expertas ("receta dorada") para calcular la pertenencia.
    SOLO para entrenamiento.
    """
    df_out = df.copy()
    logging.info("Calculando pertenencia a arquetipos (Reglas Expertas)...")

    # --- REGLAS REFINADAS PARA EVITAR COLISIONES ---

    def _clasificar_comunicador_desafiado(r):
        # REGLA DE ORO: Debe tener Capital Humano ALTO.
        # Si es Bajo o Medio, se fuerza a 0.0 para limpiar el dataset.
        if r.get('CAPITAL_HUMANO') != '3_Alto':
            return 0.0
            
        pdif, slab, get = r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral'), r.get('GRUPO_ETARIO_INDEC')
        ei, sn = r.get('MBTI_EI_score_sim'), r.get('MBTI_SN_score_sim')
        
        es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        
        prob_base = 0.0
        if es_dificultad_com and es_inclusion_deficiente: 
            prob_base = 0.95 # Muy alta certeza si cumple la definición exacta
        elif es_inclusion_deficiente and pdif == '3_Tres_o_Mas_Dificultades':
             # Caso borde: múltiples dificultades a veces enmascaran lo comunicacional
             prob_base = 0.3
        
        if prob_base == 0.0: return 0.0
        
        # Moduladores MBTI
        factor_ei = 1.0 - (0.2 * ei) if pd.notna(ei) else 1.0 # Introversión favorece
        prob_final = prob_base * factor_ei
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_navegante_informal(r):
        # REGLA DE ORO: Capital Humano BAJO.
        if r.get('CAPITAL_HUMANO') != '1_Bajo':
            return 0.0
            
        slab = r.get('Espectro_Inclusion_Laboral')
        jp = r.get('MBTI_JP_score_sim')
        
        # Debe estar en inclusión precaria o búsqueda
        if slab not in ['3_Inclusion_Precaria_Aprox', '2_Busqueda_Sin_Exito']:
            return 0.0
            
        prob_base = 0.9
        # Moduladores
        factor_jp = 1.0 + (0.2 * jp) if pd.notna(jp) else 1.0 # Percepción (flexibilidad) favorece
        return round(max(0.0, min(prob_base * factor_jp, 1.0)), 2)

    def _clasificar_profesional_subutilizado(r):
        # REGLA DE ORO: Capital Humano ALTO o MEDIO (Técnicos).
        if r.get('CAPITAL_HUMANO') == '1_Bajo':
            return 0.0
            
        pdif, slab = r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        
        es_inclusion_mala = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        # Dificultad "leve" o manejable
        es_dificultad_menor = pdif in ['0_Sin_Dificultad_Registrada', '4_Solo_Certificado', '1A_Motora_Unica', '1B_Visual_Unica']
        
        if es_inclusion_mala and es_dificultad_menor:
            return 0.85
        return 0.0

    def _clasificar_potencial_latente(r):
        # Foco: Exclusión total + Barreras + Desaliento
        if r.get('Espectro_Inclusion_Laboral') != '1_Exclusion_del_Mercado':
            return 0.0
            
        pdif = r.get('Perfil_Dificultad_Agrupado')
        ei = r.get('MBTI_EI_score_sim')
        
        prob_base = 0.6
        if pdif in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']: 
            prob_base = 0.95
        
        # Modulador: Introversión alta refuerza el aislamiento
        factor_ei = 1.0 - (0.3 * ei) if pd.notna(ei) else 1.0
        return round(max(0.0, min(prob_base * factor_ei, 1.0)), 2)

    def _clasificar_candidato_necesidades_sig(r):
        pdif = r.get('Perfil_Dificultad_Agrupado')
        # Definición por severidad de la dificultad
        if pdif in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']:
            return 0.9
        if pdif == '2_Dos_Dificultades':
            return 0.6
        return 0.0

    def _clasificar_joven_transicion(r):
        # Regla estricta de edad
        if r.get('GRUPO_ETARIO_INDEC') != '1_Joven_Adulto_Temprano (14-39)':
            return 0.0
        
        # Priorizar a quienes nunca trabajaron o estudian
        asiste = r.get('PC08') # Asistencia escolar
        if asiste == 1: 
            return 0.9
        
        slab = r.get('Espectro_Inclusion_Laboral')
        if slab == '2_Busqueda_Sin_Exito': # Primer empleo frustrado
            return 0.8
            
        return 0.0

    # Mapeo de funciones
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
    """Wrapper para la fase 2."""
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

    logging.info("--- ⚙️ Iniciando Pipeline de Procesamiento (FIXED SHAP) ---")
    
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
    except Exception as e:
        logging.error(f"Error leyendo CSV: {e}")
        exit()

    # --- Pipeline ---
    df_featured = run_feature_engineering(df_raw)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)

    # --- FIX CRÍTICO: Limpieza de Filas Huérfanas (Todo 0) ---
    archetype_cols = [f'Pertenencia_{name}' for name in ALL_ARCHETYPES]
    
    # Calcular el máximo score de pertenencia por fila
    df_fuzzified['MAX_SCORE'] = df_fuzzified[archetype_cols].max(axis=1)
    
    # Filtrar: Si el max score es < 0.1, la fila es "ruido" y se descarta.
    # Esto evita que idxmax() elija el primer arquetipo por defecto.
    initial_count = len(df_fuzzified)
    df_clean = df_fuzzified[df_fuzzified['MAX_SCORE'] > 0.1].copy()
    dropped_count = initial_count - len(df_clean)
    
    if dropped_count > 0:
        logging.warning(f"⚠️ SE ELIMINARON {dropped_count} FILAS HUÉRFANAS (Score < 0.1).")
        logging.warning("Esto corrige el sesgo detectado por SHAP donde perfiles vacíos se etiquetaban como 'Com_Desafiado'.")
    
    if df_clean.empty:
        logging.error("Error crítico: El filtrado eliminó todas las filas. Revisa las reglas expertas.")
        exit()

    # --- Generación Target ---
    # Ahora es seguro usar idxmax porque sabemos que hay al menos un score > 0.1
    df_clean[TARGET_COLUMN] = df_clean[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')
    
    logging.info("--- Distribución Limpia de Arquetipos ---")
    logging.info("\n" + df_clean[TARGET_COLUMN].value_counts().to_string())

    # --- Guardado ---
    feature_cols = [col for col in df_clean.columns if '_memb' in col]
    df_training = df_clean[feature_cols + [TARGET_COLUMN]].fillna(0.0)
    
    # Ruta de salida (asumiendo estructura de proyecto)
    base_dir = os.getcwd() # O usar lógica de config
    out_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, 'cognitive_profiles.csv')
    df_training.to_csv(out_path, index=False)
    logging.info(f"✅ Archivo de entrenamiento guardado: {out_path}")
    
    # Generar Demo
    demo_list = []
    for arch in df_clean[TARGET_COLUMN].unique():
        subset = df_clean[df_clean[TARGET_COLUMN] == arch]
        if len(subset) > 0:
            demo_list.append(subset.sample(min(2, len(subset)), random_state=42))
    
    if demo_list:
        df_demo = pd.concat(demo_list)
        df_demo['ID'] = [f'Demo_{i}' for i in range(len(df_demo))]
        df_demo.set_index('ID', inplace=True)
        df_demo.to_csv(os.path.join(out_dir, 'demo_profiles.csv'))
        logging.info("✅ Perfiles demo generados.")
