"""
Pipeline de procesamiento de datos para el Tutor Cognitivo. (FIX SHAP + BALANCEO + REGLAS)

Este script ejecuta el pipeline ETL (Extracción, Transformación, Carga) completo
para generar los datos necesarios para el entrenamiento del modelo cognitivo
y los perfiles de demostración.

Correcciones Críticas Implementadas:
1.  **Refinamiento de Reglas:** Se ajustó la regla de 'Joven en Transición' para excluir
    perfiles con Capital Humano Alto, evitando conflictos con 'Comunicador Desafiado'
    o 'Profesional Subutilizado'.
2.  **Limpieza de Filas Huérfanas:** Elimina perfiles con baja pertenencia (<0.1) a cualquier
    arquetipo para corregir el sesgo detectado por SHAP.
3.  **Balanceo de Clases (Upsampling):** Aplica sobremuestreo a los arquetipos minoritarios
    para garantizar un mínimo de muestras, superando el desbalance natural de la encuesta.

Dependencias:
    - pandas
    - numpy
    - pyyaml
    - sklearn (resample)
    - src.profile_inference (lógica compartida)
    - src.constants

Ejecución:
    python src/data_processing.py
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
import traceback
from sklearn.utils import resample  # Necesario para la estrategia de balanceo

# Importación robusta de constantes y lógica compartida
try:
    from .constants import ALL_ARCHETYPES, TARGET_COLUMN
    from .profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )
except ImportError:
    # Fallback para ejecución directa como script principal
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
    Aplica las reglas expertas ("receta dorada") para calcular la pertenencia a cada arquetipo.
    
    Esta función implementa la lógica heurística definida en la tesis para asignar
    un grado de pertenencia (0.0 a 1.0) a cada uno de los 6 arquetipos de vulnerabilidad.
    
    NOTA: Esta función se utiliza SOLO durante la fase de entrenamiento offline para
    etiquetar los datos históricos.

    Args:
        df (pd.DataFrame): DataFrame con características de Fase 1 y scores MBTI simulados.

    Returns:
        pd.DataFrame: El DataFrame original enriquecido con 6 nuevas columnas 
                      `Pertenencia_{Arquetipo}`.
    """
    df_out = df.copy()
    logging.info("Calculando pertenencia a arquetipos (Reglas Expertas)...")

    # --- DEFINICIÓN DE REGLAS REFINADAS PARA EVITAR COLISIONES ---

    def _clasificar_comunicador_desafiado(r):
        """
        Regla para 'Comunicador Desafiado'.
        Perfil: Alto Capital Humano + Barreras de Comunicación.
        """
        # REGLA DE ORO (Excluyente): Debe tener Capital Humano ALTO.
        if r.get('CAPITAL_HUMANO') != '3_Alto':
            return 0.0
            
        pdif, slab = r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        ei = r.get('MBTI_EI_score_sim')
        
        es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        
        prob_base = 0.0
        if es_dificultad_com and es_inclusion_deficiente: 
            prob_base = 0.95 # Alta certeza
        elif es_inclusion_deficiente and pdif == '3_Tres_o_Mas_Dificultades':
             # Caso borde: múltiples dificultades a veces enmascaran lo comunicacional
             prob_base = 0.3
        
        if prob_base == 0.0: return 0.0
        
        # Moduladores MBTI: Introversión favorece este perfil
        factor_ei = 1.0 - (0.2 * ei) if pd.notna(ei) else 1.0
        return round(max(0.0, min(prob_base * factor_ei, 1.0)), 2)

    def _clasificar_navegante_informal(r):
        """
        Regla para 'Navegante Informal'.
        Perfil: Bajo Capital Humano + Proactividad/Informalidad.
        """
        # REGLA DE ORO (Excluyente): Capital Humano BAJO.
        if r.get('CAPITAL_HUMANO') != '1_Bajo':
            return 0.0
            
        slab, jp = r.get('Espectro_Inclusion_Laboral'), r.get('MBTI_JP_score_sim')
        
        # Debe estar en inclusión precaria o búsqueda
        if slab not in ['3_Inclusion_Precaria_Aprox', '2_Busqueda_Sin_Exito']:
            return 0.0
            
        prob_base = 0.9
        # Moduladores: Percepción (flexibilidad) favorece
        factor_jp = 1.0 + (0.2 * jp) if pd.notna(jp) else 1.0 
        return round(max(0.0, min(prob_base * factor_jp, 1.0)), 2)

    def _clasificar_profesional_subutilizado(r):
        """
        Regla para 'Profesional Subutilizado'.
        Perfil: Alto/Medio Capital Humano + Subempleo/Desempleo.
        """
        # REGLA DE ORO: Capital Humano ALTO o MEDIO.
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
        """
        Regla para 'Potencial Latente'.
        Perfil: Exclusión Total del Mercado + Barreras significativas.
        """
        # Foco: Exclusión total
        if r.get('Espectro_Inclusion_Laboral') != '1_Exclusion_del_Mercado':
            return 0.0
            
        pdif, ei = r.get('Perfil_Dificultad_Agrupado'), r.get('MBTI_EI_score_sim')
        
        prob_base = 0.6
        if pdif in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']: 
            prob_base = 0.95
        
        # Modulador: Introversión alta refuerza el aislamiento
        factor_ei = 1.0 - (0.3 * ei) if pd.notna(ei) else 1.0
        return round(max(0.0, min(prob_base * factor_ei, 1.0)), 2)

    def _clasificar_candidato_necesidades_sig(r):
        """
        Regla para 'Candidato con Necesidades Significativas'.
        Perfil: Barreras múltiples o severas que requieren apoyos extensos.
        """
        pdif = r.get('Perfil_Dificultad_Agrupado')
        if pdif in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']:
            return 0.9
        if pdif == '2_Dos_Dificultades':
            return 0.6
        return 0.0

    def _clasificar_joven_transicion(r):
        """
        Regla para 'Joven en Transición'.
        Perfil: Edad Joven + Primeros pasos laborales/educativos.
        """
        # Regla estricta de edad
        if r.get('GRUPO_ETARIO_INDEC') != '1_Joven_Adulto_Temprano (14-39)':
            return 0.0
        
        # --- NUEVA REGLA DE EXCLUSIÓN ---
        # Si tiene Capital Humano ALTO (Universitario), NO es transición simple,
        # es un perfil profesional (Subutilizado o Comunicador).
        if r.get('CAPITAL_HUMANO') == '3_Alto':
           return 0.0
           
        asiste = r.get('PC08') 
        if asiste == 1: return 0.9 # Asiste a educación
        
        slab = r.get('Espectro_Inclusion_Laboral')
        if slab == '2_Busqueda_Sin_Exito': 
            return 0.8 # Buscando primer empleo
            
        return 0.0

    # Mapeo de funciones a nombres de arquetipos
    arch_funcs = {
        ALL_ARCHETYPES[0]: _clasificar_comunicador_desafiado,
        ALL_ARCHETYPES[1]: _clasificar_navegante_informal,
        ALL_ARCHETYPES[2]: _clasificar_profesional_subutilizado,
        ALL_ARCHETYPES[3]: _clasificar_potencial_latente,
        ALL_ARCHETYPES[4]: _clasificar_candidato_necesidades_sig,
        ALL_ARCHETYPES[5]: _clasificar_joven_transicion,
    }

    # Aplicación de reglas fila por fila
    for name, func in arch_funcs.items():
        try:
            df_out[f'Pertenencia_{name}'] = df_out.apply(func, axis=1)
        except Exception as e:
            logging.error(f"Error aplicando regla para {name}: {e}")
            df_out[f'Pertenencia_{name}'] = 0.0

    return df_out


def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper para la fase 2 completa.
    
    Orquesta la simulación de MBTI y el cálculo de pertenencia a arquetipos.
    """
    logging.info("Ejecutando Fase 2: Ingeniería de Arquetipos...")
    df_mbti = _simulate_mbti_scores(df)
    df_archetyped = _calculate_archetype_membership(df_mbti)
    return df_archetyped


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL (ETL OFFLINE)
# ==============================================================================

if __name__ == '__main__':
    # Carga de configuración
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

    logging.info("--- ⚙️ Iniciando Pipeline de Procesamiento (FIX SHAP + BALANCEO) ---")
    
    # Carga de datos crudos (Manejo de encoding y delimitador)
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
    except Exception as e:
        logging.error(f"Error leyendo CSV: {e}")
        exit()

    # --- Pipeline de Transformación ---
    df_featured = run_feature_engineering(df_raw)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)

    # --- 1. FIX CRÍTICO: Limpieza de Filas Huérfanas (Todo 0) ---
    # Identificar filas que no activaron ninguna regla experta de manera significativa
    archetype_cols = [f'Pertenencia_{name}' for name in ALL_ARCHETYPES]
    df_fuzzified['MAX_SCORE'] = df_fuzzified[archetype_cols].max(axis=1)
    
    # Filtrar ruido: Score mínimo de 0.1 para considerar la fila válida
    # Esto previene que idxmax() seleccione el primer arquetipo por defecto en filas vacías.
    df_clean = df_fuzzified[df_fuzzified['MAX_SCORE'] > 0.1].copy()
    dropped_count = len(df_fuzzified) - len(df_clean)
    if dropped_count > 0:
        logging.warning(f"⚠️ Se eliminaron {dropped_count} filas huérfanas (Score < 0.1). Corrección anti-sesgo aplicada.")

    if df_clean.empty:
        logging.error("Error crítico: Dataset vacío tras limpieza.")
        exit()

    # --- Generación Target Base ---
    # Asignar la etiqueta 'hard' (clase dominante) basada en el mayor score de pertenencia
    df_clean[TARGET_COLUMN] = df_clean[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')
    
    # --- 2. ESTRATEGIA DE BALANCEO (OVERSAMPLING) ---
    logging.info("--- Aplicando Balanceo de Clases (Oversampling) ---")
    target_counts = df_clean[TARGET_COLUMN].value_counts()
    logging.info(f"Distribución Original:\n{target_counts}")

    MIN_SAMPLES = 1000 # Umbral mínimo de muestras para garantizar aprendizaje efectivo
    dfs_to_concat = [df_clean] # Iniciar con el dataset limpio original

    for archetype in ALL_ARCHETYPES:
        count = target_counts.get(archetype, 0)
        
        if 0 < count < MIN_SAMPLES:
            # Identificar la clase minoritaria
            df_minority = df_clean[df_clean[TARGET_COLUMN] == archetype]
            
            if not df_minority.empty:
                # Calcular cuántas muestras faltan para llegar al mínimo
                n_samples_needed = MIN_SAMPLES - count
                
                # Generar copias sintéticas (resampling con reemplazo)
                df_upsampled = resample(
                    df_minority, 
                    replace=True,     
                    n_samples=n_samples_needed,    
                    random_state=42
                )
                dfs_to_concat.append(df_upsampled)
                logging.info(f"  › {archetype}: Se añadieron +{n_samples_needed} copias (Total: {MIN_SAMPLES}).")
        elif count == 0:
            logging.warning(f"  ⚠️ {archetype}: 0 muestras encontradas. No se puede hacer upsampling.")

    # Combinar todo en un nuevo DataFrame balanceado y mezclar (shuffle)
    df_balanced = pd.concat(dfs_to_concat).sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info("--- Distribución Final Balanceada ---")
    logging.info("\n" + df_balanced[TARGET_COLUMN].value_counts().to_string())

    # --- Guardado de Datos ---
    feature_cols = [col for col in df_balanced.columns if '_memb' in col]
    df_training = df_balanced[feature_cols + [TARGET_COLUMN]].fillna(0.0)
    
    # Determinar ruta de salida relativa al script
    base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    
    # Guardar dataset de entrenamiento
    out_path = os.path.join(out_dir, 'cognitive_profiles.csv')
    df_training.to_csv(out_path, index=False)
    logging.info(f"✅ Archivo de entrenamiento BALANCEADO guardado: {out_path}")
    
    # Generar Perfiles Demo (Usando el dataset balanceado para asegurar variedad)
    demo_list = []
    for arch in df_balanced[TARGET_COLUMN].unique():
        subset = df_balanced[df_balanced[TARGET_COLUMN] == arch]
        if len(subset) > 0:
            # Intentar tomar ejemplos únicos si existen para la demo
            unique_subset = subset.drop_duplicates()
            sample_source = unique_subset if len(unique_subset) >= 2 else subset
            demo_list.append(sample_source.sample(min(2, len(sample_source)), random_state=42))
    
    if demo_list:
        df_demo = pd.concat(demo_list)
        df_demo['ID'] = [f'Demo_{i}' for i in range(len(df_demo))]
        df_demo.set_index('ID', inplace=True)
        df_demo.to_csv(os.path.join(out_dir, 'demo_profiles.csv'))
        logging.info("✅ Perfiles demo generados.")
