"""
Módulo de Procesamiento de Datos y Generación de Arquetipos (Pipeline ETL).

Este script es el corazón de la ingeniería de datos del proyecto de tesis. Su responsabilidad
es transformar los datos crudos de la encuesta ENDIS en un dataset de entrenamiento
robusto y balanceado para el Tutor Cognitivo basado en Random Forest.

Estrategia Metodológica:
------------------------
El pipeline implementa una estrategia de "Defensa en Profundidad" para mitigar el sesgo
estadístico inherente en los datos reales (donde los jóvenes suelen ser etiquetados
automáticamente como 'transición' independientemente de su formación):

1.  **Inyección de Datos Sintéticos**: Generación artificial de casos borde (ej. jóvenes
    universitarios con discapacidad) que son raros en la muestra real pero críticos para la lógica.
2.  **Lógica Heurística Rica ("Fuzzy Rules")**: Asignación de arquetipos basada en reglas
    de negocio complejas que consideran educación, discapacidad, situación laboral y
    rasgos de personalidad simulados (MBTI).
3.  **Candados Lógicos ("Hard Constraints")**: Reglas de exclusión estricta (vetos) que
    impiden, por ejemplo, que un universitario sea clasificado como 'Joven en Transición'.
4.  **Limpieza Forense ("Relabeling")**: Auditoría post-generación para corregir contradicciones
    que hayan sobrevivido a las reglas anteriores.
5.  **Balanceo Híbrido**: Uso de Upsampling (para minorías) y Downsampling (para mayorías)
    para asegurar que el modelo aprenda todas las clases por igual.

Autora: [Tu Nombre/Rol]
Fecha: 2025
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
from sklearn.utils import resample

# Importación de módulos del proyecto con manejo de rutas relativas/absolutas
try:
    from src.constants import ALL_ARCHETYPES, TARGET_COLUMN
    from src.profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.constants import ALL_ARCHETYPES, TARGET_COLUMN
    from src.profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores,
        run_fuzzification
    )

# Configuración del Logging para trazabilidad del proceso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 1. INYECCIÓN DE DATOS SINTÉTICOS
# ==============================================================================
def _inject_synthetic_data() -> pd.DataFrame:
    """
    Genera un conjunto de datos sintéticos para reforzar patrones estadísticamente débiles.

    El propósito es combatir la "inercia estadística" de la base de datos real (ENDIS),
    donde ciertas combinaciones de atributos (ej. Joven + Título Universitario + Discapacidad)
    son casi inexistentes, llevando al modelo a ignorarlas.

    Returns:
        pd.DataFrame: DataFrame conteniendo las filas sintéticas generadas.
    """
    logging.info("💉 Inyectando datos sintéticos para romper la inercia estadística...")
    synthetic_rows = []
    
    # CASO A: COMUNICADOR DESAFIADO
    # Perfil: Joven, Universitario, con dificultad de Habla o Auditiva.
    # Objetivo: Enseñar al modelo que 'Joven' + 'Título' NO es igual a 'Transición'.
    for i in range(500):
        row = {
            'ID': f'SYN_COM_DES_{i}',
            'dificultad_total': 1, 
            'tipo_dificultad': np.random.choice([6, 3]), # 6: Habla, 3: Auditiva
            'dificultades': 1,
            'MNEA': 5, # Universitario Completo (Variable Crítica)
            'edad_agrupada': np.random.choice([2, 3]), # Joven (15-29) y Adulto Joven
            'Estado_ocup': 2, # Desocupado
            'cat_ocup': 9, 'certificado': 1, 'PC08': 9, 'pc03': 1, 'tipo_hogar': 2
        }
        synthetic_rows.append(row)

    # CASO B: POTENCIAL LATENTE CALIFICADO
    # Perfil: Universitario en inactividad laboral.
    # Objetivo: Diferenciarlo del Comunicador por su 'Estado_ocup' (Inactivo vs Desocupado).
    for i in range(200):
        row = {
            'ID': f'SYN_POT_LAT_{i}',
            'dificultad_total': 1, 'tipo_dificultad': 1, # Motora
            'MNEA': 5, # Universitario
            'edad_agrupada': 3,
            'Estado_ocup': 3, # Inactivo (Variable Crítica)
            'cat_ocup': 9, 'certificado': 1, 'PC08': 9, 'pc03': 1, 'tipo_hogar': 2
        }
        synthetic_rows.append(row)

    return pd.DataFrame(synthetic_rows)

# ==============================================================================
# 2. INGENIERÍA DE ARQUETIPOS (REGLAS + CANDADOS)
# ==============================================================================
def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el grado de pertenencia de cada individuo a los 6 arquetipos definidos.

    Utiliza una combinación de lógica difusa (probabilidades basadas en rasgos) y 
    candados lógicos (reglas de exclusión estrictas) para asegurar la coherencia.

    Args:
        df (pd.DataFrame): DataFrame con features enriquecidos y scores MBTI simulados.

    Returns:
        pd.DataFrame: DataFrame original con columnas nuevas 'Pertenencia_{Arquetipo}'.
    """
    df_out = df.copy()

    # --- REGLA 1: COMUNICADOR DESAFIADO ---
    def _rule_comunicador(r):
        """Identifica profesionales con barreras específicas de comunicación."""
        # CANDADO: Solo Capital Humano Alto (Educación Superior)
        if r.get('CAPITAL_HUMANO') != '3_Alto': return 0.0
        
        pdif = r.get('Perfil_Dificultad_Agrupado')
        slab = r.get('Espectro_Inclusion_Laboral')
        ei = r.get('MBTI_EI_score_sim', 0.5) # Introversión/Extroversión
        
        # Filtro: Debe estar buscando trabajo o en empleo precario
        if slab not in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox']: return 0.0

        es_comunicacion = pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica']
        
        prob = 0.0
        if es_comunicacion: 
            prob = 0.95 
        elif pdif == '3_Tres_o_Mas_Dificultades': 
            prob = 0.4 # Posible, pero menos claro
            
        if prob == 0.0: return 0.0
        
        # Matiz MBTI: La introversión (EI bajo) puede acentuar la percepción de barrera
        factor = 1.0 + (0.1 * (1 - ei)) 
        return round(min(prob * factor, 1.0), 2)

    # --- REGLA 2: NAVEGANTE INFORMAL ---
    def _rule_navegante(r):
        """Identifica trabajadores informales con alta adaptabilidad."""
        # CANDADO: Generalmente Capital Bajo (No universitarios)
        if r.get('CAPITAL_HUMANO') == '3_Alto': return 0.0
        
        slab = r.get('Espectro_Inclusion_Laboral')
        jp = r.get('MBTI_JP_score_sim', 0.5) # Judging/Perceiving
        
        if slab not in ['3_Inclusion_Precaria_Aprox', '2_Busqueda_Sin_Exito']: return 0.0
        
        prob = 0.85
        # Matiz MBTI: Alta 'Percepción' (flexibilidad/improvisación) favorece este perfil
        factor = 1.0 + (0.1 * jp)
        return round(min(prob * factor, 1.0), 2)

    # --- REGLA 3: PROFESIONAL SUBUTILIZADO ---
    def _rule_profesional(r):
        """Identifica capital humano alto en roles que no aprovechan sus competencias."""
        # CANDADO: Requiere Capital Medio o Alto
        if r.get('CAPITAL_HUMANO') == '1_Bajo': return 0.0
        
        pdif = r.get('Perfil_Dificultad_Agrupado')
        slab = r.get('Espectro_Inclusion_Laboral')
        
        if slab not in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox']: return 0.0
        
        # Dificultades "manejables" con ajustes razonables estándar
        es_dificultad_menor = pdif in ['0_Sin_Dificultad_Registrada', '4_Solo_Certificado', '1A_Motora_Unica', '1B_Visual_Unica']
        
        if es_dificultad_menor: return 0.90
        return 0.0

    # --- REGLA 4: POTENCIAL LATENTE ---
    def _rule_potencial(r):
        """Identifica personas inactivas laboralmente por desaliento o barreras sistémicas."""
        slab = r.get('Espectro_Inclusion_Laboral')
        pdif = r.get('Perfil_Dificultad_Agrupado')
        
        # CANDADO: Debe estar en Inactividad (Exclusión del mercado)
        if slab != '1_Exclusion_del_Mercado': return 0.0
        
        prob = 0.6
        # Mayor probabilidad si hay dificultades severas o desaliento profesional
        if pdif in ['1E_Autocuidado_Unica', '3_Tres_o_Mas_Dificultades']: prob = 0.95
        elif r.get('CAPITAL_HUMANO') == '3_Alto': prob = 0.85 
        
        return prob

    # --- REGLA 5: CANDIDATO CON NECESIDADES SIGNIFICATIVAS ---
    def _rule_necesidades(r):
        """Identifica perfiles que requieren apoyos intensivos y personalizados."""
        pdif = r.get('Perfil_Dificultad_Agrupado')
        
        # CANDADO: Si es Universitario, se requiere dificultad extrema para caer aquí
        if r.get('CAPITAL_HUMANO') == '3_Alto' and pdif not in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']:
             return 0.0

        if pdif in ['3_Tres_o_Mas_Dificultades', '1E_Autocuidado_Unica']: return 0.95
        return 0.0

    # --- REGLA 6: JOVEN EN TRANSICIÓN (LA CRÍTICA) ---
    def _rule_joven(r):
        """
        Identifica jóvenes en etapa formativa o de primera inserción.
        Esta regla contiene el FIX PRINCIPAL para el sesgo de edad.
        """
        grupo_edad = r.get('GRUPO_ETARIO_INDEC')
        capital = r.get('CAPITAL_HUMANO')
        asiste_educacion = r.get('PC08')
        
        # CANDADO 1: EDAD (Solo jóvenes adultos tempranos)
        if grupo_edad != '1_Joven_Adulto_Temprano (14-39)': return 0.0
        
        # ---------------------------------------------------------
        # CANDADO 2 (CRÍTICO): EXCLUSIÓN POR NIVEL EDUCATIVO
        # Si tiene título universitario (Capital Alto), NO es transición.
        # Debe ser clasificado como Profesional o Comunicador.
        # ---------------------------------------------------------
        if capital == '3_Alto': return 0.0
        
        prob = 0.0
        if asiste_educacion == 1: prob = 0.95 # Asiste actualmente
        else: prob = 0.75 # Joven sin capital alto y sin asistir
        
        return prob

    # Mapeo de reglas y ejecución
    reglas = {
        'Com_Desafiado': _rule_comunicador,
        'Nav_Informal': _rule_navegante,
        'Prof_Subutil': _rule_profesional,
        'Potencial_Latente': _rule_potencial,
        'Cand_Nec_Sig': _rule_necesidades,
        'Joven_Transicion': _rule_joven,
    }

    for arch, func in reglas.items():
        try:
            df_out[f'Pertenencia_{arch}'] = df_out.apply(func, axis=1)
        except Exception as e:
            logging.error(f"Error aplicando regla para {arch}: {e}")
            df_out[f'Pertenencia_{arch}'] = 0.0

    return df_out

# ==============================================================================
# 3. LIMPIEZA DE ETIQUETAS (RELABELING POST-GENERACIÓN)
# ==============================================================================
def _fix_inconsistent_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza una auditoría final sobre las etiquetas generadas para corregir contradicciones lógicas.

    Incluso con reglas estrictas, datos sucios o combinaciones extrañas en la base original
    pueden generar etiquetas inconsistentes. Esta función actúa como un "filtro de calidad".

    Correcciones específicas implementadas:
    - Reclasificación de 'Joven_Transicion' si posee Título Universitario.

    Args:
        df (pd.DataFrame): DataFrame con la columna TARGET_COLUMN asignada.

    Returns:
        pd.DataFrame: DataFrame con etiquetas corregidas.
    """
    df_clean = df.copy()
    
    # DETECTAR: Etiquetado como 'Joven_Transicion' PERO con Capital Humano Alto (> 0.5)
    mask_error = (df_clean[TARGET_COLUMN] == 'Joven_Transicion') & (df_clean['CH_Alto_memb'] > 0.5)
    
    count_errors = mask_error.sum()
    if count_errors > 0:
        logging.warning(f"🔄 RELABELING: Corrigiendo {count_errors} inconsistencias (Joven Transición con Título Univ).")
        
        # Sub-regla A: Si tiene dificultad de Habla o Sensorial -> Com_Desafiado
        mask_com = mask_error & ((df_clean['PD_ComCog_memb'] > 0.5) | (df_clean['PD_Sensorial_memb'] > 0.5))
        df_clean.loc[mask_com, TARGET_COLUMN] = 'Com_Desafiado'
        
        # Sub-regla B: El resto -> Prof_Subutil
        mask_prof = mask_error & (~mask_com)
        df_clean.loc[mask_prof, TARGET_COLUMN] = 'Prof_Subutil'
        
        logging.info(f"   › Reasignados a Com_Desafiado: {mask_com.sum()}")
        logging.info(f"   › Reasignados a Prof_Subutil: {mask_prof.sum()}")
    
    return df_clean

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquestador principal de la generación de arquetipos.
    
    Pasos:
    1. Simulación de rasgos de personalidad (MBTI).
    2. Cálculo de membresía a arquetipos usando reglas y candados.
    
    Args:
        df (pd.DataFrame): DataFrame con features procesados.
        
    Returns:
        pd.DataFrame: DataFrame enriquecido con columnas de pertenencia a arquetipos.
    """
    # 1. Simular MBTI (aporta "textura" y realismo psicológico a los datos)
    df_mbti = _simulate_mbti_scores(df)
    
    # 2. Calcular Pertenencias (Núcleo de la lógica de negocio)
    df_archetyped = _calculate_archetype_membership(df_mbti)
    
    return df_archetyped

if __name__ == '__main__':
    # --- Bloque Principal de Ejecución ---
    
    # 1. Carga de Configuración
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error cargando config.yaml: {e}")
        config = {'data_paths': {'raw_data': 'data/ENDIS_Completo.csv'}}

    RAW_DATA_PATH = config.get('data_paths', {}).get('raw_data', 'data/ENDIS_Completo.csv')
    
    logging.info("--- ⚙️ Iniciando Pipeline de Procesamiento de Datos ---")

    # 2. Carga de Datos Crudos
    if os.path.exists(RAW_DATA_PATH):
        try:
            df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
            logging.info(f"Datos crudos cargados: {len(df_raw)} filas.")
        except Exception as e:
            logging.error(f"Error leyendo CSV: {e}"); exit()
    else:
        logging.error(f"No se encuentra el archivo: {RAW_DATA_PATH}"); exit()

    # 3. Inyección de Datos Sintéticos (Capa 1 de Defensa)
    df_synthetic = _inject_synthetic_data()
    df_combined = pd.concat([df_raw, df_synthetic], ignore_index=True)
    logging.info(f"Dataset combinado: {len(df_combined)} filas (Originales + Sintéticos).")

    # 4. Feature Engineering y Fuzzificación
    df_featured = run_feature_engineering(df_combined)
    df_archetyped = run_archetype_engineering(df_featured)
    df_fuzzified = run_fuzzification(df_archetyped)

    # 5. Asignación del Target Inicial (Winner takes all)
    archetype_cols = [f'Pertenencia_{name}' for name in ALL_ARCHETYPES]
    
    # Filtro de calidad: eliminar filas donde ninguna regla aplicó significativamente
    df_fuzzified['MAX_SCORE'] = df_fuzzified[archetype_cols].max(axis=1)
    df_clean = df_fuzzified[df_fuzzified['MAX_SCORE'] > 0.1].copy()
    
    # Asignar etiqueta ganadora
    df_clean[TARGET_COLUMN] = df_clean[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')

    # 6. Limpieza de Etiquetas / Relabeling (Capa 3 de Defensa)
    df_corrected = _fix_inconsistent_labels(df_clean)

    # 7. Balanceo Híbrido (Capa 4 de Defensa)
    logging.info("--- ⚖️ Ejecutando Balanceo Híbrido (Tijera) ---")
    
    MIN_SAMPLES = 1000  # Piso para minorías (Upsampling)
    MAX_SAMPLES = 3000  # Techo para mayorías (Downsampling)
    
    dfs_balanced = []
    
    counts_before = df_corrected[TARGET_COLUMN].value_counts()
    logging.info(f"Distribución antes del balanceo:\n{counts_before}")

    for archetype in ALL_ARCHETYPES:
        df_arch = df_corrected[df_corrected[TARGET_COLUMN] == archetype]
        count = len(df_arch)
        
        if count == 0:
            logging.warning(f"⚠️ Arquetipo vacío: {archetype}")
            continue
            
        if count < MIN_SAMPLES:
            # Upsample: Duplicar muestras para alcanzar el mínimo
            df_res = resample(df_arch, replace=True, n_samples=MIN_SAMPLES, random_state=42)
            logging.info(f"  ⬆️ Upsample {archetype}: {count} -> {MIN_SAMPLES}")
        elif count > MAX_SAMPLES:
            # Downsample: Reducir muestras para no exceder el máximo
            df_res = resample(df_arch, replace=False, n_samples=MAX_SAMPLES, random_state=42)
            logging.info(f"  ⬇️ Downsample {archetype}: {count} -> {MAX_SAMPLES}")
        else:
            # Mantener: Cantidad ideal
            df_res = df_arch
            
        dfs_balanced.append(df_res)

    df_final = pd.concat(dfs_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 8. Guardado de Resultados
    # Seleccionar solo columnas numéricas (fuzzy features) + Target para el entrenamiento
    feature_cols = [col for col in df_final.columns if '_memb' in col]
    df_training = df_final[feature_cols + [TARGET_COLUMN]].fillna(0.0)
    
    out_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cognitive_profiles.csv')
    
    df_training.to_csv(out_path, index=False)
    logging.info(f"✅ Dataset de entrenamiento guardado en: {out_path}")
    logging.info(f"Distribución Final:\n{df_final[TARGET_COLUMN].value_counts()}")
    
    # Generar Demo Set (pequeña muestra para pruebas rápidas de inferencia)
    demo_samples = []
    for arch in ALL_ARCHETYPES:
        subset = df_final[df_final[TARGET_COLUMN] == arch]
        if not subset.empty:
            demo_samples.append(subset.iloc[0:2]) # Tomar 2 ejemplos de cada clase
    if demo_samples:
        pd.concat(demo_samples).to_csv(os.path.join(out_dir, 'demo_profiles.csv'), index=False)

    logging.info("🏁 Pipeline finalizado con éxito.")
