"""Pipeline de procesamiento de datos para el Tutor Cognitivo. (Refactorizado Épica 2)

Este script ejecuta un pipeline ETL (Extracción, Transformación, Carga) completo
para generar los datos necesarios para el entrenamiento del modelo cognitivo
y los perfiles de demostración.

Utiliza la lógica de ingeniería de características y fuzzificación importada
desde 'src.profile_inference' para asegurar la consistencia con la
inferencia en tiempo real.

Pasos:
1. Carga los datos crudos de la encuesta.
2. Ejecuta ingeniería de características (importada).
3. Simula scores MBTI y calcula pertenencia a arquetipos (receta dorada).
4. Ejecuta fuzzificación (importada).
5. Limpia y valida los datos.
6. Genera 'cognitive_profiles.csv' (para entrenamiento) y 'demo_profiles.csv'.
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
    # Fallback para ejecución directa
    ALL_ARCHETYPES = ['Com_Desafiado', 'Nav_Informal', 'Prof_Subutil', 'Potencial_Latente', 'Cand_Nec_Sig', 'Joven_Transicion']
    TARGET_COLUMN = 'ARQUETIPO_PRED'
    logging.warning("No se pudo importar 'src.constants'. Usando valores fallback.")

# --- ¡IMPORTACIONES CLAVE DE LA REFACTORIZACIÓN (ÉPICA 2)! ---
# Importamos la lógica de features/fuzzificación desde el módulo de inferencia
try:
    from src.profile_inference import (
        run_feature_engineering,
        _simulate_mbti_scores, # Importar también la simulación MBTI si se usa aquí
        run_fuzzification
    )
    logging.info("Módulo 'src.profile_inference' cargado exitosamente.")
except ImportError as e:
    logging.error(f"Error crítico: No se pudo importar 'src.profile_inference'. {e}")
    # Detener ejecución si el módulo compartido falta
    exit()
# --- FIN IMPORTACIONES CLAVE ---


# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# CLASES DE ÁRBOL DIFUSO (EVIDENCIA DE INVESTIGACIÓN)
# (Estas clases permanecen como documentación, no se usan en el pipeline)
# ==============================================================================
class FuzzyDecisionTreeNode:
    """Representa un único nodo en el Árbol de Decisión Difuso (IF_HUPM)."""
    # ... (código sin cambios) ...
    def __init__(self, feature_name=None, threshold=None, branches=None, leaf_value=None, is_uncertain=None, class_probabilities=None, n_samples=0):
        self.feature_name = feature_name
        self.threshold = threshold
        self.branches = branches
        self.leaf_value = leaf_value
        self.is_uncertain = is_uncertain
        self.class_probabilities = class_probabilities
        self.n_samples = n_samples

class IF_HUPM:
    """Implementación de un Árbol de Decisión Difuso simple (IF-HUPM)."""
    # ... (código sin cambios) ...
    def __init__(self, min_samples_split=2, max_depth=10, uncertainty_threshold=0.1):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.uncertainty_threshold = uncertainty_threshold
        self.root = None

    def _calculate_entropy(self, y):
        n = len(y); counts = np.array(list(Counter(y).values()))
        if n == 0: return 0
        probabilities = counts / n
        return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

    def _information_gain(self, y, left_y, right_y):
        n, n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0: return 0
        parent_entropy = self._calculate_entropy(y)
        child_entropy = (n_left / n) * self._calculate_entropy(left_y) + (n_right / n) * self._calculate_entropy(right_y)
        return parent_entropy - child_entropy

    def _find_best_split(self, X, y):
        best_gain, best_feature, best_threshold = -1, None, None
        if len(y) < self.min_samples_split: return None, None, None
        num_features = X.shape[1]
        features_to_check = np.random.choice(X.columns, size=int(np.sqrt(num_features)) if num_features > 1 else 1, replace=False)
        for feature_name in features_to_check:
            thresholds = np.unique(X[feature_name])
            for threshold in thresholds:
                left_indices = y.index[X[feature_name] <= threshold]; right_indices = y.index[X[feature_name] > threshold]
                if len(left_indices) == 0 or len(right_indices) == 0: continue
                gain = self._information_gain(y, y.loc[left_indices], y.loc[right_indices])
                if gain > best_gain: best_gain, best_feature, best_threshold = gain, feature_name, threshold
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y.unique()) == 1 or len(y) < self.min_samples_split: return self._create_leaf_node(y)
        feature, threshold, gain = self._find_best_split(X, y)
        if gain is None or gain <= 0: return self._create_leaf_node(y)
        left_indices = y.index[X[feature] <= threshold]; right_indices = y.index[X[feature] > threshold]
        branches = {'<= ' + str(threshold): self._build_tree(X.loc[left_indices], y.loc[left_indices], depth + 1),
                    '> ' + str(threshold): self._build_tree(X.loc[right_indices], y.loc[right_indices], depth + 1)}
        return FuzzyDecisionTreeNode(feature_name=feature, threshold=threshold, branches=branches, n_samples=len(y))

    def _create_leaf_node(self, y):
        counts = Counter(y)
        if not counts: return FuzzyDecisionTreeNode(leaf_value="Incierto (Hoja Vacía)", is_uncertain=True, class_probabilities={}, n_samples=0)
        total = len(y); probabilities = {k: v / total for k, v in counts.items()}
        most_common_class = max(probabilities, key=probabilities.get)
        sorted_probs = sorted(probabilities.values(), reverse=True)
        is_uncertain = (len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < self.uncertainty_threshold) or (probabilities[most_common_class] < (1. - self.uncertainty_threshold))
        leaf_value = most_common_class if not is_uncertain else f"Incierto (posiblemente {most_common_class})"
        return FuzzyDecisionTreeNode(leaf_value=leaf_value, is_uncertain=is_uncertain, class_probabilities=probabilities, n_samples=total)

    def fit(self, X, y): self.root = self._build_tree(X, y)
    def _predict_single(self, x, node):
        if node.leaf_value: return node.leaf_value
        value = x.get(node.feature_name)
        if value is None: return "Incierto (Valor Faltante)"
        branch_key = '<= ' + str(node.threshold) if value <= node.threshold else '> ' + str(node.threshold)
        return self._predict_single(x, node.branches[branch_key])
    def predict(self, X): return X.apply(self._predict_single, axis=1, args=(self.root,))

# ==============================================================================
# FASE 1: INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================

# --- ¡FUNCIÓN ELIMINADA! ---
# La función 'run_feature_engineering' ahora se importa desde
# 'src.profile_inference' en la parte superior del script.

# ==============================================================================
# FASE 2: INGENIERÍA DE ARQUETIPOS (SOLO PARA ENTRENAMIENTO)
# ==============================================================================

# --- ¡FUNCIÓN ELIMINADA! ---
# La función '_simulate_mbti_scores' ahora se importa desde
# 'src.profile_inference'.


def _calculate_archetype_membership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas expertas ("receta dorada") para calcular la pertenencia
    a cada arquetipo. ESTA FUNCIÓN ES SOLO PARA EL ENTRENAMIENTO OFFLINE.

    Args:
        df (pd.DataFrame): DataFrame con características de Fase 1 y scores MBTI simulados.

    Returns:
        pd.DataFrame: El DataFrame con 6 nuevas columnas `Pertenencia_` (scores 0-1).
    """
    df_out = df.copy()
    logging.info("Calculando pertenencia a arquetipos (receta dorada)...")

    # --- INICIO DE LAS REGLAS DE ALTO RENDIMIENTO ---
    # (Las funciones internas _clasificar_... permanecen aquí)
    def _clasificar_comunicador_desafiado(r):
        ch, pdif, slab, get = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral'), r.get('GRUPO_ETARIO_INDEC')
        ei, sn = r.get('MBTI_EI_score_sim'), r.get('MBTI_SN_score_sim')
        es_capital_alto = (ch == '3_Alto'); es_dificultad_com = (pdif in ['1F_Habla_Comunicacion_Unica', '1C_Auditiva_Unica'])
        es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
        es_dificultad_multiple = (pdif in ['2_Dos_Dificultades', '3_Tres_o_Mas_Dificultades']) and not es_dificultad_com
        prob_base = 0.0
        if es_capital_alto and es_dificultad_com and es_inclusion_deficiente: prob_base = 0.9 if get == '1_Joven_Adulto_Temprano (14-39)' else 0.75 if get == '2_Adulto_Medio (40-64)' else 0.65
        elif es_capital_alto and es_inclusion_deficiente and es_dificultad_multiple: prob_base = 0.2
        if prob_base == 0.0: return 0.0
        factor_ei = 1.0 - (0.2 * ei) if pd.notna(ei) else 1.0
        factor_sn = 1.1 if pd.notna(sn) and sn == 0.5 else (0.9 if pd.notna(sn) and sn == -0.5 else 1.0)
        prob_final = prob_base * max(0.8, min(factor_ei, 1.2)) * factor_sn
        return round(max(0.0, min(prob_final, 1.0)), 2)

    def _clasificar_navegante_informal(r):
        ch, pdif, slab = r.get('CAPITAL_HUMANO'), r.get('Perfil_Dificultad_Agrupado'), r.get('Espectro_Inclusion_Laboral')
        sn, jp = r.get('MBTI_SN_score_sim'), r.get('MBTI_JP_score_sim')
        es_capital_bajo = (ch == '1_Bajo'); es_inclusion_deficiente = (slab in ['3_Inclusion_Precaria_Aprox', '2_Busqueda_Sin_Exito'])
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
        es_capital_alto = (ch == '3_Alto'); es_inclusion_deficiente = (slab in ['2_Busqueda_Sin_Exito', '3_Inclusion_Precaria_Aprox'])
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
            if asiste == 1: prob_base = 0.85 # Asiste a establecimiento educativo
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
        # Aplicar la función de clasificación correspondiente
        try:
            df_out[f'Pertenencia_{name}'] = df_out.apply(func, axis=1)
        except Exception as e:
            logging.error(f"Error aplicando regla para {name}: {e}")
            df_out[f'Pertenencia_{name}'] = 0.0 # Asignar 0 en caso de error

    logging.info("Pertenencia a arquetipos calculada.")
    return df_out


def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta la Fase 2: Simulación MBTI (importada) y cálculo de pertenencia
    a arquetipos (local, receta dorada).
    """
    logging.info("Ejecutando Fase 2: Ingeniería de Arquetipos...")
    # Usar la función _simulate_mbti_scores importada
    df_mbti = _simulate_mbti_scores(df)
    # Usar la función _calculate_archetype_membership definida localmente
    df_archetyped = _calculate_archetype_membership(df_mbti)
    logging.info("Fase 2 completada.")
    return df_archetyped

# ==============================================================================
# FASE 3: FUZZIFICACIÓN
# ==============================================================================

# --- ¡FUNCIÓN ELIMINADA! ---
# La función 'run_fuzzification' y sus sub-funciones ahora se importan desde
# 'src.profile_inference'.

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    """
    Orquesta la ejecución completa del pipeline ETL offline.
    """
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f: # Añadir encoding
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Error crítico: No se encontró 'config.yaml'.")
        exit()
    except yaml.YAMLError as e: # Capturar error específico de YAML
        logging.error(f"Error crítico al parsear 'config.yaml': {e}")
        exit()
    except Exception as e: # Capturar otros errores de lectura
        logging.error(f"Error inesperado al leer 'config.yaml': {e}")
        exit()


    RAW_DATA_PATH = config.get('data_paths', {}).get('raw_data')
    if not RAW_DATA_PATH:
        logging.error("Error crítico: 'data_paths.raw_data' no definido en 'config.yaml'.")
        exit()

    logging.info("--- ⚙️ Iniciando Pipeline de Procesamiento de Datos (Offline ETL) ---")
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"Error crítico: No se encontró archivo crudo en '{RAW_DATA_PATH}'.")
        exit() # Detener si el archivo de entrada no existe

    logging.info(f"Cargando datos crudos desde: {RAW_DATA_PATH}")
    try:
        # Especificar la codificación y manejar errores de parseo
        df_raw = pd.read_csv(RAW_DATA_PATH, delimiter=';', encoding='latin1', low_memory=False, on_bad_lines='warn')
    except pd.errors.ParserError as e:
        logging.error(f"Error crítico al parsear CSV '{RAW_DATA_PATH}': {e}")
        exit()
    except FileNotFoundError: # Doble chequeo por si acaso
         logging.error(f"Error crítico: No se encontró archivo crudo en '{RAW_DATA_PATH}' (FileNotFoundError).")
         exit()
    except Exception as e:
        logging.error(f"Error inesperado al cargar CSV '{RAW_DATA_PATH}': {e}")
        exit()


    # --- Ejecución de las Fases del Pipeline ---
    try:
        # ¡Usar las funciones importadas donde corresponda!
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured) # Llama a _simulate_mbti (importada) y _calculate (local)
        df_fuzzified = run_fuzzification(df_archetyped) # Importada
    except Exception as e:
        logging.error(f"Error durante las fases de procesamiento: {e}")
        logging.error(traceback.format_exc()) # Imprimir traceback completo
        exit()

    # --- Fase de Limpieza y Validación Final ---
    logging.info("Ejecutando Fase 4: Limpieza y Validación de Datos...")
    feature_cols = [col for col in df_fuzzified.columns if '_memb' in col]
    if not feature_cols:
        logging.error("Error crítico: No se generaron columnas de features ('_memb') tras la fuzzificación.")
        exit()

    # Convertir a numérico y rellenar NaNs
    for col in feature_cols:
        df_fuzzified[col] = pd.to_numeric(df_fuzzified[col], errors='coerce')
    df_fuzzified[feature_cols] = df_fuzzified[feature_cols].fillna(0.0)
    # Verificar que no queden NaNs en las features
    if df_fuzzified[feature_cols].isnull().values.any():
        logging.error("Error crítico: Aún existen valores NaN en las columnas de features después de fillna(0.0).")
        exit()
    logging.info("Fase 4 completada. Features limpias.")

    # --- Generación de la Columna Objetivo ---
    logging.info("Generando columna objetivo para entrenamiento...")
    archetype_cols = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
    if not archetype_cols:
        logging.error("Error crítico: No se encontraron columnas 'Pertenencia_' para determinar el target.")
        exit()
    # Usar idxmax para obtener el nombre del arquetipo con mayor pertenencia
    df_fuzzified[TARGET_COLUMN] = df_fuzzified[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')

    # Validar que la columna objetivo no tenga NaNs
    if df_fuzzified[TARGET_COLUMN].isnull().any():
        logging.warning(f"Advertencia: Se encontraron NaNs en la columna objetivo '{TARGET_COLUMN}'. Filas afectadas serán excluidas.")
        # Opcional: Investigar por qué hay NaNs aquí
        # print(df_fuzzified[df_fuzzified[TARGET_COLUMN].isnull()][archetype_cols])
        df_fuzzified.dropna(subset=[TARGET_COLUMN], inplace=True)
        if df_fuzzified.empty:
            logging.error("Error crítico: Todas las filas tenían NaN en la columna objetivo. No se puede continuar.")
            exit()

    # --- Verificación de Distribución de Clases ---
    logging.info("--- Distribución Final de Arquetipos (Target) ---")
    logging.info("\n" + df_fuzzified[TARGET_COLUMN].value_counts().to_string())
    logging.info("-" * 60)

    # Seleccionar solo las columnas necesarias para el archivo de entrenamiento
    df_cognitive_training = df_fuzzified[feature_cols + [TARGET_COLUMN]]

    # --- Guardado Permanente ---
    logging.info("Configurando rutas de guardado permanente...")
    # Determinar directorio base de forma robusta
    try:
        # Asumir que RAW_DATA_PATH es /content/drive/MyDrive/Tesis/.../base.csv
        # Queremos guardar en /content/drive/MyDrive/Tesis/.../data/
        thesis_project_dir = os.path.dirname(os.path.dirname(RAW_DATA_PATH))
        OUTPUT_DATA_DIR = os.path.join(thesis_project_dir, 'data')
    except Exception as e:
        logging.error(f"Error determinando directorio de salida: {e}. Usando 'data' local.")
        OUTPUT_DATA_DIR = 'data' # Fallback a directorio local

    try:
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    except OSError as e:
        logging.error(f"Error crítico: No se pudo crear el directorio de salida '{OUTPUT_DATA_DIR}': {e}")
        exit()


    # Guardar archivo de entrenamiento
    OUTPUT_TRAINING_PATH = os.path.join(OUTPUT_DATA_DIR, 'cognitive_profiles.csv')
    try:
        df_cognitive_training.to_csv(OUTPUT_TRAINING_PATH, index=False)
        logging.info(f"✅ Archivo de entrenamiento guardado en: {OUTPUT_TRAINING_PATH}")
    except Exception as e:
        logging.error(f"Error crítico al guardar '{OUTPUT_TRAINING_PATH}': {e}")
        exit() # Detener si no se puede guardar el archivo principal

    # --- Creación del Dataset de Demostración Curado ---
    logging.info("Creando archivo de demostración curado...")
    learned_archetypes = df_fuzzified[TARGET_COLUMN].unique().tolist()
    demo_profiles_list = []
    # Seleccionar hasta 2 ejemplos por arquetipo
    for archetype in learned_archetypes:
        subset = df_fuzzified[df_fuzzified[TARGET_COLUMN] == archetype]
        sample_size = min(2, len(subset))
        if sample_size > 0:
            demo_profiles_list.append(subset.sample(n=sample_size, random_state=42))

    if demo_profiles_list:
        df_demo = pd.concat(demo_profiles_list)
        # Usar un índice más descriptivo
        df_demo['ID'] = [f'{row[TARGET_COLUMN]}_Demo_{i+1}' for i, (idx, row) in enumerate(df_demo.iterrows())]
        df_demo.set_index('ID', inplace=True)

        OUTPUT_DEMO_PATH = os.path.join(OUTPUT_DATA_DIR, 'demo_profiles.csv')
        try:
            # Guardar todas las columnas (incluidas las no-memb) para inspección
            df_demo.to_csv(OUTPUT_DEMO_PATH)
            logging.info(f"✅ Archivo de demostración CURADO guardado en: {OUTPUT_DEMO_PATH}")
        except Exception as e:
            logging.warning(f"Advertencia: No se pudo guardar '{OUTPUT_DEMO_PATH}': {e}") # Advertencia, no error crítico
    else:
        logging.warning("No se pudieron seleccionar perfiles para la demo.")

    logging.info("--- 🎉 Pipeline de Procesamiento de Datos Finalizado ---")
