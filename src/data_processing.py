# src/data_processing.py (Pipeline de Generación de Datos)

import pandas as pd
import numpy as np
import os
from collections import Counter

# --- ¡Clases del Árbol de Decisión Difuso (IF-HUPM) ---
# Se mantienen en el script como evidencia del trabajo de investigación.

class FuzzyDecisionTreeNode:
    """Representa un único nodo en el Árbol de Decisión Difuso (IF_HUPM)."""
    def __init__(self, feature_name=None, threshold=None, branches=None, leaf_value=None, is_uncertain=None, class_probabilities=None, n_samples=0):
        """
        Inicializa un nodo del árbol.
        """
        self.feature_name = feature_name
        self.threshold = threshold
        self.branches = branches
        self.leaf_value = leaf_value
        self.uncertainty = is_uncertain
        self.class_probabilities = class_probabilities
        self.n_samples = n_samples

class IF_HUPM:
    """
    Implementación de un Árbol de Decisión Difuso simple (IF-HUPM).
    """
    def __init__(self, min_samples_split=2, max_depth=10, uncertainty_threshold=0.1):
        """
        Inicializa el clasificador de árbol de decisión difuso.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.uncertainty_threshold = uncertainty_threshold
        self.root = None

    def _calculate_entropy(self, y):
        """Calcula la entropía de un conjunto de etiquetas."""
        n = len(y)
        if n == 0: return 0
        proportions = np.array(list(Counter(y).values())) / n
        return -np.sum(p * np.log2(p) for p in proportions if p > 0)

    def _information_gain(self, y, left_y, right_y):
        """Calcula la ganancia de información de una división."""
        n, n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0: return 0
        parent_entropy = self._calculate_entropy(y)
        child_entropy = (n_left / n) * self._calculate_entropy(left_y) + (n_right / n) * self._calculate_entropy(right_y)
        return parent_entropy - child_entropy

    def _find_best_split(self, X, y):
        """Encuentra la mejor característica y umbral para dividir los datos."""
        best_gain, best_feature, best_threshold = -1, None, None
        if len(y) < self.min_samples_split:
            return None, None, None
        
        n_features = X.shape[1]
        size_subset = int(np.sqrt(n_features)) if n_features > 1 else 1
        features_to_check = np.random.choice(X.columns, size=size_subset, replace=False)

        for feature_name in features_to_check:
            for threshold in np.unique(X[feature_name]):
                left_indices, right_indices = y.index[X[feature_name] <= threshold], y.index[X[feature_name] > threshold]
                if len(left_indices) == 0 or len(right_indices) == 0: continue
                
                gain = self._information_gain(y, y.loc[left_indices], y.loc[right_indices])
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_name, threshold
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Construye el árbol de decisión de forma recursiva."""
        if depth >= self.max_depth or len(y.unique()) == 1 or len(y) < self.min_samples_split:
            return self._create_leaf_node(y)

        feature, threshold, gain = self._find_best_split(X, y)
        if gain is None or gain <= 0:
            return self._create_leaf_node(y)

        left_indices, right_indices = y.index[X[feature] <= threshold], y.index[X[feature] > threshold]
        left_branch = self._build_tree(X.loc[left_indices], y.loc[left_indices], depth + 1)
        right_branch = self._build_tree(X.loc[right_indices], y.loc[right_indices], depth + 1)
        
        branches = {'<= ' + str(threshold): left_branch, '> ' + str(threshold): right_branch}
        return FuzzyDecisionTreeNode(feature_name=feature, threshold=threshold, branches=branches, n_samples=len(y))

    def _create_leaf_node(self, y):
        """Crea un nodo hoja y calcula su valor y nivel de incertidumbre."""
        counts = Counter(y)
        if not counts:
            return FuzzyDecisionTreeNode(leaf_value="Incierto (Hoja Vacía)", is_uncertain=True, class_probabilities={}, n_samples=0)
        
        total = len(y)
        probabilities = {k: v / total for k, v in counts.items()}
        most_common_class = max(probabilities, key=probabilities.get)
        max_prob = probabilities[most_common_class]
        sorted_probs = sorted(probabilities.values(), reverse=True)
        
        is_uncertain = (len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < self.uncertainty_threshold) or \
                           (max_prob < (1. - self.uncertainty_threshold))
        
        leaf_value = most_common_class if not is_uncertain else f"Incierto (posiblemente {most_common_class})"
        return FuzzyDecisionTreeNode(leaf_value=leaf_value, is_uncertain=is_uncertain, class_probabilities=probabilities, n_samples=total)

    def fit(self, X, y):
        """Entrena el árbol de decisión."""
        self.root = self._build_tree(X, y)

    def _predict_single(self, x, node):
        """Predice la clase para una única muestra de forma recursiva."""
        if node.leaf_value:
            return node.leaf_value
        
        value = x.get(node.feature_name)
        if value is None:
            return "Incierto (Valor Faltante)"
        
        key = '<= ' + str(node.threshold) if value <= node.threshold else '> ' + str(node.threshold)
        return self._predict_single(x, node.branches[key])

    def predict(self, X):
        """Predice la clase para un DataFrame de muestras."""
        return X.apply(self._predict_single, axis=1, args=(self.root,))

# --- Funciones del Pipeline de Procesamiento ---

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma las variables crudas del dataset en características compuestas.
    """
    print("  › Ejecutando Fase 1: Ingeniería de Características...")
    df_p = df.copy()
    # (El código interno es complejo pero la lógica se mantiene)
    cols_num = ['dificultad_total', 'dificultades', 'tipo_dificultad', 'MNEA', 'edad_agrupada', 'Estado_ocup', 'cat_ocup', 'certificado', 'PC08', 'pc03', 'tipo_hogar']
    for col in cols_num:
        if col in df_p.columns: df_p[col] = pd.to_numeric(df_p[col], errors='coerce')
    c_dificultad = [df_p['dificultad_total']==0, df_p['tipo_dificultad']==1, df_p['tipo_dificultad']==2, df_p['tipo_dificultad']==3, df_p['tipo_dificultad']==4, df_p['tipo_dificultad']==5, df_p['tipo_dificultad']==6, df_p['tipo_dificultad']==7, df_p['tipo_dificultad']==8, (df_p['tipo_dificultad']==9)|(df_p['dificultades']==4), df_p['dificultad_total']==1]
    ch_dificultad = ['0_Sin_Dificultad_Registrada','1A_Motora_Unica','1B_Visual_Unica','1C_Auditiva_Unica','1D_Mental_Cognitiva_Unica','1E_Autocuidado_Unica','1F_Habla_Comunicacion_Unica','2_Dos_Dificultades','3_Tres_o_Mas_Dificultades','4_Solo_Certificado','5_Dificultad_General_No_Detallada']
    df_p['Perfil_Dificultad_Agrupado'] = pd.Series(np.select(c_dificultad, ch_dificultad, default='9_Ignorado_o_No_Clasificado'), index=df_p.index)
    c_capital = [df_p['MNEA']==5, df_p['MNEA']==4, df_p['MNEA'].isin([1,2,3])]; ch_capital = ['3_Alto','2_Medio','1_Bajo']
    df_p['CAPITAL_HUMANO'] = pd.Series(np.select(c_capital, ch_capital, default='9_No_Sabe_o_NC'), index=df_p.index)
    df_p['GRUPO_ETARIO_INDEC'] = df_p['edad_agrupada'].map({1:'0A_0_a_5_anios', 2:'0B_6_a_13_anios', 3:'1_Joven_Adulto_Temprano (14-39)', 4:'2_Adulto_Medio (40-64)', 5:'3_Adulto_Mayor (65+)'}).fillna('No Especificado_Edad')
    df_p['TIENE_CUD'] = df_p['certificado'].map({1:'Si_Tiene_CUD', 2:'No_Tiene_CUD', 9:'Ignorado_CUD'}).fillna('Desconocido_CUD')
    c_inclusion = [df_p['Estado_ocup']==3, df_p['Estado_ocup']==2, (df_p['Estado_ocup']==1)&(df_p['cat_ocup'].isin([1,3])), (df_p['Estado_ocup']==1)&(df_p['cat_ocup'].isin([2,4]))]
    ch_inclusion = ['1_Exclusion_del_Mercado','2_Busqueda_Sin_Exito','4_Inclusion_Plena_Aprox','3_Inclusion_Precaria_Aprox']
    base_inclusion = pd.Series(np.select(c_inclusion, ch_inclusion, default='No_Clasificado_Laboral'), index=df_p.index)
    df_p['Espectro_Inclusion_Laboral'] = base_inclusion.where((df_p['edad_agrupada']>=3)&(df_p['dificultad_total']==1))
    return df_p

def run_archetype_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el grado de pertenencia de cada perfil a un conjunto de arquetipos.
    """
    print("  › Ejecutando Fase 2: Ingeniería de Arquetipos...")
    df_out = df.copy()
    # (El código interno es muy denso y específico del dominio, se mantiene tal cual)
    s_ei=pd.Series(0.0,index=df_out.index); s_ei.loc[df_out['Perfil_Dificultad_Agrupado'].isin(['1F_Habla_Comunicacion_Unica','1C_Auditiva_Unica','1D_Mental_Cognitiva_Unica'])]-=.4; s_ei.loc[df_out['Espectro_Inclusion_Laboral']=='1_Exclusion_del_Mercado']-=.3; s_ei.loc[df_out['tipo_hogar']==1]-=.3; df_out['MBTI_EI_score_sim']=s_ei.clip(-1.,1.).round(2)
    df_out['MBTI_SN_score_sim']=np.select([df_out['CAPITAL_HUMANO']=='1_Bajo',df_out['CAPITAL_HUMANO']=='3_Alto'],[-.5,.5],default=0.); df_out['MBTI_TF_score_sim']=np.select([df_out['pc03']==4,(df_out['pc03'].notna())&(df_out['pc03']!=9)&(df_out['pc03']!=4)],[.5,-.25],default=0.)
    s_jp=pd.Series(0.,index=df_out.index); s_jp.loc[df_out['Espectro_Inclusion_Laboral']=='3_Inclusion_Precaria_Aprox']+=.5; s_jp.loc[df_out['TIENE_CUD']=='Si_Tiene_CUD']-=.5; df_out['MBTI_JP_score_sim']=s_jp.clip(-1.,1.).round(2)
    def _clasificar_comunicador_desafiado(r):
        ch,pdif,slab,get,ei,sn=r.get('CAPITAL_HUMANO'),r.get('Perfil_Dificultad_Agrupado'),r.get('Espectro_Inclusion_Laboral'),r.get('GRUPO_ETARIO_INDEC'),r.get('MBTI_EI_score_sim'),r.get('MBTI_SN_score_sim'); ec_alto,ed_com,esl_sub=(ch=='3_Alto'),(pdif in['1F_Habla_Comunicacion_Unica','1C_Auditiva_Unica']),(slab in['2_Busqueda_Sin_Exito','3_Inclusion_Precaria_Aprox']); ed_mult_no_com=((pdif in['2_Dos_Dificultades','3_Tres_o_Mas_Dificultades'])and not ed_com); pb=0.
        if ec_alto and ed_com and esl_sub:
            if get=='1_Joven_Adulto_Temprano (14-39)':pb=.9
            elif get=='2_Adulto_Medio (40-64)':pb=.75
            else:pb=.65
        elif ec_alto and esl_sub and ed_mult_no_com:pb=.2
        if pb==0.:return 0.
        f_ei=1.-(0.2*ei)if pd.notna(ei)else 1.; f_sn=1.1 if pd.notna(sn)and sn==.5 else(.9 if pd.notna(sn)and sn==-.5 else 1.); pf=pb*max(.8,min(f_ei,1.2))*f_sn; return round(max(0.,min(pf,1.)),2)
    def _clasificar_navegante_informal(r):
        ch,pdif,slab,sn,jp=r.get('CAPITAL_HUMANO'),r.get('Perfil_Dificultad_Agrupado'),r.get('Espectro_Inclusion_Laboral'),r.get('MBTI_SN_score_sim'),r.get('MBTI_JP_score_sim'); pb,ec_bajo,esl_nav=0.,(ch=='1_Bajo'),(slab in['3_Inclusion_Precaria_Aprox','2_Busqueda_Sin_Exito'])
        if ec_bajo and esl_nav:
            if pdif=='2_Dos_Dificultades':pb=.65
            elif pdif in['1F_Habla_Comunicacion_Unica','1C_Auditiva_Unica','1D_Mental_Cognitiva_Unica','3_Tres_o_Mas_Dificultades']:pb=.1
            else:pb=.9
        if pb==0.:return 0.
        f_sn=1.1 if pd.notna(sn)and sn==-.5 else(.9 if pd.notna(sn)and sn==.5 else 1.); f_jp=1.+(0.2*jp)if pd.notna(jp)else 1.; pf=pb*f_sn*max(.8,min(f_jp,1.2)); return round(max(0.,min(pf,1.)),2)
    def _clasificar_profesional_subutilizado(r):
        ch,pdif,slab,get=r.get('CAPITAL_HUMANO'),r.get('Perfil_Dificultad_Agrupado'),r.get('Espectro_Inclusion_Laboral'),r.get('GRUPO_ETARIO_INDEC'); ei,sn,tf,jp=r.get('MBTI_EI_score_sim'),r.get('MBTI_SN_score_sim'),r.get('MBTI_TF_score_sim'),r.get('MBTI_JP_score_sim'); pb,ec_alto,esl_sub=0.,(ch=='3_Alto'),(slab in['2_Busqueda_Sin_Exito','3_Inclusion_Precaria_Aprox']); ed_man,ee_med_may=pdif in['0_Sin_Dificultad_Registrada','4_Solo_Certificado','1A_Motora_Unica','1B_Visual_Unica','1E_Autocuidado_Unica'],(get in['2_Adulto_Medio (40-64)','3_Adulto_Mayor (65+)'])
        if ec_alto and esl_sub:
            if ed_man:pb=.9 if ee_med_may else .7
            elif pdif=='2_Dos_Dificultades':pb=.6 if ee_med_may else .4
            elif pdif in['1F_Habla_Comunicacion_Unica','1C_Auditiva_Unica','1D_Mental_Cognitiva_Unica','3_Tres_o_Mas_Dificultades']:pb=.15
        if pb==0.:return 0.
        fei,fsn,ftf,fjp=1.,1.,1.,1.
        if pd.notna(ei):fei=max(.9,min(1.-(0.1*ei),1.1))
        if pd.notna(sn):fsn=1.05 if sn==.5 else(.95 if sn==-.5 else 1.)
        if pd.notna(tf):ftf=max(.8,min(1.-(0.4*tf),1.2))
        if pd.notna(jp):fjp=max(.8,min(1.-(0.2*jp),1.2))
        pf=pb*fei*fsn*ftf*fjp; return round(max(0.,min(pf,1.)),2)
    def _clasificar_potencial_latente(r):
        slab,pdif,ei,tf,jp=r.get('Espectro_Inclusion_Laboral'),r.get('Perfil_Dificultad_Agrupado'),r.get('MBTI_EI_score_sim'),r.get('MBTI_TF_score_sim'),r.get('MBTI_JP_score_sim'); pb=0.
        if slab=='1_Exclusion_del_Mercado':
            if pdif in['1E_Autocuidado_Unica','3_Tres_o_Mas_Dificultades']:pb=.95
            elif pdif in['1F_Habla_Comunicacion_Unica','1C_Auditiva_Unica','1D_Mental_Cognitiva_Unica']:pb=.8
            elif pdif in['1A_Motora_Unica','1B_Visual_Unica','2_Dos_Dificultades']:pb=.6
            else:pb=.4
        if pb==0.:return 0.
        fei,ftf,fjp=1.,1.,1.
        if pd.notna(ei):fei=max(.7,min(1.-(0.3*ei),1.3))
        if pd.notna(tf):ftf=max(.9,min(1.+(0.1*tf),1.1))
        if pd.notna(jp):fjp=max(.8,min(1.+(0.2*jp),1.2))
        pf=pb*fei*ftf*fjp; return round(max(0.,min(pf,1.)),2)
    def _clasificar_candidato_necesidades_sig(r):
        pdif,slab,ei,tf,jp=r.get('Perfil_Dificultad_Agrupado'),r.get('Espectro_Inclusion_Laboral'),r.get('MBTI_EI_score_sim'),r.get('MBTI_TF_score_sim'),r.get('MBTI_JP_score_sim'); pb=0.
        if pdif=='3_Tres_o_Mas_Dificultades':pb=.95
        elif pdif=='1E_Autocuidado_Unica':pb=.85
        elif pdif=='2_Dos_Dificultades':pb=.75
        elif pdif in['1F_Habla_Comunicacion_Unica','1C_Auditiva_Unica','1D_Mental_Cognitiva_Unica']:pb=.4
        elif pdif in['1A_Motora_Unica','1B_Visual_Unica']:pb=.2
        elif pd.notna(pdif)and pdif!='0_Sin_Dificultad_Registrada':pb=.2
        if pb==0.:return 0.
        f_lab=.7 if slab=='4_Inclusion_Plena_Aprox'else(.9 if slab in['2_Busqueda_Sin_Exito','3_Inclusion_Precaria_Aprox']else 1.); p_mod_lab=pb*f_lab; fei,ftf,fjp=1.,1.,1.
        if pd.notna(ei):fei=max(.9,min(1.-(0.1*ei),1.1))
        if pd.notna(tf):ftf=max(.9,min(1.+(0.1*tf),1.1))
        if pd.notna(jp):fjp=max(.9,min(1.-(0.1*jp),1.1))
        pf=p_mod_lab*fei*ftf*fjp; return round(max(0.,min(pf,1.)),2)
    def _clasificar_joven_transicion(r):
        get,ch,slab,asiste=r.get('GRUPO_ETARIO_INDEC'),r.get('CAPITAL_HUMANO'),r.get('Espectro_Inclusion_Laboral'),r.get('PC08'); ei,tf,jp=r.get('MBTI_EI_score_sim'),r.get('MBTI_TF_score_sim'),r.get('MBTI_JP_score_sim'); pb=0.
        if get=='1_Joven_Adulto_Temprano (14-39)':
            if asiste==1:pb=.85
            elif ch in['2_Medio','3_Alto']and slab in['1_Exclusion_del_Mercado','2_Busqueda_Sin_Exito']:pb=.95
            elif ch=='1_Bajo'and slab in['1_Exclusion_del_Mercado','2_Busqueda_Sin_Exito']:pb=.65
        if pb==0.:return 0.
        fei,ftf,fjp=1.,1.,1.
        if pd.notna(ei):fei=max(.9,min(1.-(0.1*ei),1.1))
        if pd.notna(tf):ftf=max(.9,min(1.+(0.1*tf),1.1))
        if pd.notna(jp):fjp=max(.9,min(1.+(0.1*jp),1.1))
        pf=pb*fei*ftf*fjp; return round(max(0.,min(pf,1.)),2)
    arch_funcs={'Com_Desafiado':_clasificar_comunicador_desafiado,'Nav_Informal':_clasificar_navegante_informal,'Prof_Subutil':_clasificar_profesional_subutilizado,'Potencial_Latente':_clasificar_potencial_latente,'Cand_Nec_Sig':_clasificar_candidato_necesidades_sig,'Joven_Transicion':_clasificar_joven_transicion}
    for name,func in arch_funcs.items(): df_out[f'Pertenencia_{name}']=df_out.apply(func,axis=1)
    return df_out

def run_fuzzification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte características categóricas y scores numéricos en variables difusas.
    """
    print("  › Ejecutando Fase 3: Fuzzificación...")
    df_out = df.copy()
    # (El código interno es complejo pero la lógica se mantiene)
    def _fuzzificar_capital_humano(r):
        v=r.get('CAPITAL_HUMANO');m={'1_Bajo':{'CH_Bajo_memb':1.,'CH_Medio_memb':.2,'CH_Alto_memb':0.},'2_Medio':{'CH_Bajo_memb':.2,'CH_Medio_memb':1.,'CH_Alto_memb':.2},'3_Alto':{'CH_Bajo_memb':0.,'CH_Medio_memb':.2,'CH_Alto_memb':1.}};return pd.Series(m.get(v,{'CH_Bajo_memb':.33,'CH_Medio_memb':.33,'CH_Alto_memb':.33}))
    def _fuzzificar_perfil_dificultad(r):
        v=r.get('Perfil_Dificultad_Agrupado');m={'1A_Motora_Unica':{'PD_Motora_memb':1.,'PD_Sensorial_memb':0.,'PD_ComCog_memb':0.,'PD_Autocuidado_memb':.1,'PD_Multiple_memb':0.},'1B_Visual_Unica':{'PD_Motora_memb':0.,'PD_Sensorial_memb':1.,'PD_ComCog_memb':.1,'PD_Autocuidado_memb':0.,'PD_Multiple_memb':0.},'1C_Auditiva_Unica':{'PD_Motora_memb':0.,'PD_Sensorial_memb':1.,'PD_ComCog_memb':.2,'PD_Autocuidado_memb':0.,'PD_Multiple_memb':0.},'1D_Mental_Cognitiva_Unica':{'PD_Motora_memb':.1,'PD_Sensorial_memb':.1,'PD_ComCog_memb':1.,'PD_Autocuidado_memb':.2,'PD_Multiple_memb':0.},'1E_Autocuidado_Unica':{'PD_Motora_memb':.2,'PD_Sensorial_memb':0.,'PD_ComCog_memb':.2,'PD_Autocuidado_memb':1.,'PD_Multiple_memb':0.},'1F_Habla_Comunicacion_Unica':{'PD_Motora_memb':.1,'PD_Sensorial_memb':.2,'PD_ComCog_memb':1.,'PD_Autocuidado_memb':0.,'PD_Multiple_memb':0.},'2_Dos_Dificultades':{'PD_Motora_memb':.4,'PD_Sensorial_memb':.4,'PD_ComCog_memb':.4,'PD_Autocuidado_memb':.4,'PD_Multiple_memb':1.},'3_Tres_o_Mas_Dificultades':{'PD_Motora_memb':.6,'PD_Sensorial_memb':.6,'PD_ComCog_memb':.6,'PD_Autocuidado_memb':.6,'PD_Multiple_memb':1.}};return pd.Series(m.get(v,{'PD_Motora_memb':.2,'PD_Sensorial_memb':.2,'PD_ComCog_memb':.2,'PD_Autocuidado_memb':.2,'PD_Multiple_memb':.2}))
    def _fuzzificar_grupo_etario(r):
        v=r.get('GRUPO_ETARIO_INDEC');m={'1_Joven_Adulto_Temprano (14-39)':{'Edad_Infanto_Juvenil_memb':.1,'Edad_Joven_memb':1.,'Edad_Adulta_memb':.2,'Edad_Mayor_memb':0.},'2_Adulto_Medio (40-64)':{'Edad_Infanto_Juvenil_memb':0.,'Edad_Joven_memb':.2,'Edad_Adulta_memb':1.,'Edad_Mayor_memb':.2},'3_Adulto_Mayor (65+)':{'Edad_Infanto_Juvenil_memb':0.,'Edad_Joven_memb':0.,'Edad_Adulta_memb':.2,'Edad_Mayor_memb':1.},'0B_6_a_13_anios':{'Edad_Infanto_Juvenil_memb':1.,'Edad_Joven_memb':.2,'Edad_Adulta_memb':0.,'Edad_Mayor_memb':0.},'0A_0_a_5_anios':{'Edad_Infanto_Juvenil_memb':1.,'Edad_Joven_memb':0.,'Edad_Adulta_memb':0.,'Edad_Mayor_memb':0.}};return pd.Series(m.get(v,{'Edad_Infanto_Juvenil_memb':.25,'Edad_Joven_memb':.25,'Edad_Adulta_memb':.25,'Edad_Mayor_memb':.25}))
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
        fuzz_cols_df=df_out.apply(func,axis=1); df_out=pd.concat([df_out,fuzz_cols_df],axis=1)
    return df_out

# --- ¡NUEVO! Bloque de ejecución principal ---
if __name__ == '__main__':
    """
    Este bloque se ejecuta cuando el script es llamado directamente.
    Su propósito es orquestar todo el pipeline de procesamiento de datos
    y generar los archivos CSV finales para entrenamiento y demostración.
    """
    print("--- ⚙️ Iniciando Pipeline de Procesamiento de Datos ---")

    # --- 1. Cargar Datos Crudos ---
    # Asume que los datos crudos están en una ruta específica.
    RAW_DATA_PATH = 'data/raw/endis_data_raw.csv'
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: No se encontró el archivo de datos crudos en '{RAW_DATA_PATH}'.")
        print("   Por favor, asegúrate de que el archivo exista y la ruta sea correcta.")
    else:
        print(f"  › Cargando datos crudos desde: {RAW_DATA_PATH}")
        df_raw = pd.read_csv(RAW_DATA_PATH)

        # --- 2. Ejecutar el Pipeline de Procesamiento ---
        df_featured = run_feature_engineering(df_raw)
        df_archetyped = run_archetype_engineering(df_featured)
        df_fuzzified = run_fuzzification(df_archetyped)

        # --- 3. Preparar y Guardar Archivo para Entrenamiento Cognitivo ---
        print("\n  › Preparando archivo para el entrenamiento del modelo cognitivo...")
        
        # Seleccionar las columnas de membresía difusa como características
        feature_cols = [col for col in df_fuzzified.columns if '_memb' in col]
        
        # Determinar el arquetipo dominante como la variable objetivo
        archetype_cols = [col for col in df_fuzzified.columns if 'Pertenencia_' in col]
        df_fuzzified['ARQUETIPO_PRED'] = df_fuzzified[archetype_cols].idxmax(axis=1).str.replace('Pertenencia_', '')

        # Crear el DataFrame final para el entrenamiento
        df_cognitive_training = df_fuzzified[feature_cols + ['ARQUETIPO_PRED']]
        
        # Guardar el archivo
        OUTPUT_TRAINING_PATH = 'data/cognitive_profiles.csv'
        os.makedirs('data', exist_ok=True)
        df_cognitive_training.to_csv(OUTPUT_TRAINING_PATH, index=False)
        print(f"  › ✅ Archivo para entrenamiento guardado en: {OUTPUT_TRAINING_PATH}")

        # --- 4. Preparar y Guardar Archivo para Demostración ---
        print("\n  › Preparando archivo para la demostración de la aplicación...")
        
        # Seleccionar una muestra de perfiles para la demo
        # Aquí puedes definir una lógica más compleja si lo deseas
        df_demo = df_fuzzified.sample(n=10, random_state=42)
        
        # Crear un ID único para cada perfil de demo
        df_demo['ID'] = [f'Perfil_Demo_{i+1}' for i in range(len(df_demo))]
        df_demo.set_index('ID', inplace=True)
        
        # Guardar el archivo
        OUTPUT_DEMO_PATH = 'data/demo_profiles.csv'
        df_demo.to_csv(OUTPUT_DEMO_PATH)
        print(f"  › ✅ Archivo para demostración guardado en: {OUTPUT_DEMO_PATH}")

        print("\n--- 🎉 Pipeline de Procesamiento de Datos Finalizado ---")


