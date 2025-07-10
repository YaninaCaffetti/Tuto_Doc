# src/data_processing.py

import pandas as pd
import numpy as np
from collections import Counter

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta la Fase 1: Ingeniería de Características a partir del dataset crudo."""
    print("  › Ejecutando Fase 1: Ingeniería de Características...")
    df_p = df.copy()
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
    print("  › Ejecutando Fase 2: Ingeniería de Arquetipos...")
    df_out = df.copy()
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
    print("  › Ejecutando Fase 3: Fuzzificación...")
    df_out = df.copy()
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
