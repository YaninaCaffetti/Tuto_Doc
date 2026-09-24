"""
Reproducción de los resultados del artículo CACIC 2026 (N° 16613):
"Un Marco de Trabajo Integral para Interfaces de Tutoría Empáticas y Explicables".

Usa el pipeline de este commit (octubre 2025) sin modificaciones y restringe el
entrenamiento y la evaluación a las personas con dificultad registrada
(dificultad_total == 1), dado que la ENDIS 2018 releva a todos los integrantes
del hogar y las reglas de arquetipos solo aplican a personas con dificultad.

Salidas (carpeta cacic2026/resultados/):
  - metricas_por_clase.csv, matriz_confusion.csv  (Tabla 1)
  - fig2_shap_global.png                           (Figura 2)
  - fig3_waterfall_prof_subutil.png                (Figura 3)

Uso (desde la raíz del repositorio):
  python -m cacic2026.reproducir_resultados_cacic2026 --endis ruta/base_estudio_discapacidad_2018.csv
Resultado esperado: Macro F1 = 0.618 ± 0.019 sobre 7.944 personas.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from src.profile_inference import run_feature_engineering, run_fuzzification
from src.data_processing import run_archetype_engineering

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endis", required=True, help="CSV de la ENDIS 2018 (INDEC), separador ';'")
    ap.add_argument("--out", default=os.path.join("cacic2026", "resultados"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    raw = pd.read_csv(a.endis, sep=";", encoding="latin1", low_memory=False)
    df = run_fuzzification(run_archetype_engineering(run_feature_engineering(raw)))
    memb = [c for c in df.columns if c.endswith("_memb")]
    df[memb] = df[memb].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    perten = [c for c in df.columns if c.startswith("Pertenencia_")]
    df["ARQUETIPO"] = df[perten].idxmax(axis=1).str.replace("Pertenencia_", "")

    # Población del artículo: personas con dificultad registrada
    d = df[pd.to_numeric(raw["dificultad_total"], errors="coerce") == 1].reset_index(drop=True)
    X, y = d[memb], d["ARQUETIPO"]
    print(f"Registros ENDIS: {len(raw)} | personas con dificultad: {len(d)}")
    print(y.value_counts().to_string())

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                random_state=42, class_weight="balanced", n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")
    print(f"Macro F1 (5-fold CV): {s.mean():.3f} ± {s.std():.3f}")

    labs = ["Potencial_Latente", "Cand_Nec_Sig", "Joven_Transicion", "Nav_Informal", "Prof_Subutil", "Com_Desafiado"]
    p = cross_val_predict(rf, X, y, cv=cv)
    pd.DataFrame(classification_report(y, p, labels=labs, digits=3, output_dict=True)).T.round(3)\
      .to_csv(os.path.join(a.out, "metricas_por_clase.csv"))
    pd.DataFrame(confusion_matrix(y, p, labels=labs), index=labs, columns=labs)\
      .to_csv(os.path.join(a.out, "matriz_confusion.csv"))

    import shap
    rf.fit(X, y); cls = list(rf.classes_)
    sv = shap.TreeExplainer(rf)(X)                      # (n, variables, clases)
    imp = pd.DataFrame(np.abs(sv.values).mean(0), index=X.columns, columns=cls)
    top = imp.assign(t=imp.sum(1)).sort_values("t", ascending=False).drop(columns="t").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2)); left = np.zeros(len(top))
    for k, c in enumerate(cls):
        ax.barh(top.index, top[c], left=left, label=c, color=plt.cm.tab10.colors[k]); left += top[c].values
    ax.set_xlabel("Media de |valor SHAP| (impacto sobre la probabilidad de cada arquetipo)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig2_shap_global.png"), dpi=300); plt.close(fig)

    k = cls.index("Prof_Subutil"); pr = rf.predict_proba(X)
    cand = np.where((y.values == "Prof_Subutil") & (pr.argmax(1) == k))[0]
    i = cand[np.argmax(pr[cand, k])]
    e = shap.Explanation(values=sv.values[i, :, k], base_values=sv.base_values[i, k],
                         data=X.iloc[i].values, feature_names=list(X.columns))
    shap.plots.waterfall(e, max_display=8, show=False)
    f = plt.gcf(); f.set_size_inches(9, 4.6); f.subplots_adjust(left=0.36, right=0.93, top=0.88, bottom=0.14)
    f.savefig(os.path.join(a.out, "fig3_waterfall_prof_subutil.png"), dpi=300); plt.close(f)
    print(f"Waterfall: E[f(X)] = {sv.base_values[i, k]:.3f} -> f(x) = {pr[i, k]:.2f}")

if __name__ == "__main__":
    main()
