# Tuto_Doc – Rama `cacic-2026`: reproducción del artículo CACIC 2026 (N° 16613)

**"Un Marco de Trabajo Integral para Interfaces de Tutoría Empáticas y Explicables"**
Caffetti, Y. A.; Acosta, N.; Kuna, H.; Conte, C.

Esta rama parte del commit `9460479` (30/10/2025), versión del código con la que se
obtuvieron los resultados del artículo: https://github.com/YaninaCaffetti/Tuto_Doc/tree/9460479
Solo agrega el script `reproducir_resultados_cacic2026.py`, que reproduce la Tabla 1 y las
Figuras 2 y 3. No se modificó ningún archivo de `src/`.
La descripción general del proyecto (tesis doctoral) se encuentra en la rama `main`.

## Datos
Base usuaria de la ENDIS 2018 (INDEC): `base_estudio_discapacidad_2018.csv`,
disponible en https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos-7
(no se incluye en el repositorio).

## Ejecución (desde la raíz del repositorio)
```bash
git clone -b cacic-2026 https://github.com/YaninaCaffetti/Tuto_Doc.git
cd Tuto_Doc
pip install -r requirements.txt
python reproducir_resultados_cacic2026.py --endis ruta/base_estudio_discapacidad_2018.csv
```

## Resultados esperados
- 82.327 registros; 7.944 personas con dificultad registrada (población del artículo).
- Clasificador de arquetipos (RandomForest, validación cruzada estratificada de 5 pliegues):
  **Macro F1 = 0.618 ± 0.019**.
- `cacic2026/resultados/metricas_por_clase.csv` y `matriz_confusion.csv` → Tabla 1.
- `cacic2026/resultados/fig2_shap_global.png` → Figura 2.
- `cacic2026/resultados/fig3_waterfall_prof_subutil.png` → Figura 3 (E[f(X)] = 0.167 → f(x) = 0.94).

El clasificador de emociones (BETO) se entrena con `python train.py --model emotion`
(requiere GPU y descarga del dataset `emotion` de Hugging Face).
