# Reproducción – CACIC 2026 (artículo N° 16613)

**"Un Marco de Trabajo Integral para Interfaces de Tutoría Empáticas y Explicables"**
Caffetti, Y. A.; Acosta, N.; Kuna, H.; Conte, C.

El artículo remite al commit `9460479` (30/10/2025), versión del código con la que se
obtuvieron los resultados: https://github.com/YaninaCaffetti/Tuto_Doc/tree/9460479
Esta carpeta complementa ese código con el script que reproduce la Tabla 1 y las
Figuras 2 y 3. Debe ejecutarse sobre esa versión; no modifica ningún archivo de `src/`.

## Datos
Base usuaria de la ENDIS 2018 (INDEC): `base_estudio_discapacidad_2018.csv`,
disponible en https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos-7
(no se incluye en el repositorio).

## Ejecución
```bash
pip install -r cacic2026/requirements.txt
python -m cacic2026.reproducir_resultados_cacic2026 --endis ruta/base_estudio_discapacidad_2018.csv
```

## Resultados esperados
- 82.327 registros; 7.944 personas con dificultad registrada (población del artículo).
- Clasificador de arquetipos (RandomForest, validación cruzada estratificada de 5 pliegues):
  **Macro F1 = 0.618 ± 0.019**.
- `resultados/metricas_por_clase.csv` y `resultados/matriz_confusion.csv` → Tabla 1.
- `resultados/fig2_shap_global.png` → Figura 2.
- `resultados/fig3_waterfall_prof_subutil.png` → Figura 3 (E[f(X)] = 0.167 → f(x) = 0.94).

El clasificador de emociones (BETO) se entrena con `python train.py --model emotion`
(requiere GPU y descarga del dataset `emotion` de Hugging Face).
