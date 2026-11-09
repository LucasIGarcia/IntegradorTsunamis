# ¿Qué sismos marca la USGS con alerta de tsunami?

Proyecto Integrador · Ciencia de Datos · UTN FRM · 2026 · Grupo 9 (Comisión 5K09)

**Pregunta:** ¿Qué características de un sismo permiten estimar si está asociado a un tsunami?

**Aplicación:** https://integradortsunamis-grupo09.streamlit.app/

La aplicación cuenta el proyecto entero, con estas secciones:

- **Portada:** la pregunta, la idea central y tres números que la sostienen.
- **Tallo:** tres secciones (los datos, lo que encontramos y el modelo), cada una con su desplegable «Cómo lo sabemos».
- **Copa:** dos secciones para el usuario, «Explorá» (los sismos con lo que dijo el modelo de cada uno) y «Probá el modelo» (cargar un sismo nuevo).

## Cómo encaja cada parte

```
API USGS ──► pipeline/ (Airflow) ──► CSV de sismos ──► build_app.py ──► app_data/ ──► app.py (Streamlit)
                                          │
                                          └──► notebooks/ (EDA y modelado)
```

| Carpeta o archivo | Qué es |
|---|---|
| `pipeline/sismos_tsunami_v2.py` | DAG de Airflow (Astro): extrae los sismos M ≥ 5 de la API de la USGS, los aplana, agrega `dist_costa_km`, valida y consolida en `sismos_master.csv`. Corre todos los días a las 02:00 |
| `notebooks/` | Entrega 2 (análisis exploratorio) y Entrega 3 (modelado) |
| `datos/DatasetTsunami.csv` | Copia congelada de la salida del pipeline, con sismos del 01/01/2019 al 30/07/2026. Es la que usan el notebook de la Entrega 3 y la app |
| `modelo.py` | Las piezas del modelo (el transformador `EsOceanico`, el pipeline, la validación por grupos, la semilla y el umbral). Es el mismo código del notebook de la Entrega 3 |
| `build_app.py` | Toma el CSV, entrena el modelo elegido y calcula lo que muestra la app |
| `app_data/` | Lo que genera `build_app.py`: `sismos_app.csv`, `resumen.json` y `modelo.joblib` |
| `app.py` | La aplicación de Streamlit. Los gráficos están hechos con Altair |

## Cómo se corre

Hace falta Python 3.12 o más nuevo.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (en Linux o Mac: source .venv/bin/activate)
pip install -r requirements.txt

python build_app.py               # genera app_data/ (tarda unos segundos)
streamlit run app.py              # abre la app en http://localhost:8501
```

`build_app.py` imprime los números del modelo. Tienen que coincidir con el notebook de la Entrega 3:

| Medida | Valor |
|---|---|
| f1 macro en validación cruzada | 0,888 |
| f1 macro en test | 0,876 |
| Recall en test | 0,917 |
| Falsos negativos en test | 9 |
| Falsas alarmas en test | 52 |

## Cómo se actualiza si cambian los datos

El pipeline agrega sismos nuevos todos los días a `sismos_master.csv`. Para regenerar la app con esos datos:

```bash
python build_app.py --datos ruta/a/sismos_master.csv
```

Después hay que subir la carpeta `app_data/` al repositorio, y Streamlit Community Cloud vuelve a publicar la app sola.

Los sismos desde el 01/04/2026 no se usan para entrenar, porque la USGS todavía no les cargó la marca de tsunami. Están definidos en `CORTE`, dentro de `modelo.py`. Cuando la USGS cargue esas marcas, se puede mover `CORTE` y volver a correr el notebook y el script.

## Cómo se publica

1. Subir el repositorio a GitHub, con la carpeta `app_data/` incluida.
2. En [share.streamlit.io](https://share.streamlit.io), elegir **Create app**, este repositorio, la rama `main` y el archivo `app.py`.
3. En **Advanced settings**, elegir una versión de Python para la que exista scikit-learn 1.9.1.

El modelo guardado sólo se carga con esa misma versión de scikit-learn.

## El modelo, en dos líneas

Es un gradient boosting (`HistGradientBoostingClassifier`) con magnitud, profundidad, latitud, longitud, tipo de magnitud y mar o tierra. La validación es por grupos de 2° × 2° y mes. Usa umbral 0,27, elegido por validación cruzada para priorizar el recall.

Ojo: la etiqueta `tsunami` no dice si hubo una ola. Es la marca que pone la USGS según su criterio de alerta, que depende de la magnitud y de la región.
