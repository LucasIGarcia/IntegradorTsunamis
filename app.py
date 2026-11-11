"""¿Qué sismos marca la USGS con alerta de tsunami? — Proyecto Integrador, Grupo 9 (UTN FRM, 2026).

Aplicación de Streamlit con forma de copa de martini:
  portada → tallo (los datos, lo que encontramos, el modelo) → copa (explorá, probá el modelo).

Lee lo que genera build_app.py en app_data/. Para correrla:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import sklearn
import streamlit as st

import modelo  # define EsOceanico: joblib la necesita para cargar el pipeline

BASE = Path(__file__).parent
APP_DATA = BASE / "app_data"
MUNDO = alt.topo_feature("https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/world-110m.json", "countries")
alt.data_transformers.disable_max_rows()   # el mapa tiene ~13.600 sismos

# --- Colores: significan lo mismo en toda la aplicación -----------------------------------------
AZUL, GRIS_CLARO, GRIS, NARANJA, VIOLETA = "#2a78d6", "#cfcac3", "#7a7974", "#d95926", "#8a5cc2"
RESULTADOS = ["acierto: con marca", "falso negativo", "falsa alarma", "acierto: sin marca", "sin marca cargada"]
COLORES_RESULTADO = [AZUL, NARANJA, VIOLETA, GRIS_CLARO, GRIS]
ESCALA_RESULTADO = alt.Scale(domain=RESULTADOS, range=COLORES_RESULTADO)

# Regiones donde la USGS usa un umbral regional (sección 2.1 de la Entrega 3)
REGION_ALASKA_OESTE = ["Alaska", "Oregon", "off the coast of Oregon", "Canada", "CA", "California", "OR",
                       "south of Alaska"]


# --- Formato en castellano ----------------------------------------------------------------------
def pct(x, dec=0):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def num(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


# --- Carga ---------------------------------------------------------------------------------------
@st.cache_data
def cargar_sismos():
    return pd.read_csv(APP_DATA / "sismos_app.csv")


@st.cache_data
def cargar_resumen():
    with open(APP_DATA / "resumen.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo():
    return joblib.load(APP_DATA / "modelo.joblib")


st.set_page_config(page_title="Sismos y marca de tsunami · Grupo 9", layout="wide")

if not (APP_DATA / "sismos_app.csv").exists():
    st.error("Faltan los datos de la aplicación: correr primero `python build_app.py`.")
    st.stop()

sismos = cargar_sismos()
resumen = cargar_resumen()
guardado = cargar_modelo()
UMBRAL = guardado["umbral"]
modelados = sismos[sismos["conjunto"] != "sin marca cargada"]
test = resumen["test"]
vc = resumen["validacion_cruzada"]

if guardado.get("version_sklearn") != sklearn.__version__:
    st.warning(f"El modelo se guardó con scikit-learn {guardado.get('version_sklearn')} y acá corre "
               f"{sklearn.__version__}. Instalar la versión de requirements.txt.")


# =================================================================================================
# PORTADA
# =================================================================================================
st.caption("Ciencia de Datos · UTN FRM · 2026 · Proyecto Integrador · Grupo 9 · Comisión 5K09")
st.title("¿Qué características de un sismo permiten estimar si está asociado a un tsunami?")

# TODO (grupo): la idea central en UNA oración con sujeto y verbo. Borrador:
st.markdown(f"#### Con la magnitud y el lugar del sismo se anticipa el {pct(test['recall_tsunami'])} "
            "de los eventos que la USGS marca con alerta de tsunami, y el lugar pesa tanto como la magnitud.")

chicos = modelados[(modelados["tsunami"] == 1) & (modelados["mag"] < 6)]
c1, c2, c3 = st.columns(3)
c1.metric("sismos con marca detectados en el test",
          f"{test['aciertos_con_marca']} de {test['aciertos_con_marca'] + test['falsos_negativos']}",
          help=f"Recall de la clase tsunami con umbral {num(UMBRAL, 2)}")
c2.metric("f1 macro en el test", num(test["f1_macro"]),
          f"{num(test['f1_macro'] - resumen['piso_test']['f1_macro'])} sobre el piso", delta_color="off")
c3.metric("marcas con magnitud menor a 6 que están en Alaska y la costa oeste",
          f"{chicos['region'].isin(REGION_ALASKA_OESTE).sum()} de {len(chicos)}")
st.caption(f"Medido sobre {miles(len(modelados))} sismos de magnitud 5 o más registrados por la USGS entre "
           f"{resumen['datos']['desde']} y el 31/03/2026. Ojo: `tsunami = 1` no dice si hubo una ola, "
           "sino que la USGS marcó el evento con su criterio de alerta.")

st.divider()


# =================================================================================================
# TALLO 1 · LOS DATOS
# =================================================================================================
st.header("Los datos")
# TODO (grupo): el título tiene que AFIRMAR algo de los datos. Borrador:
st.subheader(f"Sólo {round(modelados['tsunami'].mean() * 100)} de cada 100 sismos llevan la marca de tsunami, "
             "y desde abril de 2026 la marca todavía no se cargó")

mensual = (sismos.groupby("mes", as_index=False)
           .agg(sismos=("id", "count"), con_marca=("tsunami", "sum")))
mensual["proporcion"] = mensual["con_marca"] / mensual["sismos"]
mensual["periodo"] = mensual["mes"].map(lambda m: "sin marca cargada" if m >= modelo.CORTE[:7] else "marca consolidada")
promedio = modelados["tsunami"].mean()

barras = alt.Chart(mensual).mark_bar().encode(
    x=alt.X("mes:O", title=None, axis=alt.Axis(values=[m for m in mensual["mes"] if m.endswith("-01")], labelAngle=0,
                                                 labelExpr="slice(datum.value, 0, 4)")),
    y=alt.Y("proporcion:Q", title="sismos con marca de tsunami", axis=alt.Axis(format="%")),
    color=alt.Color("periodo:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["marca consolidada", "sin marca cargada"], range=[AZUL, GRIS])),
    tooltip=[alt.Tooltip("mes:O", title="mes"), alt.Tooltip("sismos:Q", title="sismos"),
             alt.Tooltip("con_marca:Q", title="con marca"), alt.Tooltip("proporcion:Q", title="proporción", format=".1%")])
linea = alt.Chart(pd.DataFrame({"y": [promedio]})).mark_rule(color=GRIS, strokeDash=[4, 4]).encode(y="y:Q")
st.altair_chart((barras + linea).properties(width=900, height=260), theme=None)
st.markdown(f"La línea punteada es el promedio 2019–marzo 2026 ({pct(promedio, 1)}). Desde abril de 2026 la "
            "proporción cae a casi cero: no dejó de haber tsunamis, la USGS todavía no revisó esos eventos. "
            "Por eso esos meses quedan afuera del modelo.")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**De dónde salen.** De la API pública FDSN de la USGS, con un pipeline de Airflow (arquitectura Medallón:
Bronce con el JSON crudo, Plata con el CSV aplanado y Oro con la columna propia `dist_costa_km`).
El pipeline corre todos los días a las 02:00 y consolida los sismos por `id`. Esta aplicación usa una copia congelada
(`{resumen['datos']['archivo']}`), con sismos del {resumen['datos']['desde']} al {resumen['datos']['hasta']}.

**Qué es una fila.** Un sismo de magnitud 5 o más. Los sismos no son independientes (un sismo grande trae réplicas),
así que la validación y el test se arman por grupo: celda de 2° × 2° y mes.

**Qué se filtró y qué quedó.**
""")
    st.dataframe(pd.DataFrame(resumen["filtros"]).rename(
        columns={"paso": "Paso", "filas": "Sismos", "con_marca": "Con marca"}), hide_index=True)
    st.markdown("""
**Qué columnas se descartaron.**
- `sig`, por **fuga**: la USGS la calcula usando la alerta de tsunami.
- `id`, `place` y `time`: identificadores (`time` sólo arma los grupos). `status`, `type` y `net`: constantes.
- `nst`, `dmin`, `gap` y `rms`: no mejoran el modelo (ablación) y la USGS las recalcula al revisar el evento.
""")

st.divider()


# =================================================================================================
# TALLO 2 · LO QUE ENCONTRAMOS
# =================================================================================================
st.header("Lo que encontramos")
# TODO (grupo): título que afirma el hallazgo principal. Borrador:
st.subheader("La marca depende de la magnitud y del lugar: cerca de Alaska se marcan sismos que en el resto del mundo no")

mapa = modelados.assign(grupo=lambda d: d["tsunami"].map({0: "sin marca", 1: "con marca"}).where(
    ~((d["tsunami"] == 1) & (d["mag"] < 6)), "con marca, magnitud menor a 6"))
fondo = alt.Chart(MUNDO).mark_geoshape(fill="#eeeeee", stroke="white", strokeWidth=0.5)
puntos = alt.Chart(mapa).mark_circle().encode(
    longitude="longitude:Q", latitude="latitude:Q",
    color=alt.Color("grupo:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["sin marca", "con marca", "con marca, magnitud menor a 6"],
                                    range=[GRIS_CLARO, "#9cc3ef", AZUL])),
    size=alt.Size("grupo:N", legend=None,
                  scale=alt.Scale(domain=["sin marca", "con marca", "con marca, magnitud menor a 6"], range=[4, 18, 30])),
    order=alt.Order("tsunami:Q"),
    tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("mag:Q", title="magnitud")])
st.altair_chart(alt.layer(fondo, puntos).project(type="equirectangular").properties(width=900, height=450), theme=None)
st.markdown(f"Los puntos azul oscuro son sismos **con marca y magnitud menor a 6**: {len(chicos)} en total, y casi "
            "todos están en Alaska y la costa oeste de Norteamérica, donde la USGS opera un sistema de alerta regional. "
            "Los de magnitud 6 o más (azul claro) están en todo el Cinturón de Fuego del Pacífico.")

with st.expander("Cómo lo sabemos"):
    # TODO (grupo): revisar redacción y números con el notebook de la Entrega 2 corregida
    st.markdown("""
**Una hipótesis que se sostuvo: la magnitud separa los sismos con marca de los sin marca.**
Separación 1,38 (zona verde del semáforo de la cátedra). Pero no es una regla absoluta: en el mar, el 83 % de los
sismos de más de 7,5 tiene la marca, no el 100 %.

**Una que no: la profundidad y la distancia a la costa.** La profundidad separa 0,03 y la distancia a la costa 0,17
(zona roja; 0,42 si se miran sólo los sismos en el mar). En el modelo tampoco aportan: sacarlas no cambia el f1.

**Cómo se construyó el gráfico.** Cada punto es un sismo modelado (hasta marzo de 2026), en su latitud y longitud,
sobre un mapa equirectangular. El color separa sin marca, con marca y con marca de magnitud menor a 6.
""")

st.divider()


# =================================================================================================
# TALLO 3 · EL MODELO
# =================================================================================================
st.header("El modelo")
# TODO (grupo): título que afirma qué logró el modelo. Borrador:
st.subheader(f"El modelo detecta {test['aciertos_con_marca']} de cada "
             f"{test['aciertos_con_marca'] + test['falsos_negativos']} sismos marcados, "
             f"a cambio de una falsa alarma cada {round((test['verdaderos_negativos'] + test['falsas_alarmas']) / test['falsas_alarmas'])} sismos sin marca")

conf = (sismos[sismos["conjunto"] == "test"]
        .assign(real=lambda d: d["tsunami"].map({1: "con marca", 0: "sin marca"}))
        .groupby(["real", "resultado"], as_index=False).size())
conf["parte"] = conf["size"] / conf.groupby("real")["size"].transform("sum")
barra_conf = alt.Chart(conf).mark_bar().encode(
    y=alt.Y("real:N", title="en el test, sismos…", sort=["con marca", "sin marca"]),
    x=alt.X("parte:Q", title=None, axis=alt.Axis(format="%"), stack="normalize"),
    color=alt.Color("resultado:N", title="qué dijo el modelo", scale=ESCALA_RESULTADO, legend=alt.Legend(orient="top")),
    order=alt.Order("resultado:N", sort="descending"),
    tooltip=[alt.Tooltip("resultado:N"), alt.Tooltip("size:Q", title="sismos"),
             alt.Tooltip("parte:Q", title="parte", format=".1%")])
st.altair_chart(barra_conf.properties(width=900, height=160), theme=None)
st.markdown(f"Con umbral {num(UMBRAL, 2)}, de los {test['aciertos_con_marca'] + test['falsos_negativos']} sismos con "
            f"marca del test el modelo detecta {test['aciertos_con_marca']} ({pct(test['recall_tsunami'])}) y pierde "
            f"{test['falsos_negativos']}: ese es el error caro. A cambio, da {test['falsas_alarmas']} falsas alarmas "
            f"entre {miles(test['verdaderos_negativos'] + test['falsas_alarmas'])} sismos sin marca.")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**Métrica: f1 macro**, declarada antes de comparar modelos. Promedia el f1 de las dos clases, así que la marca
(4 % de los sismos) pesa lo mismo que el resto. La accuracy no sirve: decir siempre «sin marca» acierta el 96 %.

**El piso:** f1 macro {num(resumen['piso_test']['f1_macro'])} (siempre «sin marca»).

**Validación:** `StratifiedGroupKFold` de 5 pedazos, con grupos de 2° × 2° y mes para que las réplicas de un sismo
no queden de los dos lados. El test (20 %) se separó con el mismo criterio y se miró una sola vez.

**La comparación** (f1 macro en validación cruzada, media ± desvío; notebook de la Entrega 3). Las cuatro familias
se probaron primero con sus valores por defecto; la elegida se ajustó después:

| Modelo | f1 macro |
|---|---|
| Piso (siempre «sin marca») | 0,489 ± 0,000 |
| Regresión logística | 0,747 ± 0,025 |
| Árbol de decisión | 0,842 ± 0,014 |
| Random forest | 0,872 ± 0,016 |
| Gradient boosting | 0,867 ± 0,016 |
| **Gradient boosting ajustado (el que usa la app)** | **0,881 ± 0,011** |

**¿Por qué el boosting, si el random forest tiene un f1 más alto?** Porque sin ajustar empatan: la diferencia es de
0,005, tres veces menor que el desvío (0,016). Sus rangos típicos (0,856 a 0,888 y 0,851 a 0,883) se superponen casi
por completo, y con otra partición de los pliegues el orden podría invertirse. En PR-AUC también empatan (0,784 contra
0,783). Ante el empate se decidió por lo práctico: el boosting entrena unas tres veces más rápido (`build_app.py` lo
entrena seis veces cada vez que se actualizan los datos) y da un modelo más liviano para la app (100 árboles de 15
hojas, contra 300 árboles sin límite de profundidad). Una vez ajustado llega a 0,881 y supera a todos los de la tabla.

La logística pierde porque no puede dibujar la «caja» geográfica de Alaska.

**El ajuste:** `GridSearchCV` (36 combinaciones). Ganó tasa 0,05, 100 árboles, 15 hojas y sin `class_weight`:
0,881 ± 0,011.

**El umbral:** el f1 empata entre 0,2 y 0,5, así que se eligió el umbral con más recall que mantiene el f1
máximo: **{num(UMBRAL, 2)}**. En validación cruzada sube el recall de {pct(resumen['validacion_cruzada_umbral_05']['recall_tsunami'], 1)}
a {pct(vc['recall_tsunami'], 1)} y baja la precisión de {pct(resumen['validacion_cruzada_umbral_05']['precision_tsunami'], 1)}
a {pct(vc['precision_tsunami'], 1)}.

**El test, una vez:** f1 macro {num(test['f1_macro'])} (validación cruzada: {num(vc['f1_macro'])}),
recall {pct(test['recall_tsunami'])}, precisión {pct(test['precision_tsunami'])}, PR-AUC {num(test['pr_auc'])}.

**Dónde se equivoca:** los falsos negativos son sismos de 6,3 a 6,5 fuera de Alaska (en tierra o profundos) y sismos
chicos en el mar frente a California y Alaska. Las falsas alarmas más seguras son sismos fuertes en el mar que la
USGS no marcó.

**En qué se apoya** (importancia por permutación, f1 macro): longitud 0,17, magnitud 0,16, latitud 0,15,
tipo de magnitud 0,10; profundidad y mar/tierra, nada. La ubicación no es fuga: refleja la regla regional de la USGS.
""")
    # TODO (grupo): corregir en el notebook la nota de 10.2 (la permutación usa predict = umbral 0,5) y actualizar acá

st.divider()


# =================================================================================================
# COPA · EXPLORÁ
# =================================================================================================
st.header("Tu turno · Explorá")
st.markdown("Todos los sismos, con lo que dijo el modelo de cada uno. Cada sismo lo predice un modelo que no lo vio: "
            "en el entrenamiento, la predicción sale de la validación cruzada. Achicá el universo con los controles, "
            "hacé clic en la leyenda para resaltar un resultado y pasá el mouse por un sismo para ver el detalle.")

f1, f2, f3 = st.columns(3)
rango_mag = f1.slider("Magnitud", 5.0, float(sismos["mag"].max()), (5.0, float(sismos["mag"].max())), step=0.1)
que_mostrar = f2.multiselect("Qué dijo el modelo", RESULTADOS, default=RESULTADOS)
regiones = ["todas"] + sorted(sismos["region"].dropna().unique())
region = f3.selectbox("Región", regiones)

vista = sismos[sismos["mag"].between(*rango_mag) & sismos["resultado"].isin(que_mostrar)]
if region != "todas":
    vista = vista[vista["region"] == region]
vista = vista.assign(marca_real=vista["tsunami"].map({1: "sí", 0: "no"}).where(vista["conjunto"] != "sin marca cargada",
                                                                               "todavía no cargada"))

resaltar = alt.selection_point(fields=["resultado"], bind="legend")
puntos_copa = alt.Chart(vista).mark_circle().encode(
    longitude="longitude:Q", latitude="latitude:Q",
    color=alt.Color("resultado:N", title="qué dijo el modelo", scale=ESCALA_RESULTADO, legend=alt.Legend(orient="top")),
    size=alt.Size("mag:Q", title="magnitud", scale=alt.Scale(domain=[5, 9], range=[6, 260])),
    opacity=alt.condition(resaltar, alt.value(0.85), alt.value(0.05)),
    order=alt.Order("probabilidad:Q"),
    tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("time:N", title="fecha (UTC)"),
             alt.Tooltip("mag:Q", title="magnitud"), alt.Tooltip("magType:N", title="tipo de magnitud"),
             alt.Tooltip("depth:Q", title="profundidad (km)"), alt.Tooltip("en_el_mar:N", title="mar o tierra"),
             alt.Tooltip("marca_real:N", title="marca de la USGS"),
             alt.Tooltip("probabilidad:Q", title="probabilidad según el modelo", format=".0%"),
             alt.Tooltip("resultado:N", title="resultado"), alt.Tooltip("conjunto:N", title="conjunto")],
).add_params(resaltar)
st.altair_chart(alt.layer(fondo, puntos_copa).project(type="equirectangular").properties(width=900, height=450),
                theme=None)
st.caption(f"{miles(len(vista))} sismos en la vista. Alerta = probabilidad de {num(UMBRAL, 2)} o más.")

st.divider()


# =================================================================================================
# COPA · PROBÁ EL MODELO
# =================================================================================================
st.header("Tu turno · Probá el modelo")
st.markdown("Armá un sismo y mirá qué probabilidad le da el modelo de que la USGS lo marque. Los controles van del "
            "mínimo al máximo que vio el entrenamiento, así que no se puede cargar un valor que el modelo no conoce.")

r = resumen["rangos"]
p1, p2 = st.columns(2)
mag = p1.slider("Magnitud", r["mag"][0], r["mag"][1], 7.0, step=0.1, format="%.1f")
depth = p1.slider("Profundidad (km)", r["depth"][0], r["depth"][1], 25.0, step=1.0)
mag_type = p1.selectbox("Tipo de magnitud", resumen["magtypes"],
                        help="Sólo los tipos con al menos 20 sismos en el entrenamiento")
lat = p2.slider("Latitud", r["latitude"][0], r["latitude"][1], -33.5, step=0.1, format="%.1f")
lon = p2.slider("Longitud", r["longitude"][0], r["longitude"][1], -72.0, step=0.1, format="%.1f")
en_el_mar = p2.radio("¿El epicentro está en el mar o en tierra?", ["mar", "tierra"], horizontal=True) == "mar"

probabilidad = guardado["pipeline"].predict_proba(modelo.caso(mag, depth, lat, lon, mag_type, en_el_mar))[0, 1]
m1, m2 = st.columns(2)
m1.metric("probabilidad de que la USGS lo marque", pct(probabilidad))
m2.metric(f"alerta (umbral {num(UMBRAL, 2)})", "sí" if probabilidad >= UMBRAL else "no")

tu_sismo = alt.Chart(pd.DataFrame({"latitude": [lat], "longitude": [lon]})).mark_point(
    shape="cross", size=400, strokeWidth=3, color="#222222").encode(longitude="longitude:Q", latitude="latitude:Q")
st.altair_chart(alt.layer(fondo, tu_sismo).project(type="equirectangular").properties(width=900, height=360), theme=None)
# TODO (grupo): ¿sumar un menú «partir de un sismo real», como el ejemplo de la cátedra?

st.divider()
st.caption(f"Datos: USGS (earthquake.usgs.gov), sismos del {resumen['datos']['desde']} al {resumen['datos']['hasta']}. "
           f"Modelo: gradient boosting entrenado con datos hasta el 31/03/2026, semilla {resumen['semilla']}.")
