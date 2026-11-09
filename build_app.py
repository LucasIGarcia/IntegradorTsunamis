"""Construye los datos de la aplicación a partir del CSV que produce el pipeline.

Uso:
    python build_app.py                                  # usa la copia congelada datos/DatasetTsunami.csv
    python build_app.py --datos ruta/a/sismos_master.csv # usa una salida nueva del pipeline

Hace lo mismo que el notebook de la Entrega 3, con la misma semilla:
  1. lee los sismos y corta en CORTE (las marcas desde abril de 2026 todavía no están cargadas);
  2. separa entrenamiento y test por grupo (un pliegue de StratifiedGroupKFold, 20 %);
  3. entrena el modelo elegido con los hiperparámetros ajustados;
  4. calcula la probabilidad de cada sismo con un modelo que no lo vio:
     - entrenamiento: validación cruzada (cross_val_predict), igual que la sección 8.2;
     - test y sismos sin marca cargada: el modelo entrenado;
  5. guarda en app_data/ lo que lee app.py: sismos_app.csv, resumen.json y modelo.joblib.

Si los datos cambian, correrlo de nuevo actualiza la aplicación.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix

from modelo import (SEMILLA, CORTE, UMBRAL, CANDIDATAS, HIPERPARAMETROS,
                    modelo_elegido, grupos_de, validacion)


def resultado(real, alerta):
    """Cómo le fue al modelo con cada sismo, en palabras."""
    return np.select([(real == 1) & (alerta == 1), (real == 1) & (alerta == 0), (real == 0) & (alerta == 1)],
                     ["acierto: con marca", "falso negativo", "falsa alarma"], default="acierto: sin marca")


def metricas(y, proba, umbral):
    pred = (proba >= umbral).astype(int)
    return {"f1_macro": f1_score(y, pred, average="macro"),
            "recall_tsunami": pred[y == 1].mean(),
            "precision_tsunami": y[pred == 1].mean(),
            "pr_auc": average_precision_score(y, proba),
            "alertas_cada_100": pred.mean() * 100}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default="datos/DatasetTsunami.csv", help="CSV con los sismos (salida del pipeline)")
    parser.add_argument("--salida", default="app_data", help="carpeta donde se guardan los archivos de la app")
    args = parser.parse_args()
    salida = Path(args.salida)
    salida.mkdir(exist_ok=True)

    # 1. Los datos y el corte
    df = pd.read_csv(args.datos)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    datos = df[df["time"] < CORTE].reset_index(drop=True)
    sin_marca = df[df["time"] >= CORTE].reset_index(drop=True)

    # 2. La partición, por grupo
    X, y = datos[CANDIDATAS], datos["tsunami"]
    grupos = grupos_de(datos)
    cv = validacion()
    idx_train, idx_test = next(cv.split(X, y, grupos))
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train].to_numpy(), y.iloc[idx_test].to_numpy()
    g_train = grupos.iloc[idx_train]

    # 3. El modelo elegido, entrenado con todo el entrenamiento
    modelo = modelo_elegido().fit(X_train, y_train)

    # 4. Probabilidades de modelos que no vieron cada sismo
    proba_cv = cross_val_predict(clone(modelo_elegido()), X_train, y_train, groups=g_train, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
    proba_test = modelo.predict_proba(X_test)[:, 1]
    proba_sin_marca = modelo.predict_proba(sin_marca[CANDIDATAS])[:, 1] if len(sin_marca) else np.array([])

    # 5a. Una tabla con todos los sismos y lo que dijo el modelo de cada uno
    columnas = ["id", "time", "place", "mag", "magType", "depth", "latitude", "longitude", "dist_costa_km", "tsunami"]
    partes = []
    for tabla, proba, conjunto in [(datos.iloc[idx_train], proba_cv, "entrenamiento"),
                                   (datos.iloc[idx_test], proba_test, "test"),
                                   (sin_marca, proba_sin_marca, "sin marca cargada")]:
        t = tabla[columnas].copy()
        t["conjunto"] = conjunto
        t["probabilidad"] = proba.round(4)
        t["alerta"] = (proba >= UMBRAL).astype(int)
        t["resultado"] = (resultado(t["tsunami"].to_numpy(), t["alerta"].to_numpy())
                          if conjunto != "sin marca cargada" else "sin marca cargada")
        partes.append(t)
    app = pd.concat(partes).sort_values("time").reset_index(drop=True)
    app["region"] = app["place"].str.split(", ").str[-1]
    app["en_el_mar"] = np.where(app["dist_costa_km"] > 0, "mar", "tierra")
    app["anio"] = app["time"].dt.year
    app["mes"] = app["time"].dt.strftime("%Y-%m")
    app["time"] = app["time"].dt.strftime("%Y-%m-%d %H:%M")
    app.to_csv(salida / "sismos_app.csv", index=False)

    # 5b. Los números que muestra la app
    conteo_magtype = X_train["magType"].value_counts()
    tn, fp, fn, tp = confusion_matrix(y_test, (proba_test >= UMBRAL).astype(int)).ravel()
    resumen = {
        "umbral": UMBRAL, "semilla": SEMILLA, "corte": CORTE, "hiperparametros": HIPERPARAMETROS,
        "version_sklearn": sklearn.__version__,
        "datos": {"archivo": Path(args.datos).name,
                  "desde": str(df["time"].min().date()), "hasta": str(df["time"].max().date())},
        "filtros": [
            {"paso": "Dataset del pipeline", "filas": len(df), "con_marca": int(df["tsunami"].sum())},
            {"paso": f"Corte antes de {CORTE}", "filas": len(datos), "con_marca": int(y.sum())},
            {"paso": "Entrenamiento (80 %)", "filas": len(X_train), "con_marca": int(y_train.sum())},
            {"paso": "Test (20 %)", "filas": len(X_test), "con_marca": int(y_test.sum())},
        ],
        "validacion_cruzada": metricas(y_train, proba_cv, UMBRAL),
        "validacion_cruzada_umbral_05": metricas(y_train, proba_cv, 0.5),
        "test": {**metricas(y_test, proba_test, UMBRAL),
                 "verdaderos_negativos": int(tn), "falsas_alarmas": int(fp),
                 "falsos_negativos": int(fn), "aciertos_con_marca": int(tp)},
        "piso_test": {"f1_macro": f1_score(y_test, np.zeros_like(y_test), average="macro"),
                      "pr_auc": float(y_test.mean())},
        # Rangos de lo que vio el modelo: "Probá el modelo" no deja cargar valores fuera de esto
        "rangos": {c: [float(X_train[c].min()), float(X_train[c].max())]
                   for c in ["mag", "depth", "latitude", "longitude"]},
        "magtypes": [m for m in conteo_magtype.index if conteo_magtype[m] >= 20],
    }
    with open(salida / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=float)

    # 5c. El modelo, para "Probá el modelo"
    joblib.dump({"pipeline": modelo, "umbral": UMBRAL, "columnas": CANDIDATAS, "semilla": SEMILLA,
                 "datos_hasta": CORTE, "version_sklearn": sklearn.__version__}, salida / "modelo.joblib")

    vc, te = resumen["validacion_cruzada"], resumen["test"]
    print(f"Sismos: {len(df)} | modelados: {len(datos)} | sin marca cargada: {len(sin_marca)}")
    print(f"Validación cruzada (umbral {UMBRAL}): f1 macro {vc['f1_macro']:.3f} · recall {vc['recall_tsunami']:.3f}")
    print(f"Test (umbral {UMBRAL}): f1 macro {te['f1_macro']:.3f} · recall {te['recall_tsunami']:.3f} · "
          f"precision {te['precision_tsunami']:.3f} · FN {fn} · FP {fp}")
    print(f"Archivos guardados en {salida}/")


if __name__ == "__main__":
    main()
