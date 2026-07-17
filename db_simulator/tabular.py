"""Conversion entre tablas planas (DataFrame) y las piezas de una especificacion.

Estas funciones son el puente entre un formato tabular (una hoja Excel, o
cualquier tabla editable) y los diccionarios que espera `spec_from_dict`.
No dependen de Streamlit ni de ficheros: solo trabajan con DataFrames de
pandas, para poder probarlas y reutilizarlas facilmente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import SpecError

VARIABLE_COLUMNS = ["nombre", "tipo", "distribucion", "media", "mediana", "min", "max", "sd", "categorias"]
EFFECT_COLUMNS = ["variable_continua", "variable_categorica", "nivel", "delta_media", "delta_mediana"]


def row_value(row: pd.Series, key: str):
    """Devuelve el valor de una celda, o None si esta vacia/():NaN."""
    if key not in row:
        return None
    value = row[key]
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def parse_categories_text(name: str, text) -> dict[str, float]:
    """Convierte 'Nivel:proporcion;Nivel:proporcion' en un diccionario."""
    categories: dict[str, float] = {}
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SpecError(
                f"Variable '{name}': formato de categorias invalido en '{part}'. Usa 'Nivel:proporcion;Nivel:proporcion'."
            )
        level, prop = part.rsplit(":", 1)
        level = level.strip()
        try:
            categories[level] = float(prop.strip())
        except ValueError as exc:
            raise SpecError(f"Variable '{name}': proporcion invalida para '{level}' en '{part}'.") from exc
    return categories


def variables_from_table(df: pd.DataFrame) -> list[dict]:
    """Convierte la hoja/tabla de variables (una fila por variable) en specs crudas."""
    variables = []
    for _, row in df.iterrows():
        name = str(row_value(row, "nombre") or "").strip()
        if not name:
            continue
        var_type = str(row_value(row, "tipo") or "continuous").strip()
        if var_type == "continuous":
            missing = [f for f in ("media", "mediana", "min", "max") if row_value(row, f) is None]
            if missing:
                raise SpecError(f"Variable '{name}': faltan valores en {missing}.")
            var = {
                "name": name,
                "type": "continuous",
                "distribution": row_value(row, "distribucion") or "normal",
                "mean": row_value(row, "media"),
                "median": row_value(row, "mediana"),
                "min": row_value(row, "min"),
                "max": row_value(row, "max"),
            }
            sd = row_value(row, "sd")
            if sd is not None:
                var["sd"] = sd
            variables.append(var)
        elif var_type == "categorical":
            variables.append(
                {
                    "name": name,
                    "type": "categorical",
                    "categories": parse_categories_text(name, row_value(row, "categorias")),
                }
            )
        else:
            raise SpecError(f"Variable '{name}': tipo '{var_type}' no reconocido (usa 'continuous' o 'categorical').")
    return variables


def get_categorical_levels(df: pd.DataFrame, cat_name: str) -> list[str]:
    matches = df[df["nombre"].astype(str).str.strip() == cat_name]
    if matches.empty:
        return []
    try:
        return list(parse_categories_text(cat_name, row_value(matches.iloc[0], "categorias")).keys())
    except SpecError:
        return []


def effects_from_table(df: pd.DataFrame | None) -> dict[str, list[dict]]:
    """Convierte la hoja/tabla 'efectos' (formato largo) en `effects` por variable continua.

    Columnas esperadas: variable_continua, variable_categorica, nivel,
    delta_media, delta_mediana. Varias filas con la misma (variable_continua,
    variable_categorica) forman los niveles de un mismo efecto.
    """
    if df is None or df.empty:
        return {}

    grouped: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for _, row in df.iterrows():
        cont = str(row_value(row, "variable_continua") or "").strip()
        cat = str(row_value(row, "variable_categorica") or "").strip()
        level = str(row_value(row, "nivel") or "").strip()
        if not cont or not cat or not level:
            continue
        mean_delta = row_value(row, "delta_media")
        median_delta = row_value(row, "delta_mediana")
        entry = grouped.setdefault(cont, {}).setdefault(cat, {"mean_delta": {}, "median_delta": {}})
        if mean_delta is not None:
            entry["mean_delta"][level] = float(mean_delta)
        if median_delta is not None:
            entry["median_delta"][level] = float(median_delta)

    result: dict[str, list[dict]] = {}
    for cont, by_map in grouped.items():
        result[cont] = [{"by": cat, "mean_delta": deltas["mean_delta"], "median_delta": deltas["median_delta"]} for cat, deltas in by_map.items()]
    return result


def correlation_from_table(df: pd.DataFrame | None) -> dict | None:
    """Convierte la hoja/tabla 'correlaciones' (matriz con nombres en filas y columnas) en un dict."""
    if df is None or df.empty:
        return None
    names = [str(c).strip() for c in df.columns]
    df = df.copy()
    df.columns = names
    df.index = [str(i).strip() for i in df.index]
    try:
        df = df.loc[names, names]
    except KeyError as exc:
        raise SpecError(
            "correlaciones: los nombres de las filas y las columnas no coinciden. "
            "Usa exactamente los mismos nombres de variable en ambos."
        ) from exc
    return {"variables": names, "matrix": df.to_numpy(dtype=float).tolist()}
