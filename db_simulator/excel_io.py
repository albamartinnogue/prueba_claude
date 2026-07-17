"""Lectura de la especificacion desde un Excel, y generacion de la plantilla.

El Excel debe tener una hoja llamada 'variables' (una fila por variable) y,
opcionalmente, una hoja 'correlaciones' (matriz de correlaciones entre
variables continuas) y una hoja 'efectos' (efecto de una categorica sobre
una continua). Ver `build_template_bytes` para el formato exacto de cada
hoja, o generar la plantilla directamente con esa funcion.
"""

from __future__ import annotations

import io
from typing import Any, BinaryIO

import pandas as pd

from .spec import CONTINUOUS_DISTRIBUTIONS, SpecError
from .tabular import (
    EFFECT_COLUMNS,
    VARIABLE_COLUMNS,
    correlation_from_table,
    effects_from_table,
    variables_from_table,
)

VARIABLES_SHEET = "variables"
CORRELATIONS_SHEET = "correlaciones"
EFFECTS_SHEET = "efectos"
INSTRUCTIONS_SHEET = "instrucciones"


def read_workbook(file: BinaryIO | str) -> dict[str, Any]:
    """Lee un Excel y devuelve un dict crudo listo para `spec_from_dict` (sin n_rows/seed)."""
    try:
        sheets = pd.read_excel(file, sheet_name=None, engine="openpyxl")
    except Exception as exc:
        raise SpecError(f"No se pudo leer el fichero Excel: {exc}") from exc

    sheet_lookup = {name.strip().lower(): name for name in sheets}
    if VARIABLES_SHEET not in sheet_lookup:
        raise SpecError(f"El Excel debe incluir una hoja llamada '{VARIABLES_SHEET}'.")

    variables = variables_from_table(sheets[sheet_lookup[VARIABLES_SHEET]])

    correlation = None
    if CORRELATIONS_SHEET in sheet_lookup:
        corr_df = sheets[sheet_lookup[CORRELATIONS_SHEET]]
        if not corr_df.empty:
            corr_df = corr_df.set_index(corr_df.columns[0])
            correlation = correlation_from_table(corr_df)

    effects_map = {}
    if EFFECTS_SHEET in sheet_lookup:
        effects_map = effects_from_table(sheets[sheet_lookup[EFFECTS_SHEET]])

    for var in variables:
        if var["type"] == "continuous" and var["name"] in effects_map:
            var["effects"] = effects_map[var["name"]]

    raw: dict[str, Any] = {"variables": variables}
    if correlation is not None:
        raw["correlation_matrix"] = correlation
    return raw


def build_template_bytes() -> bytes:
    """Genera un Excel de plantilla (con ejemplos) listo para rellenar y subir."""
    variables_example = pd.DataFrame(
        [
            {
                "nombre": "edad",
                "tipo": "continuous",
                "distribucion": "normal",
                "media": 40,
                "mediana": 38,
                "min": 18,
                "max": 85,
                "sd": None,
                "categorias": None,
            },
            {
                "nombre": "ingresos",
                "tipo": "continuous",
                "distribucion": "lognormal",
                "media": 30000,
                "mediana": 27000,
                "min": 5000,
                "max": 250000,
                "sd": None,
                "categorias": None,
            },
            {
                "nombre": "satisfaccion",
                "tipo": "continuous",
                "distribucion": "triangular",
                "media": 7,
                "mediana": 7.5,
                "min": 0,
                "max": 10,
                "sd": None,
                "categorias": None,
            },
            {
                "nombre": "sexo",
                "tipo": "categorical",
                "distribucion": None,
                "media": None,
                "mediana": None,
                "min": None,
                "max": None,
                "sd": None,
                "categorias": "Hombre:50;Mujer:50",
            },
        ],
        columns=VARIABLE_COLUMNS,
    )

    # Nota: 'ingresos' no participa aqui porque ya tiene un efecto en la hoja
    # 'efectos' (una variable no puede tener ambas cosas a la vez).
    correlaciones_example = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]], index=["edad", "satisfaccion"], columns=["edad", "satisfaccion"]
    )

    efectos_example = pd.DataFrame(
        [{"variable_continua": "ingresos", "variable_categorica": "sexo", "nivel": "Mujer", "delta_media": 5000, "delta_mediana": 4000}],
        columns=EFFECT_COLUMNS,
    )

    instrucciones = pd.DataFrame(
        {
            "Instrucciones": [
                "Rellena la hoja 'variables': una fila por variable.",
                "Columnas 'distribucion','media','mediana','min','max','sd' solo aplican a tipo=continuous.",
                f"Distribuciones validas: {', '.join(sorted(CONTINUOUS_DISTRIBUTIONS))}.",
                "'sd' (desviacion tipica objetivo) es opcional; dejala vacia si no quieres fijarla.",
                "Columna 'categorias' solo aplica a tipo=categorical. Formato: Nivel:proporcion;Nivel:proporcion",
                "  Ejemplo: Hombre:48;Mujer:50;Otro:2  (no hace falta que sumen 100, se normalizan solas).",
                "",
                "Hoja 'correlaciones' (opcional): matriz de correlaciones entre variables CONTINUAS.",
                "La primera columna y la primera fila deben tener los mismos nombres de variable, en el",
                "mismo orden. La diagonal debe ser 1 y la matriz debe ser simetrica.",
                "",
                "Hoja 'efectos' (opcional): efecto de una categorica sobre la media/mediana de una continua.",
                "Una fila por (variable_continua, variable_categorica, nivel). Deja delta_media/delta_mediana",
                "vacios si ese nivel no tiene efecto.",
                "",
                "Una variable con efectos en la hoja 'efectos' NO puede aparecer tambien en 'correlaciones':",
                "forzar una correlacion reordena filas y deshace el efecto por categoria.",
                "",
                "Borra las filas de ejemplo y sustituyelas por tus variables antes de subir el fichero.",
            ]
        }
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        instrucciones.to_excel(writer, sheet_name=INSTRUCTIONS_SHEET, index=False)
        variables_example.to_excel(writer, sheet_name=VARIABLES_SHEET, index=False)
        correlaciones_example.to_excel(writer, sheet_name=CORRELATIONS_SHEET, index=True)
        efectos_example.to_excel(writer, sheet_name=EFFECTS_SHEET, index=False)
        _add_data_validations(writer.sheets[VARIABLES_SHEET])
        _autosize_columns(writer.sheets[INSTRUCTIONS_SHEET], width=100)
        _autosize_columns(writer.sheets[VARIABLES_SHEET])
        _autosize_columns(writer.sheets[EFFECTS_SHEET])

    return buffer.getvalue()


def _add_data_validations(worksheet) -> None:
    from openpyxl.worksheet.datavalidation import DataValidation

    max_row = 1000

    tipo_dv = DataValidation(type="list", formula1='"continuous,categorical"', allow_blank=True)
    worksheet.add_data_validation(tipo_dv)
    tipo_dv.add(f"B2:B{max_row}")

    dist_options = ",".join(sorted(CONTINUOUS_DISTRIBUTIONS))
    dist_dv = DataValidation(type="list", formula1=f'"{dist_options}"', allow_blank=True)
    worksheet.add_data_validation(dist_dv)
    dist_dv.add(f"C2:C{max_row}")


def _autosize_columns(worksheet, width: int = 20) -> None:
    for column_cells in worksheet.columns:
        letter = column_cells[0].column_letter
        worksheet.column_dimensions[letter].width = width
