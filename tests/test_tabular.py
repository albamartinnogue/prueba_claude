import numpy as np
import pandas as pd
import pytest

from db_simulator.spec import SpecError
from db_simulator.tabular import (
    correlation_from_table,
    effects_from_table,
    get_categorical_levels,
    parse_categories_text,
    row_value,
    variables_from_table,
)


def test_row_value_treats_nan_and_blank_as_none():
    row = pd.Series({"a": np.nan, "b": "  ", "c": "x", "d": 5.0})
    assert row_value(row, "a") is None
    assert row_value(row, "b") is None
    assert row_value(row, "c") == "x"
    assert row_value(row, "d") == 5.0
    assert row_value(row, "missing") is None


def test_parse_categories_text_valid():
    assert parse_categories_text("v", "Hombre:48;Mujer:50;Otro:2") == {"Hombre": 48.0, "Mujer": 50.0, "Otro": 2.0}


def test_parse_categories_text_rejects_bad_format():
    with pytest.raises(SpecError):
        parse_categories_text("v", "Hombre-48;Mujer-50")


def variables_df(rows):
    columns = ["nombre", "tipo", "distribucion", "media", "mediana", "min", "max", "sd", "categorias"]
    return pd.DataFrame(rows, columns=columns)


def test_variables_from_table_skips_blank_rows():
    df = variables_df(
        [
            {"nombre": "edad", "tipo": "continuous", "distribucion": "normal", "media": 40, "mediana": 38, "min": 18, "max": 85, "sd": np.nan, "categorias": ""},
            {"nombre": "", "tipo": None, "distribucion": None, "media": np.nan, "mediana": np.nan, "min": np.nan, "max": np.nan, "sd": np.nan, "categorias": ""},
        ]
    )
    variables = variables_from_table(df)
    assert len(variables) == 1
    assert variables[0]["name"] == "edad"
    assert "sd" not in variables[0]


def test_variables_from_table_includes_sd_when_present():
    df = variables_df(
        [{"nombre": "edad", "tipo": "continuous", "distribucion": "normal", "media": 40, "mediana": 38, "min": 18, "max": 85, "sd": 6.0, "categorias": ""}]
    )
    variables = variables_from_table(df)
    assert variables[0]["sd"] == 6.0


def test_variables_from_table_raises_on_missing_continuous_fields():
    df = variables_df(
        [{"nombre": "edad", "tipo": "continuous", "distribucion": "normal", "media": 40, "mediana": np.nan, "min": 18, "max": 85, "sd": np.nan, "categorias": ""}]
    )
    with pytest.raises(SpecError):
        variables_from_table(df)


def test_variables_from_table_parses_categorical():
    df = variables_df(
        [{"nombre": "sexo", "tipo": "categorical", "distribucion": None, "media": np.nan, "mediana": np.nan, "min": np.nan, "max": np.nan, "sd": np.nan, "categorias": "Hombre:50;Mujer:50"}]
    )
    variables = variables_from_table(df)
    assert variables[0]["categories"] == {"Hombre": 50.0, "Mujer": 50.0}


def test_variables_from_table_rejects_unknown_type():
    df = variables_df(
        [{"nombre": "x", "tipo": "weird", "distribucion": None, "media": np.nan, "mediana": np.nan, "min": np.nan, "max": np.nan, "sd": np.nan, "categorias": ""}]
    )
    with pytest.raises(SpecError):
        variables_from_table(df)


def test_get_categorical_levels():
    df = variables_df(
        [{"nombre": "sexo", "tipo": "categorical", "distribucion": None, "media": np.nan, "mediana": np.nan, "min": np.nan, "max": np.nan, "sd": np.nan, "categorias": "Hombre:50;Mujer:50"}]
    )
    assert get_categorical_levels(df, "sexo") == ["Hombre", "Mujer"]
    assert get_categorical_levels(df, "no_existe") == []


def test_effects_from_table_groups_by_variable_pair():
    df = pd.DataFrame(
        [
            {"variable_continua": "ingresos", "variable_categorica": "sexo", "nivel": "Mujer", "delta_media": 5000, "delta_mediana": 4000},
            {"variable_continua": "ingresos", "variable_categorica": "sexo", "nivel": "Hombre", "delta_media": np.nan, "delta_mediana": np.nan},
        ]
    )
    effects = effects_from_table(df)
    assert effects["ingresos"] == [{"by": "sexo", "mean_delta": {"Mujer": 5000.0}, "median_delta": {"Mujer": 4000.0}}]


def test_effects_from_table_empty():
    assert effects_from_table(None) == {}
    assert effects_from_table(pd.DataFrame()) == {}


def test_correlation_from_table_valid():
    df = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]], index=["edad", "ingresos"], columns=["edad", "ingresos"])
    result = correlation_from_table(df)
    assert result == {"variables": ["edad", "ingresos"], "matrix": [[1.0, 0.6], [0.6, 1.0]]}


def test_correlation_from_table_rejects_mismatched_names():
    df = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]], index=["edad", "otra_cosa"], columns=["edad", "ingresos"])
    with pytest.raises(SpecError):
        correlation_from_table(df)


def test_correlation_from_table_none_when_empty():
    assert correlation_from_table(None) is None
    assert correlation_from_table(pd.DataFrame()) is None
