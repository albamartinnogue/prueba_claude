import io

import pandas as pd
import pytest

from db_simulator.excel_io import build_template_bytes, read_workbook
from db_simulator.simulator import Simulator
from db_simulator.spec import SpecError, spec_from_dict


def test_build_template_bytes_produces_valid_xlsx():
    data = build_template_bytes()
    assert isinstance(data, bytes)
    assert len(data) > 0
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
    assert {"instrucciones", "variables", "correlaciones", "efectos"} <= set(sheets)


def test_read_workbook_template_produces_valid_spec():
    raw = read_workbook(io.BytesIO(build_template_bytes()))
    raw["n_rows"] = 200
    spec = spec_from_dict(raw)
    assert len(spec.variables) == 4  # edad, ingresos, satisfaccion, sexo
    assert spec.correlation is not None


def test_read_workbook_template_generates_data():
    raw = read_workbook(io.BytesIO(build_template_bytes()))
    raw["n_rows"] = 500
    raw["seed"] = 1
    df = Simulator.from_dict(raw).generate()
    assert len(df) == 500
    assert set(df.columns) == {"edad", "ingresos", "satisfaccion", "sexo"}


def test_read_workbook_requires_variables_sheet():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="otra_hoja", index=False)
    with pytest.raises(SpecError):
        read_workbook(io.BytesIO(buffer.getvalue()))


def test_read_workbook_without_correlations_or_effects_sheets():
    variables = pd.DataFrame(
        [{"nombre": "x", "tipo": "continuous", "distribucion": "normal", "media": 5, "mediana": 5, "min": 0, "max": 10, "sd": None, "categorias": None}]
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        variables.to_excel(writer, sheet_name="variables", index=False)
    raw = read_workbook(io.BytesIO(buffer.getvalue()))
    assert "correlation_matrix" not in raw
    assert raw["variables"][0]["name"] == "x"


def test_read_workbook_is_case_insensitive_to_sheet_names():
    variables = pd.DataFrame(
        [{"nombre": "x", "tipo": "continuous", "distribucion": "normal", "media": 5, "mediana": 5, "min": 0, "max": 10, "sd": None, "categorias": None}]
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        variables.to_excel(writer, sheet_name="Variables", index=False)
    raw = read_workbook(io.BytesIO(buffer.getvalue()))
    assert raw["variables"][0]["name"] == "x"
