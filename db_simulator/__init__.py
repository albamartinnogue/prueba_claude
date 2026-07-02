"""Simulador de bases de datos tabulares a partir de especificaciones de usuario."""

from .simulator import Simulator, ValidationReport
from .spec import SimulationSpec, SpecError, load_spec, spec_from_dict

__all__ = [
    "Simulator",
    "ValidationReport",
    "SimulationSpec",
    "SpecError",
    "load_spec",
    "spec_from_dict",
]
