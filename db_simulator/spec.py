"""Carga y validacion de las especificaciones de simulacion.

El usuario describe la base de datos a simular mediante un fichero YAML/JSON
con esta forma general:

    n_rows: 1000
    seed: 42
    variables:
      - name: edad
        type: continuous
        distribution: normal
        mean: 40
        median: 39
        min: 18
        max: 90
      - name: genero
        type: categorical
        categories:
          Hombre: 0.49
          Mujer: 0.49
          Otro: 0.02
    correlation_matrix:
      variables: [edad, ingresos]
      matrix:
        - [1.0, 0.6]
        - [0.6, 1.0]

Una variable continua tambien puede depender de una categorica mediante
`effects` (p.ej. que la media de ingresos suba para un nivel concreto de
genero):

    variables:
      - name: ingresos
        type: continuous
        distribution: lognormal
        mean: 30000
        median: 27000
        min: 5000
        max: 250000
        effects:
          - by: genero
            mean_delta: {Mujer: 5000}
            median_delta: {Mujer: 4000}

Una variable con `effects` no puede aparecer ademas en `correlation_matrix`:
forzar una correlacion reordena las filas de esa variable y deshace el
efecto por categoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONTINUOUS_DISTRIBUTIONS = {"normal", "lognormal", "uniform", "triangular", "exponential", "beta"}


class SpecError(ValueError):
    """Error en la especificacion proporcionada por el usuario."""


@dataclass
class CategoryEffect:
    """Efecto de una variable categorica sobre una continua.

    Para cada nivel de la variable categorica `by`, `mean_delta`/`median_delta`/
    `sd_delta` indican cuanto se desplaza la media/mediana/sd base de la
    variable continua para las filas de ese nivel (los niveles no listados no
    sufren desplazamiento). Varios efectos sobre la misma variable continua
    (p.ej. sexo y region) se combinan sumando sus desplazamientos.
    """

    by: str
    mean_delta: dict[str, float] = field(default_factory=dict)
    median_delta: dict[str, float] = field(default_factory=dict)
    sd_delta: dict[str, float] = field(default_factory=dict)


@dataclass
class ContinuousVariableSpec:
    name: str
    distribution: str
    mean: float
    median: float
    min: float
    max: float
    sd: float | None = None
    effects: list[CategoryEffect] = field(default_factory=list)

    def validate(self) -> None:
        if self.distribution not in CONTINUOUS_DISTRIBUTIONS:
            raise SpecError(
                f"Variable '{self.name}': distribucion '{self.distribution}' no soportada. "
                f"Opciones validas: {sorted(CONTINUOUS_DISTRIBUTIONS)}"
            )
        if self.min > self.max:
            raise SpecError(f"Variable '{self.name}': min ({self.min}) no puede ser mayor que max ({self.max}).")
        if not (self.min <= self.mean <= self.max):
            raise SpecError(
                f"Variable '{self.name}': mean ({self.mean}) debe estar entre min ({self.min}) y max ({self.max})."
            )
        if not (self.min <= self.median <= self.max):
            raise SpecError(
                f"Variable '{self.name}': median ({self.median}) debe estar entre min ({self.min}) y max ({self.max})."
            )
        if self.distribution == "lognormal" and self.min < 0:
            raise SpecError(f"Variable '{self.name}': la distribucion lognormal requiere min >= 0.")
        if self.sd is not None and self.sd <= 0:
            raise SpecError(f"Variable '{self.name}': sd ({self.sd}) debe ser positiva.")


@dataclass
class CategoricalVariableSpec:
    name: str
    categories: dict[str, float]

    def validate(self) -> None:
        if not self.categories:
            raise SpecError(f"Variable '{self.name}': debe definir al menos una categoria.")
        if any(p < 0 for p in self.categories.values()):
            raise SpecError(f"Variable '{self.name}': las proporciones no pueden ser negativas.")
        total = sum(self.categories.values())
        if total <= 0:
            raise SpecError(f"Variable '{self.name}': la suma de proporciones debe ser positiva.")

    def normalized_categories(self) -> dict[str, float]:
        total = sum(self.categories.values())
        return {k: v / total for k, v in self.categories.items()}


@dataclass
class CorrelationMatrixSpec:
    variables: list[str]
    matrix: list[list[float]]

    def validate(self, continuous_names: set[str]) -> None:
        n = len(self.variables)
        if n < 2:
            raise SpecError("correlation_matrix.variables debe incluir al menos dos variables.")
        if len(self.matrix) != n or any(len(row) != n for row in self.matrix):
            raise SpecError(
                f"correlation_matrix.matrix debe ser cuadrada de tamano {n}x{n} "
                f"(una fila/columna por cada variable listada)."
            )
        seen = set()
        for name in self.variables:
            if name in seen:
                raise SpecError(f"correlation_matrix: la variable '{name}' aparece repetida.")
            seen.add(name)
            if name not in continuous_names:
                raise SpecError(
                    f"correlation_matrix: '{name}' no es una variable continua definida. "
                    "Solo se pueden forzar correlaciones entre variables continuas."
                )
        for i in range(n):
            if abs(self.matrix[i][i] - 1.0) > 1e-6:
                raise SpecError(f"correlation_matrix: la diagonal debe valer 1.0 (fila {i}).")
            for j in range(n):
                if not (-1.0 <= self.matrix[i][j] <= 1.0):
                    raise SpecError(
                        f"correlation_matrix: el valor en ({i},{j})={self.matrix[i][j]} debe estar entre -1 y 1."
                    )
                if abs(self.matrix[i][j] - self.matrix[j][i]) > 1e-6:
                    raise SpecError("correlation_matrix: la matriz debe ser simetrica.")


@dataclass
class SimulationSpec:
    n_rows: int
    variables: list[ContinuousVariableSpec | CategoricalVariableSpec]
    seed: int | None = None
    correlation: CorrelationMatrixSpec | None = None

    def validate(self) -> None:
        if self.n_rows <= 0:
            raise SpecError("n_rows debe ser un entero positivo.")
        if not self.variables:
            raise SpecError("Debe definir al menos una variable.")
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise SpecError("Los nombres de variable deben ser unicos.")
        for v in self.variables:
            v.validate()
        continuous_names = {v.name for v in self.variables if isinstance(v, ContinuousVariableSpec)}
        if self.correlation is not None:
            self.correlation.validate(continuous_names)

        categoricals_by_name = {v.name: v for v in self.variables if isinstance(v, CategoricalVariableSpec)}
        for v in self.continuous_variables():
            for effect in v.effects:
                if effect.by == v.name:
                    raise SpecError(f"Variable '{v.name}': un efecto no puede depender de si misma.")
                if effect.by not in categoricals_by_name:
                    raise SpecError(
                        f"Variable '{v.name}': el efecto hace referencia a '{effect.by}', que no es "
                        "una variable categorica definida."
                    )
                valid_levels = set(categoricals_by_name[effect.by].categories)
                for delta_dict, field_name in (
                    (effect.mean_delta, "mean_delta"),
                    (effect.median_delta, "median_delta"),
                    (effect.sd_delta, "sd_delta"),
                ):
                    unknown = set(delta_dict) - valid_levels
                    if unknown:
                        raise SpecError(
                            f"Variable '{v.name}': efecto por '{effect.by}' ({field_name}) referencia "
                            f"niveles inexistentes: {sorted(unknown)}. Niveles validos: {sorted(valid_levels)}."
                        )
            if v.effects and self.correlation is not None and v.name in self.correlation.variables:
                raise SpecError(
                    f"Variable '{v.name}': no puede tener 'effects' y participar a la vez en "
                    "correlation_matrix. Forzar una correlacion reordena las filas y deshace el efecto "
                    "por categoria. Usa una de las dos cosas para esta variable, no ambas."
                )

    def continuous_variables(self) -> list[ContinuousVariableSpec]:
        return [v for v in self.variables if isinstance(v, ContinuousVariableSpec)]

    def categorical_variables(self) -> list[CategoricalVariableSpec]:
        return [v for v in self.variables if isinstance(v, CategoricalVariableSpec)]


def _parse_variable(raw: dict[str, Any]) -> ContinuousVariableSpec | CategoricalVariableSpec:
    if "name" not in raw or "type" not in raw:
        raise SpecError(f"Cada variable debe tener 'name' y 'type': {raw}")
    name = raw["name"]
    var_type = raw["type"]
    if var_type == "continuous":
        required = ["distribution", "mean", "median", "min", "max"]
        missing = [k for k in required if k not in raw]
        if missing:
            raise SpecError(f"Variable '{name}': faltan campos {missing} para tipo continuous.")
        sd = raw.get("sd")
        effects = [
            CategoryEffect(
                by=e["by"],
                mean_delta={str(k): float(v) for k, v in e.get("mean_delta", {}).items()},
                median_delta={str(k): float(v) for k, v in e.get("median_delta", {}).items()},
                sd_delta={str(k): float(v) for k, v in e.get("sd_delta", {}).items()},
            )
            for e in raw.get("effects", [])
        ]
        return ContinuousVariableSpec(
            name=name,
            distribution=raw["distribution"],
            mean=float(raw["mean"]),
            median=float(raw["median"]),
            min=float(raw["min"]),
            max=float(raw["max"]),
            sd=float(sd) if sd not in (None, "") else None,
            effects=effects,
        )
    if var_type == "categorical":
        if "categories" not in raw:
            raise SpecError(f"Variable '{name}': falta 'categories' para tipo categorical.")
        categories = {str(k): float(v) for k, v in raw["categories"].items()}
        return CategoricalVariableSpec(name=name, categories=categories)
    raise SpecError(f"Variable '{name}': type '{var_type}' no soportado (usa 'continuous' o 'categorical').")


def spec_from_dict(raw: dict[str, Any]) -> SimulationSpec:
    if "variables" not in raw:
        raise SpecError("La especificacion debe incluir la clave 'variables'.")
    variables = [_parse_variable(v) for v in raw["variables"]]

    correlation = None
    corr_raw = raw.get("correlation_matrix")
    if corr_raw is not None:
        correlation = CorrelationMatrixSpec(
            variables=list(corr_raw["variables"]),
            matrix=[list(row) for row in corr_raw["matrix"]],
        )

    spec = SimulationSpec(
        n_rows=int(raw.get("n_rows", 1000)),
        variables=variables,
        seed=raw.get("seed"),
        correlation=correlation,
    )
    spec.validate()
    return spec


def load_spec(path: str | Path) -> SimulationSpec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    return spec_from_dict(raw)
