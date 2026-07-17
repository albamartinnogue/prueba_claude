"""Orquestador principal: junta variables, correlaciones y genera el DataFrame."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .categorical import sample_categorical
from .correlation import induce_correlation
from .distributions import build_sampler
from .spec import CategoricalVariableSpec, ContinuousVariableSpec, SimulationSpec, load_spec, spec_from_dict


def _sample_continuous_with_effects(
    var: ContinuousVariableSpec, categorical_columns: dict[str, np.ndarray], n: int, rng: np.random.Generator
) -> np.ndarray:
    """Genera una variable continua cuyos parametros dependen de una o mas categoricas.

    Para cada fila se suman los desplazamientos (delta) de todos los efectos
    aplicables segun el nivel de la fila en cada variable categorica, y se
    agrupan las filas con el mismo desplazamiento total para muestrear cada
    grupo con una distribucion ajustada (media/mediana/sd base + delta).
    """
    mean_deltas = np.zeros(n)
    median_deltas = np.zeros(n)
    sd_deltas = np.zeros(n)
    for effect in var.effects:
        col = categorical_columns[effect.by]
        for level, delta in effect.mean_delta.items():
            mean_deltas[col == level] += delta
        for level, delta in effect.median_delta.items():
            median_deltas[col == level] += delta
        for level, delta in effect.sd_delta.items():
            sd_deltas[col == level] += delta

    combos = np.stack([mean_deltas, median_deltas, sd_deltas], axis=1)
    unique_combos, inverse = np.unique(combos, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)

    result = np.empty(n)
    for combo_idx, (mean_delta, median_delta, sd_delta) in enumerate(unique_combos):
        mask = inverse == combo_idx
        sub_spec = replace(
            var,
            mean=float(np.clip(var.mean + mean_delta, var.min, var.max)),
            median=float(np.clip(var.median + median_delta, var.min, var.max)),
            sd=(max(var.sd + sd_delta, 1e-6) if var.sd is not None else None),
            effects=[],
        )
        result[mask] = build_sampler(sub_spec)(rng, int(mask.sum()))
    return result


@dataclass
class ValidationReport:
    continuous: dict[str, dict[str, float]]
    categorical: dict[str, dict[str, float]]
    correlation: dict[str, "pd.DataFrame"] | None
    effects: dict[str, list[dict]]

    def to_text(self) -> str:
        lines = ["=== Informe de validacion ==="]
        if self.continuous:
            lines.append("\nVariables continuas (objetivo -> obtenido):")
            for name, stats in self.continuous.items():
                sd_part = ""
                if stats.get("target_sd") is not None:
                    sd_part = f" | sd {stats['target_sd']:.3g} -> {stats['achieved_sd']:.3g}"
                lines.append(
                    f"  {name}: mean {stats['target_mean']:.3g} -> {stats['achieved_mean']:.3g} | "
                    f"median {stats['target_median']:.3g} -> {stats['achieved_median']:.3g} | "
                    f"min {stats['target_min']:.3g} -> {stats['achieved_min']:.3g} | "
                    f"max {stats['target_max']:.3g} -> {stats['achieved_max']:.3g}" + sd_part
                )
        if self.categorical:
            lines.append("\nVariables categoricas (objetivo -> obtenido):")
            for name, props in self.categorical.items():
                parts = [f"{cat}: {target:.3f}->{achieved:.3f}" for cat, (target, achieved) in props.items()]
                lines.append(f"  {name}: " + ", ".join(parts))
        if self.correlation is not None:
            lines.append("\nMatriz de correlacion objetivo:")
            lines.append(self.correlation["target"].to_string(float_format=lambda x: f"{x:.2f}"))
            lines.append("\nMatriz de correlacion obtenida:")
            lines.append(self.correlation["achieved"].to_string(float_format=lambda x: f"{x:.2f}"))
        if self.effects:
            lines.append("\nEfectos categorica -> continua (media por nivel):")
            for name, rows in self.effects.items():
                for row in rows:
                    lines.append(
                        f"  {name} | {row['by']}={row['level']}: media ajustada {row['media_base_ajustada']:.3g} "
                        f"-> obtenida {row['media_obtenida']:.3g} (n={row['n']})"
                    )
        return "\n".join(lines)


class Simulator:
    def __init__(self, spec: SimulationSpec):
        self.spec = spec

    @classmethod
    def from_file(cls, path: str | Path) -> "Simulator":
        return cls(load_spec(path))

    @classmethod
    def from_dict(cls, raw: dict) -> "Simulator":
        return cls(spec_from_dict(raw))

    def generate(self, n_rows: int | None = None, seed: int | None = None) -> pd.DataFrame:
        df, _ = self._generate_with_raw(n_rows, seed)
        return df

    def generate_with_report(self, n_rows: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, ValidationReport]:
        df, continuous_data = self._generate_with_raw(n_rows, seed)
        report = self._build_report(df, continuous_data)
        return df, report

    def _generate_with_raw(self, n_rows: int | None, seed: int | None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        spec = self.spec
        n = n_rows or spec.n_rows
        rng = np.random.default_rng(seed if seed is not None else spec.seed)

        # Las categoricas se generan primero: las continuas con `effects` las
        # necesitan para saber a que subgrupo pertenece cada fila.
        categorical_columns: dict[str, np.ndarray] = {
            var.name: sample_categorical(var, n, rng) for var in spec.categorical_variables()
        }

        continuous_specs = {v.name: v for v in spec.continuous_variables()}
        raw_samples: dict[str, np.ndarray] = {}
        for name, cvar in continuous_specs.items():
            if cvar.effects:
                raw_samples[name] = _sample_continuous_with_effects(cvar, categorical_columns, n, rng)
            else:
                raw_samples[name] = build_sampler(cvar)(rng, n)

        if spec.correlation is not None:
            corr_vars = spec.correlation.variables
            matrix = np.array(spec.correlation.matrix, dtype=float)
            stacked = np.column_stack([raw_samples[name] for name in corr_vars])
            correlated = induce_correlation(stacked, matrix, rng)
            for i, name in enumerate(corr_vars):
                raw_samples[name] = correlated[:, i]

        columns: dict[str, np.ndarray] = {}
        for var in spec.variables:
            if isinstance(var, ContinuousVariableSpec):
                columns[var.name] = raw_samples[var.name]
            elif isinstance(var, CategoricalVariableSpec):
                columns[var.name] = categorical_columns[var.name]

        df = pd.DataFrame(columns)
        return df, raw_samples

    def _build_report(self, df: pd.DataFrame, continuous_data: dict[str, np.ndarray]) -> ValidationReport:
        spec = self.spec
        continuous_stats: dict[str, dict[str, float]] = {}
        for var in spec.continuous_variables():
            values = df[var.name].to_numpy()
            continuous_stats[var.name] = {
                "target_mean": var.mean,
                "achieved_mean": float(np.mean(values)),
                "target_median": var.median,
                "achieved_median": float(np.median(values)),
                "target_min": var.min,
                "achieved_min": float(np.min(values)),
                "target_max": var.max,
                "achieved_max": float(np.max(values)),
                "target_sd": var.sd,
                "achieved_sd": float(np.std(values)),
            }

        categorical_stats: dict[str, dict[str, tuple[float, float]]] = {}
        for var in spec.categorical_variables():
            target = var.normalized_categories()
            counts = df[var.name].value_counts(normalize=True)
            categorical_stats[var.name] = {
                cat: (target_p, float(counts.get(cat, 0.0))) for cat, target_p in target.items()
            }

        correlation_report = None
        if spec.correlation is not None:
            names = spec.correlation.variables
            target_df = pd.DataFrame(spec.correlation.matrix, index=names, columns=names)
            achieved_df = df[names].corr()
            correlation_report = {"target": target_df, "achieved": achieved_df}

        effects_stats: dict[str, list[dict]] = {}
        for var in spec.continuous_variables():
            if not var.effects:
                continue
            rows = []
            for effect in var.effects:
                col = df[effect.by]
                for level in sorted(col.unique()):
                    mask = (col == level).to_numpy()
                    delta = effect.mean_delta.get(level, 0.0)
                    rows.append(
                        {
                            "by": effect.by,
                            "level": level,
                            "media_base_ajustada": var.mean + delta,
                            "media_obtenida": float(df.loc[mask, var.name].mean()) if mask.any() else float("nan"),
                            "n": int(mask.sum()),
                        }
                    )
            if rows:
                effects_stats[var.name] = rows

        return ValidationReport(
            continuous=continuous_stats,
            categorical=categorical_stats,
            correlation=correlation_report,
            effects=effects_stats,
        )
