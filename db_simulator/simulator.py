"""Orquestador principal: junta variables, correlaciones y genera el DataFrame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .categorical import sample_categorical
from .correlation import induce_correlation
from .distributions import build_sampler
from .spec import CategoricalVariableSpec, ContinuousVariableSpec, SimulationSpec, load_spec, spec_from_dict


@dataclass
class ValidationReport:
    continuous: dict[str, dict[str, float]]
    categorical: dict[str, dict[str, float]]
    correlation: dict[str, "pd.DataFrame"] | None

    def to_text(self) -> str:
        lines = ["=== Informe de validacion ==="]
        if self.continuous:
            lines.append("\nVariables continuas (objetivo -> obtenido):")
            for name, stats in self.continuous.items():
                lines.append(
                    f"  {name}: mean {stats['target_mean']:.3g} -> {stats['achieved_mean']:.3g} | "
                    f"median {stats['target_median']:.3g} -> {stats['achieved_median']:.3g} | "
                    f"min {stats['target_min']:.3g} -> {stats['achieved_min']:.3g} | "
                    f"max {stats['target_max']:.3g} -> {stats['achieved_max']:.3g}"
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

        continuous_specs = {v.name: v for v in spec.continuous_variables()}
        raw_samples: dict[str, np.ndarray] = {
            name: build_sampler(cvar)(rng, n) for name, cvar in continuous_specs.items()
        }

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
                columns[var.name] = sample_categorical(var, n, rng)

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

        return ValidationReport(continuous=continuous_stats, categorical=categorical_stats, correlation=correlation_report)
