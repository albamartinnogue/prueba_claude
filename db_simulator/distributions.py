"""Ajuste y muestreo de variables continuas.

Para cada variable continua se elige una familia de distribucion y se
ajustan sus parametros para que la media y la mediana (truncadas al rango
[min, max] indicado por el usuario) se aproximen lo mas posible a los
valores objetivo. Despues se generan muestras respetando siempre el rango
[min, max] de forma exacta.

No se busca una coincidencia perfecta con los estadisticos objetivo (el
propio enunciado del problema lo permite), sino una aproximacion razonable
obtenida mediante ajuste numerico u formulas cerradas segun la familia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize
from scipy import stats

from .spec import ContinuousVariableSpec, SpecError

Sampler = Callable[[np.random.Generator, int], np.ndarray]


def _truncated_sample(dist, params: dict, low: float, high: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Muestrea por inversion de la CDF truncada al intervalo [low, high]."""
    a_cdf = dist.cdf(low, **params)
    b_cdf = dist.cdf(high, **params)
    if b_cdf <= a_cdf:
        return np.full(size, (low + high) / 2.0)
    u = rng.uniform(a_cdf, b_cdf, size=size)
    return np.clip(dist.ppf(u, **params), low, high)


def _truncated_mean(dist, params: dict, low: float, high: float) -> float:
    mean = dist.expect(lambda x: x, args=(), loc=params.get("loc", 0), scale=params.get("scale", 1), lb=low, ub=high, conditional=True)
    return float(mean)


def _truncated_median(dist, params: dict, low: float, high: float) -> float:
    a_cdf = dist.cdf(low, **params)
    b_cdf = dist.cdf(high, **params)
    if b_cdf <= a_cdf:
        return (low + high) / 2.0
    return float(dist.ppf(0.5 * (a_cdf + b_cdf), **params))


def _fit_by_mean_median(
    dist,
    build_params: Callable[[np.ndarray], dict],
    x0: np.ndarray,
    target_mean: float,
    target_median: float,
    low: float,
    high: float,
    bounds: list[tuple[float, float]],
) -> dict:
    """Minimiza el error cuadratico entre (media, mediana) truncadas y los objetivos."""

    def loss(x: np.ndarray) -> float:
        params = build_params(x)
        try:
            m = _truncated_mean(dist, params, low, high)
            med = _truncated_median(dist, params, low, high)
        except Exception:
            return 1e12
        span = max(high - low, 1e-9)
        return ((m - target_mean) / span) ** 2 + ((med - target_median) / span) ** 2

    result = optimize.minimize(loss, x0=x0, method="Nelder-Mead", bounds=bounds)
    return build_params(result.x)


def _fit_normal(spec: ContinuousVariableSpec) -> tuple[Sampler]:
    low, high, mean, median = spec.min, spec.max, spec.mean, spec.median
    span = max(high - low, 1e-9)
    x0 = np.array([np.clip((mean + median) / 2, low, high), np.log(max(span / 6, 1e-6))])

    def build_params(x: np.ndarray) -> dict:
        return {"loc": x[0], "scale": max(np.exp(x[1]), 1e-9)}

    bounds = [(low - span, high + span), (np.log(span / 100 + 1e-9), np.log(span * 10 + 1e-9))]
    params = _fit_by_mean_median(stats.norm, build_params, x0, mean, median, low, high, bounds)

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return _truncated_sample(stats.norm, params, low, high, size, rng)

    return sampler


def _fit_lognormal(spec: ContinuousVariableSpec) -> Sampler:
    low, high, mean, median = spec.min, spec.max, spec.mean, spec.median
    eps = max(high, 1.0) * 1e-9
    low_support = max(low, eps)
    ref = max(median, eps)
    mu0 = np.log(ref)
    ratio = max(mean / ref, 1.0 + 1e-6)
    sigma0 = np.sqrt(max(2 * np.log(ratio), 1e-3))
    x0 = np.array([mu0, np.log(sigma0)])

    def build_params(x: np.ndarray) -> dict:
        return {"s": max(np.exp(x[1]), 1e-6), "loc": 0.0, "scale": np.exp(x[0])}

    bounds = [(np.log(eps), np.log(high * 10 + eps)), (np.log(1e-3), np.log(5.0))]
    params = _fit_by_mean_median(stats.lognorm, build_params, x0, mean, median, low_support, high, bounds)

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return _truncated_sample(stats.lognorm, params, low_support, high, size, rng)

    return sampler


def _fit_uniform(spec: ContinuousVariableSpec) -> Sampler:
    low, high = spec.min, spec.max

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(low, high, size=size)

    return sampler


def _fit_triangular(spec: ContinuousVariableSpec) -> Sampler:
    low, high, mean = spec.min, spec.max, spec.mean
    span = max(high - low, 1e-9)
    mode = np.clip(3 * mean - low - high, low, high)
    c = (mode - low) / span

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return stats.triang.rvs(c, loc=low, scale=span, size=size, random_state=rng)

    return sampler


def _fit_exponential(spec: ContinuousVariableSpec) -> Sampler:
    low, high, mean, median = spec.min, spec.max, spec.mean, spec.median
    span = max(high - low, 1e-9)
    ln2 = np.log(2)
    beta0 = max((mean - median) / (1 - ln2), span / 4, 1e-6)
    loc0 = np.clip(mean - beta0, low - 5 * beta0, high)
    x0 = np.array([loc0, np.log(beta0)])

    def build_params(x: np.ndarray) -> dict:
        return {"loc": x[0], "scale": max(np.exp(x[1]), 1e-9)}

    bounds = [(low - 20 * span, high), (np.log(span / 100 + 1e-9), np.log(span * 20 + 1e-9))]
    params = _fit_by_mean_median(stats.expon, build_params, x0, mean, median, low, high, bounds)

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return _truncated_sample(stats.expon, params, low, high, size, rng)

    return sampler


def _fit_beta(spec: ContinuousVariableSpec) -> Sampler:
    low, high, mean, median = spec.min, spec.max, spec.mean, spec.median
    span = max(high - low, 1e-9)
    target_mean = np.clip((mean - low) / span, 1e-3, 1 - 1e-3)
    target_median = np.clip((median - low) / span, 1e-3, 1 - 1e-3)

    k0 = 4.0
    a0 = target_mean * k0
    b0 = (1 - target_mean) * k0
    x0 = np.array([np.log(a0), np.log(b0)])

    def loss(x: np.ndarray) -> float:
        a, b = np.exp(x[0]), np.exp(x[1])
        m = a / (a + b)
        med = stats.beta.ppf(0.5, a, b)
        return (m - target_mean) ** 2 + (med - target_median) ** 2

    bounds = [(np.log(0.05), np.log(500)), (np.log(0.05), np.log(500))]
    result = optimize.minimize(loss, x0=x0, method="Nelder-Mead", bounds=bounds)
    a, b = np.exp(result.x[0]), np.exp(result.x[1])

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return low + stats.beta.rvs(a, b, size=size, random_state=rng) * span

    return sampler


_FITTERS: dict[str, Callable[[ContinuousVariableSpec], Sampler]] = {
    "normal": _fit_normal,
    "lognormal": _fit_lognormal,
    "uniform": _fit_uniform,
    "triangular": _fit_triangular,
    "exponential": _fit_exponential,
    "beta": _fit_beta,
}


def build_sampler(spec: ContinuousVariableSpec) -> Sampler:
    """Devuelve una funcion (rng, size) -> np.ndarray para la variable dada."""
    fitter = _FITTERS.get(spec.distribution)
    if fitter is None:
        raise SpecError(f"Distribucion '{spec.distribution}' no implementada.")
    return fitter(spec)
