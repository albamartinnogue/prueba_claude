import numpy as np
import pytest

from db_simulator.distributions import build_sampler
from db_simulator.spec import ContinuousVariableSpec

N = 20000
TOL_REL = 0.15


@pytest.mark.parametrize(
    "distribution,mean,median,low,high",
    [
        ("normal", 42, 40, 18, 85),
        ("lognormal", 32000, 27000, 5000, 250000),
        ("uniform", 5, 5, 0, 10),
        ("triangular", 6, 6, 0, 10),
        ("exponential", 24, 15, 0, 240),
        ("beta", 6, 6.2, 0, 10),
    ],
)
def test_sample_within_bounds_and_close_to_target(distribution, mean, median, low, high):
    spec = ContinuousVariableSpec(
        name="v", distribution=distribution, mean=mean, median=median, min=low, max=high
    )
    sampler = build_sampler(spec)
    rng = np.random.default_rng(0)
    values = sampler(rng, N)

    assert values.min() >= low - 1e-9
    assert values.max() <= high + 1e-9

    span = high - low
    assert abs(np.mean(values) - mean) <= max(TOL_REL * span, TOL_REL * abs(mean))
    assert abs(np.median(values) - median) <= max(TOL_REL * span, TOL_REL * abs(median))


def test_uniform_ignores_target_and_covers_range():
    spec = ContinuousVariableSpec(name="v", distribution="uniform", mean=1, median=1, min=0, max=10)
    sampler = build_sampler(spec)
    rng = np.random.default_rng(1)
    values = sampler(rng, N)
    assert values.min() >= 0
    assert values.max() <= 10
    assert abs(np.mean(values) - 5) < 0.5
