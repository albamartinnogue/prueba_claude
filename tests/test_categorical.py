import numpy as np
import pytest

from db_simulator.categorical import sample_categorical
from db_simulator.spec import CategoricalVariableSpec

N = 20000


def test_categorical_proportions_are_approximated():
    spec = CategoricalVariableSpec(name="c", categories={"A": 0.7, "B": 0.2, "C": 0.1})
    rng = np.random.default_rng(0)
    values = sample_categorical(spec, N, rng)

    unique, counts = np.unique(values, return_counts=True)
    freqs = dict(zip(unique, counts / N))

    assert freqs["A"] == pytest.approx(0.7, abs=0.02)
    assert freqs["B"] == pytest.approx(0.2, abs=0.02)
    assert freqs["C"] == pytest.approx(0.1, abs=0.02)


def test_categorical_proportions_are_renormalized_when_not_summing_to_one():
    spec = CategoricalVariableSpec(name="c", categories={"A": 2, "B": 2})
    rng = np.random.default_rng(0)
    values = sample_categorical(spec, N, rng)
    unique, counts = np.unique(values, return_counts=True)
    freqs = dict(zip(unique, counts / N))
    assert freqs["A"] == pytest.approx(0.5, abs=0.02)
    assert freqs["B"] == pytest.approx(0.5, abs=0.02)
