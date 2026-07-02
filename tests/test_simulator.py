import pytest

from db_simulator.simulator import Simulator
from db_simulator.spec import SpecError

N = 5000


def make_spec():
    return {
        "n_rows": N,
        "seed": 7,
        "variables": [
            {"name": "edad", "type": "continuous", "distribution": "normal", "mean": 42, "median": 40, "min": 18, "max": 85},
            {
                "name": "ingresos",
                "type": "continuous",
                "distribution": "lognormal",
                "mean": 32000,
                "median": 27000,
                "min": 5000,
                "max": 250000,
            },
            {"name": "genero", "type": "categorical", "categories": {"H": 0.5, "M": 0.5}},
        ],
        "correlation_matrix": {"variables": ["edad", "ingresos"], "matrix": [[1.0, 0.6], [0.6, 1.0]]},
    }


def test_generate_produces_expected_shape_and_columns():
    sim = Simulator.from_dict(make_spec())
    df = sim.generate()
    assert len(df) == N
    assert list(df.columns) == ["edad", "ingresos", "genero"]


def test_generate_is_reproducible_with_seed():
    sim = Simulator.from_dict(make_spec())
    df1 = sim.generate(seed=123)
    df2 = sim.generate(seed=123)
    pd_equal = (df1["edad"].to_numpy() == df2["edad"].to_numpy()).all()
    assert pd_equal


def test_generate_with_report_matches_targets_approximately():
    sim = Simulator.from_dict(make_spec())
    df, report = sim.generate_with_report()

    edad_stats = report.continuous["edad"]
    assert edad_stats["achieved_mean"] == pytest.approx(edad_stats["target_mean"], abs=3)
    assert df["edad"].min() >= 18
    assert df["edad"].max() <= 85

    genero_stats = report.categorical["genero"]
    for target, achieved in genero_stats.values():
        assert achieved == pytest.approx(target, abs=0.05)

    achieved_corr = report.correlation["achieved"].loc["edad", "ingresos"]
    assert achieved_corr == pytest.approx(0.6, abs=0.1)


def test_n_rows_and_seed_can_be_overridden():
    sim = Simulator.from_dict(make_spec())
    df = sim.generate(n_rows=250, seed=1)
    assert len(df) == 250


def test_invalid_spec_raises_spec_error():
    raw = make_spec()
    raw["variables"][0]["min"] = 200
    with pytest.raises(SpecError):
        Simulator.from_dict(raw)
