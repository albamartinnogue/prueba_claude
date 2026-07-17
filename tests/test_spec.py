import pytest

from db_simulator.spec import SpecError, spec_from_dict


def base_raw():
    return {
        "n_rows": 100,
        "seed": 1,
        "variables": [
            {"name": "x", "type": "continuous", "distribution": "normal", "mean": 5, "median": 5, "min": 0, "max": 10},
            {"name": "cat", "type": "categorical", "categories": {"A": 0.5, "B": 0.5}},
        ],
    }


def test_valid_spec_parses():
    spec = spec_from_dict(base_raw())
    assert spec.n_rows == 100
    assert len(spec.variables) == 2


def test_duplicate_names_rejected():
    raw = base_raw()
    raw["variables"][1]["name"] = "x"
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_min_greater_than_max_rejected():
    raw = base_raw()
    raw["variables"][0]["min"] = 20
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_mean_outside_range_rejected():
    raw = base_raw()
    raw["variables"][0]["mean"] = 50
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_unknown_distribution_rejected():
    raw = base_raw()
    raw["variables"][0]["distribution"] = "made_up"
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_lognormal_requires_nonnegative_min():
    raw = base_raw()
    raw["variables"][0]["distribution"] = "lognormal"
    raw["variables"][0]["min"] = -5
    raw["variables"][0]["mean"] = 5
    raw["variables"][0]["median"] = 4
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_negative_category_proportion_rejected():
    raw = base_raw()
    raw["variables"][1]["categories"]["A"] = -0.1
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_categories_are_renormalized():
    raw = base_raw()
    raw["variables"][1]["categories"] = {"A": 1, "B": 3}
    spec = spec_from_dict(raw)
    cat_var = spec.categorical_variables()[0]
    normalized = cat_var.normalized_categories()
    assert normalized["A"] == pytest.approx(0.25)
    assert normalized["B"] == pytest.approx(0.75)


def test_correlation_matrix_valid():
    raw = base_raw()
    raw["variables"].append(
        {"name": "y", "type": "continuous", "distribution": "normal", "mean": 5, "median": 5, "min": 0, "max": 10}
    )
    raw["correlation_matrix"] = {"variables": ["x", "y"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}
    spec = spec_from_dict(raw)
    assert spec.correlation is not None


def test_correlation_matrix_rejects_categorical_variable():
    raw = base_raw()
    raw["correlation_matrix"] = {"variables": ["x", "cat"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_correlation_matrix_rejects_bad_diagonal():
    raw = base_raw()
    raw["variables"].append(
        {"name": "y", "type": "continuous", "distribution": "normal", "mean": 5, "median": 5, "min": 0, "max": 10}
    )
    raw["correlation_matrix"] = {"variables": ["x", "y"], "matrix": [[0.9, 0.5], [0.5, 1.0]]}
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_correlation_matrix_rejects_asymmetric():
    raw = base_raw()
    raw["variables"].append(
        {"name": "y", "type": "continuous", "distribution": "normal", "mean": 5, "median": 5, "min": 0, "max": 10}
    )
    raw["correlation_matrix"] = {"variables": ["x", "y"], "matrix": [[1.0, 0.5], [0.2, 1.0]]}
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_category_effect_valid():
    raw = base_raw()
    raw["variables"][0]["effects"] = [{"by": "cat", "mean_delta": {"A": 2.0}, "median_delta": {"A": 1.5}}]
    spec = spec_from_dict(raw)
    effect = spec.continuous_variables()[0].effects[0]
    assert effect.by == "cat"
    assert effect.mean_delta == {"A": 2.0}


def test_category_effect_rejects_unknown_categorical_variable():
    raw = base_raw()
    raw["variables"][0]["effects"] = [{"by": "no_existe", "mean_delta": {"A": 2.0}}]
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_category_effect_rejects_unknown_level():
    raw = base_raw()
    raw["variables"][0]["effects"] = [{"by": "cat", "mean_delta": {"Z": 2.0}}]
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_category_effect_rejects_continuous_by_reference():
    raw = base_raw()
    raw["variables"].append(
        {"name": "y", "type": "continuous", "distribution": "normal", "mean": 5, "median": 5, "min": 0, "max": 10}
    )
    raw["variables"][0]["effects"] = [{"by": "y", "mean_delta": {"A": 2.0}}]
    with pytest.raises(SpecError):
        spec_from_dict(raw)


def test_category_effect_rejects_combination_with_correlation_matrix():
    raw = base_raw()
    raw["variables"].append(
        {"name": "y", "type": "continuous", "distribution": "normal", "mean": 5, "median": 5, "min": 0, "max": 10}
    )
    raw["variables"][0]["effects"] = [{"by": "cat", "mean_delta": {"A": 2.0}}]
    raw["correlation_matrix"] = {"variables": ["x", "y"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}
    with pytest.raises(SpecError):
        spec_from_dict(raw)
