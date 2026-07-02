"""Muestreo de variables categoricas a partir de proporciones objetivo."""

from __future__ import annotations

import numpy as np

from .spec import CategoricalVariableSpec


def sample_categorical(spec: CategoricalVariableSpec, size: int, rng: np.random.Generator) -> np.ndarray:
    categories = spec.normalized_categories()
    labels = list(categories.keys())
    probs = np.array(list(categories.values()))
    return rng.choice(labels, size=size, p=probs)
