"""Induccion aproximada de correlaciones entre variables continuas.

Se implementa el metodo de Iman-Conover: dado un conjunto de muestras ya
generadas de forma independiente a partir de sus distribuciones marginales
(por lo que sus valores individuales son correctos), se reordenan dentro de
cada columna para que el orden relativo (rango) siga una estructura normal
multivariante con la correlacion deseada. Esto induce una correlacion de
Spearman/Pearson aproximada a la solicitada sin alterar los valores
muestreados (las marginales no cambian, solo su orden).
"""

from __future__ import annotations

import numpy as np


def nearest_correlation_matrix(corr: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Proyecta una matriz simetrica a la matriz de correlacion valida (PSD) mas cercana.

    Util cuando el usuario introduce una matriz de correlaciones inconsistente
    (no semidefinida positiva). Se realiza recorte de autovalores negativos y
    se renormaliza la diagonal a 1.
    """
    corr = (corr + corr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    np.fill_diagonal(corr_psd, 1.0)
    return corr_psd


def induce_correlation(samples: np.ndarray, target_corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reordena las columnas de `samples` (n x k) para aproximar `target_corr`.

    `samples` debe contener ya los valores correctos por columna (mismas
    marginales); esta funcion solo cambia el orden de cada columna.
    """
    n, k = samples.shape
    corr = nearest_correlation_matrix(np.asarray(target_corr, dtype=float))
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr = nearest_correlation_matrix(corr, epsilon=1e-4)
        L = np.linalg.cholesky(corr)

    z = rng.standard_normal((n, k)) @ L.T

    result = np.empty_like(samples)
    for j in range(k):
        rank_order = np.argsort(np.argsort(z[:, j]))
        sorted_column = np.sort(samples[:, j])
        result[:, j] = sorted_column[rank_order]
    return result
