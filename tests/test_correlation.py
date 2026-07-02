import numpy as np

from db_simulator.correlation import induce_correlation, nearest_correlation_matrix

N = 30000


def test_induce_correlation_approximates_target():
    rng = np.random.default_rng(0)
    target = np.array([[1.0, 0.7, -0.3], [0.7, 1.0, 0.0], [-0.3, 0.0, 1.0]])

    x1 = rng.normal(10, 2, N)
    x2 = rng.exponential(5, N)
    x3 = rng.uniform(0, 1, N)
    samples = np.column_stack([x1, x2, x3])

    result = induce_correlation(samples, target, rng)

    # Las marginales (multiconjunto de valores) no deben cambiar, solo el orden.
    for j in range(3):
        assert np.allclose(np.sort(result[:, j]), np.sort(samples[:, j]))

    # El metodo Iman-Conover aproxima la correlacion Pearson objetivo trabajando
    # sobre la estructura de rangos; con marginales muy asimetricas (p. ej.
    # exponencial) el resultado se acerca pero no coincide de forma exacta.
    achieved = np.corrcoef(result, rowvar=False)
    assert np.allclose(achieved, target, atol=0.1)


def test_nearest_correlation_matrix_fixes_invalid_input():
    invalid = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = nearest_correlation_matrix(invalid)
    eigvals = np.linalg.eigvalsh(fixed)
    assert (eigvals >= -1e-8).all()
    assert np.allclose(np.diag(fixed), 1.0)
