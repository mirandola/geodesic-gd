"""Carta de Cholesky: a seleção sigma(L) = L diag(sign L_ii) não é não expansiva.

Corrigir o sinal de um único L_ii troca o sinal da coluna inteira; se a coluna
estava alinhada com L*, a distância a L* aumenta.  Exemplo 2x2 explícito e
busca aleatória de violações (Observação 6.5 do artigo).

Uso:  python cholesky_selection_counterexample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geogd.charts import CholeskyChart  # noqa: E402


def explicit_2x2() -> None:
    L_star = np.array([[1.0, 0.0], [2.0, 1.0]])
    L = np.array([[-0.1, 0.0], [2.0, 1.0]])  # L_11 < 0, resto igual a L*
    S = CholeskyChart.select(L)
    print("L* =\n", L_star, "\nL =\n", L, "\nsigma(L) =\n", S)
    print(f"||L - L*||_F     = {np.linalg.norm(L - L_star):.4f}")
    print(f"||sigma(L) - L*||_F = {np.linalg.norm(S - L_star):.4f}   (maior: não expansiva falha)")
    assert np.allclose(CholeskyChart.phi(S), CholeskyChart.phi(L))  # phi o sigma = phi


def random_search(n_trials: int = 20000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    viol = 0
    for _ in range(n_trials):
        n = int(rng.integers(2, 5))
        L_star = np.tril(rng.normal(size=(n, n)))
        L_star[np.diag_indices(n)] = np.abs(np.diag(L_star)) + 0.1
        L = L_star + 0.5 * np.tril(rng.normal(size=(n, n)))
        S = CholeskyChart.select(L)
        if np.linalg.norm(S - L_star) > np.linalg.norm(L - L_star) + 1e-12:
            viol += 1
    print(f"violações de ||sigma(L)-L*|| <= ||L-L*|| em perturbações aleatórias: {viol} de {n_trials}")


if __name__ == "__main__":
    explicit_2x2()
    random_search()
