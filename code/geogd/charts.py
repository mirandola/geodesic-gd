"""Cartas (parametrizações) usadas no artigo e na palestra.

Cada carta implementa o dicionário do artigo:

    phi   : A -> closure(U)      (sobrejetiva em todo o espaço ambiente)
    chi   : U -> V               (inversa da restrição phi|_V)
    grad_L: gradiente euclidiano do objetivo levantado L = ell o phi
            na carta, ou seja, J_phi(z)^T grad ell(phi(z)) (projetado no
            espaço tangente da carta quando V é um subespaço afim ou uma
            esfera)
    select: seleção canônica na fibra sigma: A -> closure(V), com
            phi o sigma = phi  (quando existe uma escolha natural)

O passo GGD é sempre  z_{k+1} = z_k - eta * grad_L(z_k),  x_{k+1} = phi(z_{k+1}),
eventualmente seguido de z_{k+1} <- select(z_{k+1}) (reflexão) quando se
deseja o mecanismo global do ortante (Teorema 5.5(i) do artigo).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Ortante positivo: phi(z) = z^2/4, chi(x) = 2 sqrt(x)   (Exemplo 5.1)
# ---------------------------------------------------------------------------
class OrthantChart:
    """Carta quadrática escalada do ortante positivo (lift global)."""

    name = "orthant"

    @staticmethod
    def phi(z: np.ndarray) -> np.ndarray:
        return z**2 / 4.0

    @staticmethod
    def chi(x: np.ndarray) -> np.ndarray:
        return 2.0 * np.sqrt(x)

    @staticmethod
    def grad_L(z: np.ndarray, grad_ell: np.ndarray) -> np.ndarray:
        # d/dz_i ell(z^2/4) = (z_i/2) * d_i ell(x)
        return 0.5 * z * grad_ell

    @staticmethod
    def select(z: np.ndarray) -> np.ndarray:
        # Reflexão componente a componente: sigma(z) = |z| (não expansiva).
        return np.abs(z)

    @staticmethod
    def in_chart(z: np.ndarray) -> bool:
        return bool(np.all(z > 0))


# ---------------------------------------------------------------------------
# Simplex, carta afim: phi(u) = u^2/||u||^2 em H = {sum u = 1}   (Exemplo 6.1)
# ---------------------------------------------------------------------------
class SimplexAffineChart:
    """Carta de quadrados normalizados no hiperplano afim H."""

    name = "simplex_affine"

    @staticmethod
    def phi(u: np.ndarray) -> np.ndarray:
        return u**2 / np.dot(u, u)

    @staticmethod
    def chi(x: np.ndarray) -> np.ndarray:
        s = np.sqrt(x)
        return s / s.sum()

    @staticmethod
    def grad_L(u: np.ndarray, grad_ell: np.ndarray) -> np.ndarray:
        # Gradiente ambiente de ell(u^2/||u||^2) seguido da projeção em 1^perp.
        x = SimplexAffineChart.phi(u)
        amb = (2.0 / np.dot(u, u)) * u * (grad_ell - np.dot(x, grad_ell))
        return amb - amb.mean()

    @staticmethod
    def select(u: np.ndarray) -> np.ndarray:
        # A fibra de phi sobre x intersecta closure(V) em um único ponto, logo
        # a seleção é forçada: sigma_H(u) = |u| / sum|u_j|.  Não é não
        # expansiva (ver scripts/simplex_fejer_excess.py).
        a = np.abs(u)
        return a / a.sum()

    @staticmethod
    def in_chart(u: np.ndarray) -> bool:
        return bool(np.all(u > 0))


# ---------------------------------------------------------------------------
# Simplex, carta esférica: chi(x) = sqrt(x), V = octante positivo de S^{n-1}
# ---------------------------------------------------------------------------
class SimplexSphereChart:
    """Carta esférica do simplex (Observação 6.5 do artigo).

    A carta é curva (métrica de Fisher--Rao); o passo geodésico segue
    círculos máximos via Exp^S_u(v) = cos|v| u + sin|v| v/|v|.  Fórmula
    fechada no artigo, eq. (simplex-sphere-step).
    """

    name = "simplex_sphere"

    @staticmethod
    def phi(u: np.ndarray) -> np.ndarray:
        # Extensão 0-homogênea a todo R^n \ {0}: u^2/||u||^2.
        return u**2 / np.dot(u, u)

    @staticmethod
    def chi(x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    @staticmethod
    def delta(x: np.ndarray, grad_ell: np.ndarray) -> np.ndarray:
        # delta_i = d_i ell - <x, grad ell>;  satisfaz <x, delta> = 0.
        return grad_ell - np.dot(x, grad_ell)

    @staticmethod
    def grad_L(u: np.ndarray, grad_ell: np.ndarray) -> np.ndarray:
        # Gradiente tangencial na esfera: 2 u * delta(x).
        x = u**2
        return 2.0 * u * SimplexSphereChart.delta(x, grad_ell)

    @staticmethod
    def exp_sphere(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        nv = np.linalg.norm(v)
        if nv < 1e-300:
            return u
        return np.cos(nv) * u + np.sin(nv) * v / nv

    @staticmethod
    def geodesic_step(x: np.ndarray, grad_ell: np.ndarray, eta: float) -> np.ndarray:
        """Passo GGD geodésico exato na esfera, escrito em x (fórmula fechada)."""
        d = SimplexSphereChart.delta(x, grad_ell)
        D = np.sqrt(np.sum(x * d**2))
        if D < 1e-300:
            return x.copy()
        theta = 2.0 * eta * D
        x_new = x * (np.cos(theta) - np.sin(theta) * d / D) ** 2
        # Renormalização: a fórmula preserva sum x = 1 apenas exatamente; um desvio
        # eps em sum x é amplificado por (1 + 4 eta <x, grad ell>) por passo, o que
        # torna a iteração numericamente instável sem esta projeção (custo O(n)).
        return x_new / x_new.sum()

    @staticmethod
    def step_from_grad_w(w: np.ndarray, grad_w: np.ndarray, eta: float) -> np.ndarray:
        """Passo geodésico na esfera a partir do gradiente euclidiano de
        w -> ell(w^2) (evita avaliar grad ell(x) na fronteira).  Devolve w_{k+1}."""
        w = w / np.linalg.norm(w)                    # garante ||w|| = 1 (ver geodesic_step)
        g = grad_w - np.dot(w, grad_w) * w          # projeção tangencial
        w_new = SimplexSphereChart.exp_sphere(w, -eta * g)
        return w_new / np.linalg.norm(w_new)

    @staticmethod
    def select(u: np.ndarray) -> np.ndarray:
        # sigma = P_B o |.| : composição de não expansivas que fixa u* = sqrt(x*).
        a = np.abs(u)
        return a / max(1.0, np.linalg.norm(a))

    @staticmethod
    def in_chart(u: np.ndarray) -> bool:
        return bool(np.all(u > 0))


# ---------------------------------------------------------------------------
# Cone PSD, carta de Cholesky: phi(L) = L L^T   (Exemplo 6.3)
# ---------------------------------------------------------------------------
class CholeskyChart:
    """Carta de Cholesky do cone PSD. Variável de carta: matriz triangular inferior."""

    name = "cholesky"

    @staticmethod
    def phi(L: np.ndarray) -> np.ndarray:
        return L @ L.T

    @staticmethod
    def chi(X: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(X)

    @staticmethod
    def grad_L(L: np.ndarray, grad_ell: np.ndarray) -> np.ndarray:
        # d L_chol(L)[H] = 2 <sym(grad ell(X)) L, H>_F, projetado em triangulares inferiores.
        S = 0.5 * (grad_ell + grad_ell.T)
        return np.tril(2.0 * S @ L)

    @staticmethod
    def select(L: np.ndarray) -> np.ndarray:
        # Troca o sinal das colunas com diagonal negativa. Satisfaz phi o sigma = phi,
        # mas NÃO é não expansiva (scripts/cholesky_selection_counterexample.py).
        s = np.sign(np.diag(L))
        s[s == 0] = 1.0
        return L * s[None, :]

    @staticmethod
    def in_chart(L: np.ndarray) -> bool:
        return bool(np.all(np.diag(L) > 0))


CHARTS = {
    OrthantChart.name: OrthantChart,
    SimplexAffineChart.name: SimplexAffineChart,
    SimplexSphereChart.name: SimplexSphereChart,
    CholeskyChart.name: CholeskyChart,
}
