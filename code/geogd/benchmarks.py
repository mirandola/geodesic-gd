"""Os três benchmarks alinhados à fronteira da Seção 7 do artigo.

Cada benchmark é um objeto `Problem` com:

    ell(x)          objetivo no espaço original (x no fecho do conjunto viável)
    grad_x(x)       gradiente euclidiano em x (usado por PGD/APGD e MD/AMD)
    grad_chart(z)   gradiente euclidiano do objetivo levantado na carta
                    (usado por GGD/AGGD; evita a singularidade de grad_x na fronteira)
    project(v)      projeção euclidiana no conjunto viável
    mirror(x,g,eta) passo de mirror descent com o potencial do artigo
    chart           uma das classes de geogd.charts
    x_star, f_star  minimizador e valor ótimo (conhecidos por construção)
    leakage(x)      observável de "vazamento" de fronteira (Subseção 7.4)
    x0              inicialização viável comum a todos os métodos

Observação de revisão (ver README): os objetivos são quadráticos NA CARTA,
logo são suaves na carta, mas grad_x é singular na fronteira (fatores 1/sqrt(x)
e L^{-1}). Os métodos em x recebem gradientes com um piso numérico `EPS`.
Isso favorece os métodos de carta e deve ser discutido/repensado antes da
versão final dos experimentos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .charts import CholeskyChart, OrthantChart, SimplexAffineChart

EPS = 1e-12  # piso numérico para gradientes em x na fronteira


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    """Projeção euclidiana no simplex unitário (algoritmo de ordenação)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1.0))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def project_psd(Y: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Projeção espectral com limiar eps>0 (mantém os iterados estritamente PD)."""
    S = 0.5 * (Y + Y.T)
    lam, Q = np.linalg.eigh(S)
    return (Q * np.maximum(lam, eps)) @ Q.T


def random_spd(rng: np.random.Generator, n: int, cond: float = 20.0) -> np.ndarray:
    """Matriz SPD com espectro log-uniforme em [1, cond]."""
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.exp(rng.uniform(0.0, np.log(cond), size=n))
    return (Q * lam) @ Q.T


@dataclass
class Problem:
    name: str
    ell: Callable[[np.ndarray], float]
    grad_x: Callable[[np.ndarray], np.ndarray]
    grad_chart: Callable[[np.ndarray], np.ndarray]
    project: Callable[[np.ndarray], np.ndarray]
    mirror: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    chart: type
    x_star: np.ndarray
    f_star: float
    x0: np.ndarray
    leakage: Callable[[np.ndarray], float]
    step_grid: np.ndarray = field(default_factory=lambda: np.logspace(-3, 0, 13))

    family: str = "chart-quadratic"
    grad_w: Callable[[np.ndarray], np.ndarray] | None = None  # gradiente de w -> ell(w^2) (carta esférica)

    def gap(self, x: np.ndarray) -> float:
        return float(self.ell(x) - self.f_star)


def _chain_grad_chart(problem_grad_x, chart):
    """grad_chart pela regra da cadeia (Lema 2.3): J_phi(z)^T grad ell(phi(z))."""
    return lambda z: chart.grad_L(z, problem_grad_x(chart.phi(z)))


# ---------------------------------------------------------------------------
# Ortante: ell(x) = 1/2 (z - z*)^T A (z - z*),  z = 2 sqrt(x),  n = 40, |supp z*| = 6
# ---------------------------------------------------------------------------
def make_orthant(rng: np.random.Generator, n: int = 40, support: int = 6) -> Problem:
    A = random_spd(rng, n)
    z_star = np.zeros(n)
    idx = rng.choice(n, size=support, replace=False)
    z_star[idx] = rng.uniform(0.5, 2.0, size=support)
    x_star = z_star**2 / 4.0
    S_star = idx

    def ell(x):
        z = 2.0 * np.sqrt(np.maximum(x, 0.0))
        d = z - z_star
        return 0.5 * d @ A @ d

    def grad_x(x):
        z = 2.0 * np.sqrt(np.maximum(x, EPS))
        # d ell/dx = J_chi^T A (z - z*),  J_chi = diag(1/sqrt(x))
        return (A @ (z - z_star)) / np.sqrt(np.maximum(x, EPS))

    def grad_chart(z):
        # L(z) = ell(z^2/4) = 1/2 (|z| - z*)^T A (|z| - z*)
        s = np.sign(z)
        s[s == 0] = 1.0
        return s * (A @ (np.abs(z) - z_star))

    def mirror(x, g, eta):
        # Entropia separável: x_{k+1} = x_k exp(-eta g)
        return x * np.exp(-eta * g)

    def leakage(x):
        mask = np.ones(n, dtype=bool)
        mask[S_star] = False
        return float(np.sum(x[mask]))

    x0 = rng.uniform(0.5, 1.5, size=n)
    return Problem(
        name="orthant", ell=ell, grad_x=grad_x, grad_chart=grad_chart,
        project=lambda v: np.maximum(v, 0.0), mirror=mirror, chart=OrthantChart,
        x_star=x_star, f_star=0.0, x0=x0, leakage=leakage,
        step_grid=np.logspace(-5, 0, 21),
    )


# ---------------------------------------------------------------------------
# Simplex: ell(x) = 1/2 (u - u*)^T A (u - u*),  u = sqrt(x)/sum sqrt(x),  n = 30, |supp u*| = 3
# ---------------------------------------------------------------------------
def make_simplex(rng: np.random.Generator, n: int = 30, support: int = 3) -> Problem:
    # A age no hiperplano H: A = P A0 P com P = I - 11^T/n.
    P = np.eye(n) - np.ones((n, n)) / n
    A = P @ random_spd(rng, n) @ P
    u_star = np.zeros(n)
    idx = rng.choice(n, size=support, replace=False)
    u_star[idx] = rng.uniform(0.5, 2.0, size=support)
    u_star /= u_star.sum()
    x_star = u_star**2 / np.dot(u_star, u_star)
    S_star = idx

    def u_of(x):
        s = np.sqrt(np.maximum(x, 0.0))
        return s / s.sum()

    def ell(x):
        d = u_of(x) - u_star
        return 0.5 * d @ A @ d

    def grad_x(x):
        s = np.sqrt(np.maximum(x, EPS))
        sig = s.sum()
        w = A @ (s / sig - u_star)
        # d u_i/d x_j = (delta_ij sig - s_i) / (2 sig^2 s_j)
        return (w * sig - np.dot(w, s)) / (2.0 * sig**2 * s)

    def grad_chart(u):
        # L(u) = ell(u^2/||u||^2) = 1/2 (|u|/sum|u| - u*)^T A (...), projetado em 1^perp.
        a = np.abs(u)
        sig = a.sum()
        w = A @ (a / sig - u_star)
        s = np.sign(u)
        s[s == 0] = 1.0
        g = s * (w * sig - np.dot(w, a)) / sig**2
        return g - g.mean()

    def mirror(x, g, eta):
        # Pesos multiplicativos (entropia no simplex).
        y = x * np.exp(-eta * (g - g.max()))
        return y / y.sum()

    def grad_w(w):
        # ell(w^2) = 1/2 (w/sum w - u*)^T A (...): regular em w >= 0 (carta esférica).
        a = np.abs(w)
        sig = a.sum()
        v = A @ (a / sig - u_star)
        s = np.sign(w); s[s == 0] = 1.0
        return s * (v * sig - np.dot(v, a)) / sig**2

    def leakage(x):
        mask = np.ones(n, dtype=bool)
        mask[S_star] = False
        return float(np.sum(x[mask]))

    x0 = rng.uniform(0.5, 1.5, size=n)
    x0 /= x0.sum()
    return Problem(
        name="simplex", ell=ell, grad_x=grad_x, grad_chart=grad_chart,
        project=project_simplex, mirror=mirror, chart=SimplexAffineChart,
        x_star=x_star, f_star=0.0, x0=x0, leakage=leakage,
        step_grid=np.logspace(-4, 0, 17), grad_w=grad_w,
    )


# ---------------------------------------------------------------------------
# PSD: ell(X) = 1/2 ||W o (L - L*)||_F^2,  X = L L^T,  n = 5, rank L* = 2
# ---------------------------------------------------------------------------
def _phi_lower(M: np.ndarray) -> np.ndarray:
    """Operador Phi da derivada de Cholesky: tril(M) com diagonal pela metade."""
    return np.tril(M) - 0.5 * np.diag(np.diag(M))


def make_psd(rng: np.random.Generator, n: int = 5, rank: int = 2) -> Problem:
    W = np.tril(rng.uniform(0.5, 1.5, size=(n, n)))
    B = rng.normal(size=(n, rank))
    L_star = np.tril(np.linalg.qr(B)[0] @ np.diag(rng.uniform(0.5, 2.0, size=rank)) @ rng.normal(size=(rank, n)))
    # Garante L* triangular inferior de posto `rank` com diagonal >= 0.
    L_star = np.tril(L_star)
    L_star[np.diag_indices(n)] = np.abs(np.diag(L_star))
    L_star[:, rank:] = 0.0  # zera colunas -> posto <= rank
    X_star = L_star @ L_star.T
    W2 = W**2

    def chol_safe(X):
        S = 0.5 * (X + X.T)
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return np.linalg.cholesky(project_psd(S, EPS))

    def ell(X):
        L = chol_safe(X)
        return 0.5 * np.sum((W * (L - L_star)) ** 2)

    def grad_x(X):
        # Regra da cadeia através do fator de Cholesky:
        # dL = L Phi(L^{-1} dX L^{-T});  adjunto: dX* = L^{-T} Phi(L^T G) L^{-1}.
        L = chol_safe(X)
        G = np.tril(W2 * (L - L_star))
        Linv = np.linalg.solve(L, np.eye(n))
        M = Linv.T @ _phi_lower(L.T @ G) @ Linv
        return 0.5 * (M + M.T)

    def grad_chart(L):
        return np.tril(W2 * (L - L_star))

    def mirror(X, g, eta):
        # Potencial de Burg psi(X) = -log det X: X_{k+1} = (X_k^{-1} + eta sym(g))^{-1}.
        Y = np.linalg.inv(X) + eta * 0.5 * (g + g.T)
        return np.linalg.inv(project_psd(Y, EPS))

    def leakage(X):
        lam = np.sort(np.linalg.eigvalsh(0.5 * (X + X.T)))[::-1]
        return float(np.sum(np.maximum(lam[rank:], 0.0)))

    L0 = np.tril(rng.normal(size=(n, n)))
    L0[np.diag_indices(n)] = rng.uniform(0.8, 1.5, size=n)
    X0 = L0 @ L0.T
    return Problem(
        name="psd", ell=ell, grad_x=grad_x, grad_chart=grad_chart,
        project=lambda Y: project_psd(Y, 1e-10), mirror=mirror, chart=CholeskyChart,
        x_star=X_star, f_star=0.0, x0=X0, leakage=leakage,
        step_grid=np.logspace(-3, 0.5, 15),
    )


# ===========================================================================
# Família B (controle): quadráticas EUCLIDIANAS  ell(x) = 1/2 (x - xbar)^T A (x - xbar),
# suaves em todo R^n, com minimizador restrito x* na fronteira construído por KKT:
#   grad ell(x*) = A (x* - xbar) = mu,  mu >= 0,  mu_i > 0 <=> x*_i = 0   (ortante)
# de modo que xbar = x* - A^{-1} mu.  Todos os seis métodos recebem gradientes
# regulares; os métodos de carta usam a regra da cadeia (Lema 2.3).  Aqui L = ell o phi
# NÃO é convexa na carta em geral (Obs. 5.6): o teste é do regime (a') do Teo. 5.5.
# ===========================================================================
def make_orthant_euclid(rng: np.random.Generator, n: int = 40, support: int = 6) -> Problem:
    A = random_spd(rng, n)
    x_star = np.zeros(n)
    idx = rng.choice(n, size=support, replace=False)
    x_star[idx] = rng.uniform(0.5, 2.0, size=support)
    mu = rng.uniform(0.5, 2.0, size=n); mu[idx] = 0.0          # complementaridade estrita
    xbar = x_star - np.linalg.solve(A, mu)
    S_star = idx

    def ell(x):
        d = x - xbar
        return 0.5 * d @ A @ d

    def grad_x(x):
        return A @ (x - xbar)

    def mirror(x, g, eta):
        return x * np.exp(-eta * g)

    def leakage(x):
        mask = np.ones(n, dtype=bool); mask[S_star] = False
        return float(np.sum(x[mask]))

    x0 = rng.uniform(0.5, 1.5, size=n)
    return Problem(
        name="orthant-euclid", ell=ell, grad_x=grad_x,
        grad_chart=_chain_grad_chart(grad_x, OrthantChart),
        project=lambda v: np.maximum(v, 0.0), mirror=mirror, chart=OrthantChart,
        x_star=x_star, f_star=ell(x_star), x0=x0, leakage=leakage,
        step_grid=np.logspace(-4, 0, 17), family="euclidean-quadratic",
    )


def make_simplex_euclid(rng: np.random.Generator, n: int = 30, support: int = 3) -> Problem:
    A = random_spd(rng, n)
    x_star = np.zeros(n)
    idx = rng.choice(n, size=support, replace=False)
    x_star[idx] = rng.uniform(0.5, 2.0, size=support); x_star /= x_star.sum()
    # KKT no simplex: A (x* - xbar) = mu - nu 1,  mu >= 0, mu_i x*_i = 0.  Tomamos nu = 0.
    mu = rng.uniform(0.5, 2.0, size=n); mu[idx] = 0.0
    xbar = x_star - np.linalg.solve(A, mu)
    S_star = idx

    def ell(x):
        d = x - xbar
        return 0.5 * d @ A @ d

    def grad_x(x):
        return A @ (x - xbar)

    def mirror(x, g, eta):
        y = x * np.exp(-eta * (g - g.max()))
        return y / y.sum()

    def leakage(x):
        mask = np.ones(n, dtype=bool); mask[S_star] = False
        return float(np.sum(x[mask]))

    x0 = rng.uniform(0.5, 1.5, size=n); x0 /= x0.sum()
    return Problem(
        name="simplex-euclid", ell=ell, grad_x=grad_x,
        grad_chart=_chain_grad_chart(grad_x, SimplexAffineChart),
        project=project_simplex, mirror=mirror, chart=SimplexAffineChart,
        x_star=x_star, f_star=ell(x_star), x0=x0, leakage=leakage,
        step_grid=np.logspace(-4, 0, 17), family="euclidean-quadratic",
        grad_w=lambda w: 2.0 * w * grad_x(w**2),
    )


def make_psd_euclid(rng: np.random.Generator, n: int = 5, rank: int = 2) -> Problem:
    # ell(X) = 1/2 <X - Xbar, W (X - Xbar) W>,  W SPD.  KKT: W (X* - Xbar) W = M >= 0, M X* = 0.
    W = random_spd(rng, n, cond=5.0)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.zeros(n); lam[:rank] = rng.uniform(0.5, 2.0, size=rank)
    X_star = (Q * lam) @ Q.T
    dmu = np.zeros(n); dmu[rank:] = rng.uniform(0.5, 2.0, size=n - rank)   # complementaridade estrita
    M = (Q * dmu) @ Q.T
    Winv = np.linalg.inv(W)
    Xbar = X_star - Winv @ M @ Winv

    def ell(X):
        D = 0.5 * (X + X.T) - Xbar
        return 0.5 * np.sum(D * (W @ D @ W))

    def grad_x(X):
        D = 0.5 * (X + X.T) - Xbar
        return W @ D @ W

    def mirror(X, g, eta):
        Y = np.linalg.inv(X) + eta * 0.5 * (g + g.T)
        return np.linalg.inv(project_psd(Y, EPS))

    def leakage(X):
        lam = np.sort(np.linalg.eigvalsh(0.5 * (X + X.T)))[::-1]
        return float(np.sum(np.maximum(lam[rank:], 0.0)))

    L0 = np.tril(rng.normal(size=(n, n)))
    L0[np.diag_indices(n)] = rng.uniform(0.8, 1.5, size=n)
    X0 = L0 @ L0.T
    return Problem(
        name="psd-euclid", ell=ell, grad_x=grad_x,
        grad_chart=_chain_grad_chart(grad_x, CholeskyChart),
        project=lambda Y: project_psd(Y, 1e-10), mirror=mirror, chart=CholeskyChart,
        x_star=X_star, f_star=ell(X_star), x0=X0, leakage=leakage,
        step_grid=np.logspace(-4, 0.5, 19), family="euclidean-quadratic",
    )


def make_all(seed: int = 0) -> dict[str, Problem]:
    """Família A (quadráticas na carta), como no artigo."""
    rng = np.random.default_rng(seed)
    return {
        "orthant": make_orthant(rng),
        "simplex": make_simplex(rng),
        "psd": make_psd(rng),
    }


def make_all_euclid(seed: int = 0) -> dict[str, Problem]:
    """Família B (quadráticas euclidianas, minimizador na fronteira via KKT)."""
    rng = np.random.default_rng(seed + 1000)
    return {
        "orthant": make_orthant_euclid(rng),
        "simplex": make_simplex_euclid(rng),
        "psd": make_psd_euclid(rng),
    }
