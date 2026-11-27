"""Verificações numéricas da análise do chart esférico do simplex (Subseção 6.3).

Notação: u em S^{n-1}, x = u^2, L(u) = ell(x); gbar = <x, grad ell>,
delta_i = d_i ell - gbar, D^2 = sum_i x_i delta_i^2.

(a) Hessiana riemanniana:  Hess_S L(u)[v,v] = 4 (u*v)^T Hess ell (u*v) + 2 sum v_i^2 delta_i,
    contra diferenças finitas de t -> L(Exp_u(t v)); e a cota beta_S = 4 beta_ell + 2 osc(grad ell).
(b) Lema de descida na esfera inteira:  Delta_{k+1} <= Delta_k - 2 eta D_k^2  (eta <= 1/beta_S).
(c) Dominação de gradiente de grau 2 (ell convexa):  Delta(x) <= sqrt(1 + chi2(x*||x)) D(x).
(d) Complementaridade estrita:  instância estrita (delta_i(x*) >= gamma > 0 em Z) contra
    instância degenerada (delta_i(x*) = 0 para algum i em Z).  Previsão: linear vs sublinear.
(e) O passo geodésico NÃO é Fejér em geral (||u_{k+1} - u*|| pode exceder ||u_k - u*||),
    embora a seleção sigma = |.| seja não expansiva.  Contraexemplo por busca.

Uso:  python sphere_rate_checks.py [--plot]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geogd.benchmarks import random_spd  # noqa: E402
from geogd.charts import SimplexSphereChart as S  # noqa: E402

rng = np.random.default_rng(0)


def delta_D(x, g):
    d = g - x @ g
    return d, np.sqrt(np.sum(x * d**2))


# ---------------------------------------------------------------------------
# (a) Hessiana riemanniana e constante de suavidade
# ---------------------------------------------------------------------------
def check_hessian(n_trials: int = 300) -> tuple[float, int]:
    err, viol = 0.0, 0
    for _ in range(n_trials):
        n = int(rng.integers(3, 8))
        A = random_spd(rng, n); xbar = rng.normal(size=n)
        ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
        grad = lambda x: A @ (x - xbar)
        u = rng.normal(size=n); u /= np.linalg.norm(u)          # esfera inteira, não só o octante
        v = rng.normal(size=n); v -= (v @ u) * u; v /= np.linalg.norm(v)
        x = u**2; d, _ = delta_D(x, grad(x))
        hess_formula = 4 * (u * v) @ A @ (u * v) + 2 * np.sum(v**2 * d)
        h = 1e-4
        f = lambda t: ell(S.exp_sphere(u, t * v) ** 2)
        hess_fd = (f(h) - 2 * f(0) + f(-h)) / h**2
        err = max(err, abs(hess_formula - hess_fd))
        # cota: beta_ell = ||A||, osc(grad ell) sobre o simplex fechado (aprox. por amostragem nos vértices)
        beta_ell = np.linalg.norm(A, 2)
        verts = np.eye(n); osc = max(np.ptp(grad(verts[i])) for i in range(n))
        # osc real em x (basta): |delta_i| <= max - min de grad ell(x)
        osc_x = np.ptp(grad(x))
        if abs(hess_formula) > 4 * beta_ell + 2 * osc_x + 1e-9:
            viol += 1
    return err, viol


# ---------------------------------------------------------------------------
# (b) descida monótona com passo 1/beta_S
# ---------------------------------------------------------------------------
def check_descent(n_inst: int = 50, iters: int = 300) -> tuple[int, float]:
    """Conta violações acima do ruído de ponto flutuante e devolve a pior violação relativa."""
    viol, worst = 0, 0.0
    for _ in range(n_inst):
        n = int(rng.integers(3, 10))
        A = random_spd(rng, n); xbar = rng.normal(size=n)
        ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
        grad = lambda x: A @ (x - xbar)
        # beta_S = 4 beta_ell + 2 osc(grad ell) sobre o simplex fechado; osc é atingida nos vértices
        beta_ell = np.linalg.norm(A, 2)
        G = np.array([grad(np.eye(n)[i]) for i in range(n)])
        osc = G.max() - G.min()          # cota superior grosseira de sup_x osc(grad ell(x))
        eta = 1.0 / (4 * beta_ell + 2 * osc)
        x = rng.random(n); x /= x.sum()
        for _ in range(iters):
            d, D = delta_D(x, grad(x))
            x_new = S.geodesic_step(x, grad(x), eta)
            v = (ell(x_new) - ell(x) + 2 * eta * D**2) / max(1.0, abs(ell(x)))
            worst = max(worst, v)
            if v > 1e-9:
                viol += 1
            x = x_new
    return viol, worst


# ---------------------------------------------------------------------------
# (c) Delta <= sqrt(1 + chi2) D  para ell convexa
# ---------------------------------------------------------------------------
def check_chi2_bound(n_trials: int = 1500) -> int:
    viol = 0
    for _ in range(n_trials):
        n = int(rng.integers(2, 8))
        A = random_spd(rng, n); xbar = rng.normal(size=n)
        ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
        grad = lambda x: A @ (x - xbar)
        # x* = minimizador no simplex por PGD longo (precisão suficiente para o teste)
        from geogd.benchmarks import project_simplex
        xs = np.ones(n) / n; L2 = np.linalg.norm(A, 2)
        for _ in range(400):
            xs = project_simplex(xs - grad(xs) / L2)
        x = rng.random(n) + 1e-3; x /= x.sum()
        Delta = ell(x) - ell(xs)
        d, D = delta_D(x, grad(x))
        chi2 = np.sum(xs**2 / x) - 1.0
        if Delta > np.sqrt(1 + chi2) * D + 1e-9:
            viol += 1
    return viol


# ---------------------------------------------------------------------------
# (d) complementaridade estrita vs degenerada
# ---------------------------------------------------------------------------
def kkt_instance(n: int, support: int, degenerate: int, seed: int):
    """ell(x) = 1/2 (x-xbar)^T A (x-xbar) com x* prescrito por KKT:
    A(x* - xbar) = mu, mu_i = 0 em S, mu_i > 0 em Z exceto `degenerate` índices com mu_i = 0."""
    r = np.random.default_rng(seed)
    A = random_spd(r, n)
    x_star = np.zeros(n); idx = r.choice(n, support, replace=False)
    x_star[idx] = r.uniform(0.5, 2.0, support); x_star /= x_star.sum()
    Z = [i for i in range(n) if i not in idx]
    mu = np.zeros(n); mu[Z] = r.uniform(0.5, 2.0, len(Z))
    mu[Z[:degenerate]] = 0.0
    xbar = x_star - np.linalg.solve(A, mu)
    ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
    grad = lambda x: A @ (x - xbar)
    return ell, grad, x_star, r


def run_sphere(ell, grad, x0, eta, iters):
    x = x0.copy(); gaps = []
    for _ in range(iters):
        gaps.append(ell(x))
        x = S.geodesic_step(x, grad(x), eta)
    return np.array(gaps)


def check_complementarity(n: int = 20, support: int = 4, iters: int = 4000, seed: int = 0, eta: float = 0.05):
    out = {}
    for label, deg in [("estrita", 0), ("degenerada", 1)]:
        ell, grad, x_star, r = kkt_instance(n, support, deg, seed)
        x0 = r.random(n); x0 /= x0.sum()
        g = run_sphere(ell, grad, x0, eta, iters) - ell(x_star)
        k = np.arange(1, iters + 1)
        hit = int(np.argmax(g < 1e-12)) if np.any(g < 1e-12) else None
        out[label] = {"gap_100": g[100], "gap_end": g[-1], "k_1e-12": hit,
                      "gap_k2_tail": float(np.mean((g * k**2)[-100:]))}
    return out


# ---------------------------------------------------------------------------
# (e) o passo geodésico (com reflexão) não é Fejér em distância de Hellinger
# ---------------------------------------------------------------------------
def fejer_of_step_violations(n_inst: int = 40, iters: int = 200) -> tuple[int, int, float]:
    """Ao longo de trajetórias monótonas de GGD-S em instâncias KKT, conta os passos com
    ||u_{k+1} - u*|| > ||u_k - u*||,  u = sqrt(x).  A seleção |.| é não expansiva, mas o
    passo geodésico não é: a prova de Fejér do ortante exigiria convexidade geodésica."""
    r0 = np.random.default_rng(5)
    tot, viol, worst = 0, 0, 1.0
    for seed in range(n_inst):
        n = int(r0.integers(3, 12))
        ell, grad, xs, r = kkt_instance(n, int(r0.integers(1, n)), 0, seed)
        us = np.sqrt(xs)
        for eta in (0.01, 0.05, 0.2, 0.5):
            x = r.random(n); x /= x.sum()
            for _ in range(iters):
                xn = S.geodesic_step(x, grad(x), eta)
                if ell(xn) > ell(x):
                    break                      # passo instável para este eta: ignora
                d0, d1 = np.linalg.norm(np.sqrt(x) - us), np.linalg.norm(np.sqrt(xn) - us)
                tot += 1
                if d1 > d0 * (1 + 1e-12):
                    viol += 1; worst = max(worst, d1 / d0)
                x = xn
    return viol, tot, worst


if __name__ == "__main__":
    err, viol = check_hessian()
    print(f"(a) Hessiana esférica vs diferenças finitas: erro máximo {err:.2e}; violações da cota 4 beta_ell + 2 osc: {viol}")
    vb, wb = check_descent()
    print(f"(b) violações de Delta_(k+1) <= Delta_k - 2 eta D_k^2 com eta = 1/beta_S: {vb} (pior violação relativa {wb:.1e}, ruído de ponto flutuante)")
    print(f"(c) violações de Delta <= sqrt(1+chi2) D (ell convexa, 1500 amostras): {check_chi2_bound()}")
    res = check_complementarity()
    print("(d) GGD-S, quadrática euclidiana com x* por KKT (n=20, |S|=4), eta = 0.05, 4000 iterações:")
    for label, d in res.items():
        print(f"    {label:12s} gap(k=100) = {d['gap_100']:.2e}   gap(k=4000) = {d['gap_end']:.2e}   "
              f"1o k com gap<1e-12: {d['k_1e-12']}   media de gap*k^2 nas ultimas 100: {d['gap_k2_tail']:.3f}")
    v, t, w = fejer_of_step_violations()
    print(f"(e) passo geodésico + reflexão: ||u_(k+1)-u*|| > ||u_k-u*|| em {v} de {t} passos monótonos (pior razão {w:.3f})")
