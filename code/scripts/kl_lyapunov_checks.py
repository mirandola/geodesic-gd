"""Verificações numéricas da análise por divergência de Kullback--Leibler (função de Lyapunov
do mirror descent) aplicada ao passo geodésico: ortante (Teorema `thm:orthant-kl`) e chart
esférico do simplex (Proposição `prop:sphere-kl`).

Hipótese comum: ell convexa com gradiente Lambda-Lipschitz em R^n (aqui, quadráticas convexas),
minimizador x* numa face própria, prescrito por KKT.

Ortante.  x_{k+1,i} = x_{k,i} (1 - (eta/2) d_i ell(x_k))^2,  eta <= 1/max{G_0, 4 Lambda R_0}.
  (a) os fatores ficam em [1/4, 9/4]: nenhum iterado sai do ortante aberto, nenhuma reflexão;
  (b) descida:        ell(x_{k+1}) <= ell(x_k) - (eta/2) D_k^2,   D^2 = sum_i x_i (d_i ell)^2;
  (c) passo em KL:    Psi_{k+1} - Psi_k <= -eta (1 - eta Lambda ||x*||_inf) Delta_k + (eta^2/4) D_k^2,
                      Psi(x) = sum_i [x_i - x*_i + x*_i log(x*_i/x_i)];
  (d) taxa:           Delta_N <= (Psi_0 + (eta/2) Delta_0) / (eta (1 - eta Lambda ||x*||_inf) N);
  (e) com o passo eta <= 1/beta_ell do Teorema `thm:orthant-no-exit`, a reflexão z -> |z| nunca atua.

Simplex, chart esférico.  x_{k+1,i} = x_{k,i} (cos th - sin th delta_i/D)^2, th = 2 eta D,
eta <= 1/(20 beta_S), beta_S = 4 Lambda + 2 osc(grad ell).
  (f) Q(x) = sum_i x*_i delta_i(x)^2 <= 4 beta_S Delta(x);
  (g) passo em KL:    KL_{k+1} - KL_k <= -2 eta Delta_k + 5 eta^2 D_k^2;
  (h) taxa:           Delta_N <= (KL(x*||x_0) + (5/2) eta Delta_0) / (2 eta N).

Uso:  python kl_lyapunov_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geogd.benchmarks import random_spd  # noqa: E402

rng = np.random.default_rng(0)
TOL = 1e-10


def gen_kl(xs, x):
    """Divergência de Kullback--Leibler generalizada (I-divergência) de x* para x > 0."""
    S = xs > 0
    return float(np.sum(x - xs) + np.sum(xs[S] * np.log(xs[S] / x[S])))


# ---------------------------------------------------------------------------
# Ortante
# ---------------------------------------------------------------------------
def orthant_instance(n: int, support: int, degenerate: bool = False):
    """Quadrática convexa com minimizador x* >= 0 de suporte prescrito (KKT)."""
    A = random_spd(rng, n)
    xs = np.zeros(n)
    S = rng.choice(n, size=support, replace=False)
    xs[S] = rng.uniform(0.5, 2.0, size=support)
    mu = rng.uniform(0.5, 2.0, size=n)
    mu[S] = 0.0
    if degenerate:                       # um multiplicador nulo fora do suporte
        Z = np.flatnonzero(xs == 0)
        mu[Z[0]] = 0.0
    xbar = xs - np.linalg.solve(A, mu)   # grad ell(x*) = A (x* - xbar) = mu
    ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
    grad = lambda x: A @ (x - xbar)
    return A, xbar, xs, ell, grad


def check_orthant(n_trials: int = 200, iters: int = 400) -> dict:
    out = dict(factor_min=np.inf, factor_max=-np.inf, descent=-np.inf, kl_step=-np.inf,
               rate=-np.inf, reflections_old_step=0, worst_ratio=0.0)
    for t in range(n_trials):
        n = int(rng.integers(3, 25))
        support = int(rng.integers(1, n))
        A, xbar, xs, ell, grad = orthant_instance(n, support, degenerate=(t % 4 == 0))
        lam = np.linalg.eigvalsh(A)
        Lam, lam_min = lam[-1], lam[0]
        x0 = rng.uniform(0.05, 3.0, size=n)
        ell_star, ell0 = ell(xs), ell(x0)
        # cotas superiores de R_0 e G_0 no subnível {ell <= ell(x_0)} (elipsoide)
        R0 = np.max(np.abs(xbar)) + np.sqrt(2 * ell0 / lam_min)
        G0 = np.sqrt(2 * ell0 * Lam)
        eta = rng.uniform(0.3, 1.0) / max(G0, 4 * Lam * R0)
        c = 1 - eta * Lam * np.max(xs)
        x = x0.copy()
        psi0, d0 = gen_kl(xs, x0), ell0 - ell_star
        for k in range(iters):
            g = grad(x)
            fac = 1 - 0.5 * eta * g
            out["factor_min"] = min(out["factor_min"], fac.min() ** 2)
            out["factor_max"] = max(out["factor_max"], fac.max() ** 2)
            D2 = np.sum(x * g**2)
            xn = x * fac**2
            dk = ell(x) - ell_star
            scale = max(1.0, abs(ell(x)))
            out["descent"] = max(out["descent"], (ell(xn) - ell(x) + 0.5 * eta * D2) / scale)
            lhs = gen_kl(xs, xn) - gen_kl(xs, x)
            out["kl_step"] = max(out["kl_step"], (lhs - (-eta * c * dk + 0.25 * eta**2 * D2)) / max(1.0, psi0))
            x = xn
            N = k + 1
            bound = (psi0 + 0.5 * eta * d0) / (eta * c * N)
            dN = ell(x) - ell_star
            out["rate"] = max(out["rate"], (dN - bound) / max(1.0, d0))
            out["worst_ratio"] = max(out["worst_ratio"], dN / bound)

        # (e) passo do Teorema thm:orthant-no-exit: eta <= 1/beta_ell, com
        #     beta_ell >= sup ( ||diag(sqrt x) A diag(sqrt x)|| + ||grad ell||_inf / 2 ) no subnível
        beta_ell = Lam * R0 + 0.5 * G0
        eta_old = 1.0 / beta_ell
        z = 2 * np.sqrt(x0)
        for k in range(iters):
            xk = z**2 / 4
            r = z - eta_old * 0.5 * z * grad(xk)
            out["reflections_old_step"] += int(np.any(r < 0))
            z = np.abs(r)
    return out


# ---------------------------------------------------------------------------
# Simplex, chart esférico
# ---------------------------------------------------------------------------
def simplex_instance(n: int, support: int, degenerate: bool = False):
    A = random_spd(rng, n)
    xs = np.zeros(n)
    S = rng.choice(n, size=support, replace=False)
    xs[S] = rng.dirichlet(np.ones(support))
    mu = rng.uniform(0.5, 2.0, size=n)
    mu[S] = 0.0
    if degenerate and support < n:
        Z = np.flatnonzero(xs == 0)
        mu[Z[0]] = 0.0
    lam = rng.normal()
    xbar = xs - np.linalg.solve(A, mu + lam)   # grad ell(x*) = lam 1 + mu
    ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
    grad = lambda x: A @ (x - xbar)
    return A, xbar, xs, ell, grad


def sphere_step(x, g, eta):
    d = g - x @ g
    D = np.sqrt(np.sum(x * d**2))
    if D == 0.0:
        return x, d, D
    th = 2 * eta * D
    xn = x * (np.cos(th) - np.sin(th) * d / D) ** 2
    return xn / xn.sum(), d, D      # renormalização: só corrige arredondamento


def check_sphere(n_trials: int = 200, iters: int = 400) -> dict:
    out = dict(Q=-np.inf, kl_step=-np.inf, rate=-np.inf, factor_min=np.inf, worst_ratio=0.0)
    for t in range(n_trials):
        n = int(rng.integers(3, 25))
        support = int(rng.integers(1, n))
        A, xbar, xs, ell, grad = simplex_instance(n, support, degenerate=(t % 4 == 0))
        Lam = np.linalg.eigvalsh(A)[-1]
        osc = max(np.ptp(grad(e)) for e in np.eye(n))    # grad ell é afim: o máximo está num vértice
        beta_S = 4 * Lam + 2 * osc
        eta = rng.uniform(0.3, 1.0) / (20 * beta_S)
        x0 = rng.dirichlet(np.ones(n) * 2.0)
        x = x0.copy()
        ell_star = ell(xs)
        psi0, d0 = gen_kl(xs, x0), ell(x0) - ell_star
        for k in range(iters):
            g = grad(x)
            xn, d, D = sphere_step(x, g, eta)
            dk = ell(x) - ell_star
            Q = np.sum(xs * d**2)
            out["Q"] = max(out["Q"], (Q - 4 * beta_S * dk) / max(1.0, beta_S * d0))
            if D > 0:
                th = 2 * eta * D
                out["factor_min"] = min(out["factor_min"], np.min(np.cos(th) - np.sin(th) * d / D))
            lhs = gen_kl(xs, xn) - gen_kl(xs, x)
            out["kl_step"] = max(out["kl_step"], (lhs - (-2 * eta * dk + 5 * eta**2 * D**2)) / max(1.0, psi0))
            x = xn
            N = k + 1
            bound = (psi0 + 2.5 * eta * d0) / (2 * eta * N)
            dN = ell(x) - ell_star
            out["rate"] = max(out["rate"], (dN - bound) / max(1.0, d0))
            out["worst_ratio"] = max(out["worst_ratio"], dN / bound)
    return out


def main() -> int:
    ok = True
    o = check_orthant()
    print("Ortante (Teorema thm:orthant-kl)")
    print(f"  (a) fatores (1 - eta g_i/2)^2 em [{o['factor_min']:.3f}, {o['factor_max']:.3f}]  (previsto: [0.25, 2.25])")
    print(f"  (b) maior violação da descida ............ {o['descent']:.2e}")
    print(f"  (c) maior violação do passo em KL ........ {o['kl_step']:.2e}")
    print(f"  (d) maior violação da taxa O(1/N) ........ {o['rate']:.2e}   (razão gap/cota máx. {o['worst_ratio']:.3f})")
    print(f"  (e) reflexões com eta = 1/beta_ell ....... {o['reflections_old_step']}")
    ok &= 0.25 - TOL <= o["factor_min"] and o["factor_max"] <= 2.25 + TOL
    ok &= o["descent"] <= TOL and o["kl_step"] <= TOL and o["rate"] <= TOL
    ok &= o["reflections_old_step"] == 0

    s = check_sphere()
    print("Simplex, chart esférico (Proposição prop:sphere-kl)")
    print(f"  (f) maior violação de Q <= 4 beta_S Delta . {s['Q']:.2e}")
    print(f"      menor fator cos(th) - sin(th) delta_i/D  {s['factor_min']:.3f}  (> 0: sem reflexão)")
    print(f"  (g) maior violação do passo em KL ........ {s['kl_step']:.2e}")
    print(f"  (h) maior violação da taxa O(1/N) ........ {s['rate']:.2e}   (razão gap/cota máx. {s['worst_ratio']:.3f})")
    ok &= s["Q"] <= TOL and s["kl_step"] <= TOL and s["rate"] <= TOL and s["factor_min"] > 0
    print("OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
