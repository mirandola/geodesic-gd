"""Experimentos da Observação 6.5 (lifts globais além do ortante) no simplex.

1. Carta afim.  A seleção forçada sigma_H(u) = |u| / sum|u_j| não é não
   expansiva: ao longo do GGD com reflexão no benchmark do simplex, o excesso
       e_k = ||sigma_H(r_k) - u*|| - ||r_k - u*||
   é registrado e suas somas parciais são reportadas.  Há também um
   contraexemplo explícito (u* = e_1, u = (1, eps, -eps)) que mostra
   expansão por fator sqrt(3).  ATENÇÃO: na instância reconstruída aqui o
   excesso é positivo em quase toda iteração, mas as somas parciais
   CONVERGEM (o GGD converge linearmente e o excesso vai a zero); a
   afirmação de não somabilidade feita na sessão da palestra não se
   reproduz nesta instância e foi retirada do artigo.

2. Carta esférica.  chi(x) = sqrt(x); testa-se, em instâncias aleatórias,
   (i)  ||r|| >= 1 para o passo ambiente (0-homogeneidade),
   (ii) a desigualdade de Fejér ||sigma(r) - u*|| <= ||r - u*||,
        com sigma = P_B o |.|, e sua versão geodésica d_S(|u|,u*) <= d_S(u,u*),
   (iii) a fórmula fechada do passo geodésico (eq. simplex-sphere-step) contra
        o passo Exp^S na carta,
   (iv) o contraexemplo de g-convexidade: ell(x) = x_1 vira f(u) = u_1^2, e
        f'' < 0 ao longo de um arco de círculo máximo contido no octante.

Uso:  python simplex_fejer_excess.py [--plot]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geogd.benchmarks import make_all  # noqa: E402
from geogd.charts import SimplexAffineChart, SimplexSphereChart  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Carta afim: excesso de Fejér ao longo da trajetória
# ---------------------------------------------------------------------------
def affine_chart_explicit_counterexample(eps: float = 1e-3) -> tuple[float, float]:
    """u* = e_1 (vértice), u = (1, eps, -eps) em H.  sigma_H(u) = (1, eps, eps)/(1+2 eps).
    ||u - u*|| = sqrt(2) eps,  ||sigma_H(u) - u*|| ~ sqrt(6) eps  =>  expansão por fator sqrt 3."""
    u_star = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, eps, -eps])
    s = SimplexAffineChart.select(u)
    return float(np.linalg.norm(u - u_star)), float(np.linalg.norm(s - u_star))


def affine_chart_excess(eta: float = 0.02, iters: int = 2000, seed: int = 0):
    prob = make_all(seed)["simplex"]
    C = SimplexAffineChart
    u_star = C.chi(prob.x_star)
    u = C.chi(prob.x0)
    excess = np.empty(iters)
    for k in range(iters):
        r = u - eta * prob.grad_chart(u)
        s = C.select(r)
        excess[k] = np.linalg.norm(s - u_star) - np.linalg.norm(r - u_star)
        u = s
    return excess, prob.gap(C.phi(u))


# ---------------------------------------------------------------------------
# 2. Carta esférica
# ---------------------------------------------------------------------------
def sphere_checks(n_inst: int = 40, iters: int = 600, eta: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    C = SimplexSphereChart
    fejer_viol = 0
    norm_below_one = 0
    fires = 0
    total = 0
    for _ in range(n_inst):
        n = int(rng.integers(3, 12))
        # objetivo quadrático em x com minimizador x* na fronteira do simplex
        A = rng.normal(size=(n, n)); A = A @ A.T + np.eye(n)
        x_star = np.zeros(n); k = int(rng.integers(1, n)); idx = rng.choice(n, k, replace=False)
        x_star[idx] = rng.random(k); x_star /= x_star.sum()
        grad = lambda x: A @ (x - x_star)
        u_star = np.sqrt(x_star)
        x = rng.random(n); x /= x.sum()
        u = np.sqrt(x)
        for _ in range(iters):
            # passo ambiente em u do objetivo 0-homogêneo Lhat(u) = ell(u^2/||u||^2)
            g = 2.0 * u * C.delta(u**2 / (u @ u), grad(u**2 / (u @ u))) / (u @ u)
            r = u - eta * g
            if np.linalg.norm(r) < 1.0 - 1e-12:
                norm_below_one += 1
            s = C.select(r)
            if np.linalg.norm(s - u_star) > np.linalg.norm(r - u_star) + 1e-12:
                fejer_viol += 1
            if np.any(r < 0) or np.linalg.norm(r) > 1 + 1e-12:
                fires += 1
            total += 1
            u = s
    return fejer_viol, norm_below_one, fires, total


def sphere_geodesic_fejer(n_samples: int = 200000, seed: int = 1) -> int:
    """d_S(|u|, u*) <= d_S(u, u*) para u* >= 0 na esfera."""
    rng = np.random.default_rng(seed)
    viol = 0
    for _ in range(n_samples):
        n = int(rng.integers(2, 8))
        u = rng.normal(size=n); u /= np.linalg.norm(u)
        us = np.abs(rng.normal(size=n)); us /= np.linalg.norm(us)
        d = lambda a, b: np.arccos(np.clip(a @ b, -1, 1))
        if d(np.abs(u), us) > d(u, us) + 1e-12:
            viol += 1
    return viol


def sphere_closed_form_check(n_trials: int = 5000, seed: int = 2) -> float:
    """Fórmula fechada x_{k+1,i} = x_i (cos th - sin th delta_i/D)^2 vs Exp^S na carta."""
    rng = np.random.default_rng(seed)
    C = SimplexSphereChart
    err = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(2, 9))
        x = rng.random(n); x /= x.sum()
        g = rng.normal(size=n)
        eta = rng.uniform(0.01, 0.5)
        u = np.sqrt(x)
        v = -eta * C.grad_L(u, g)
        x_exp = C.exp_sphere(u, v) ** 2
        x_cf = C.geodesic_step(x, g, eta)
        err = max(err, np.max(np.abs(x_exp - x_cf)), abs(x_cf.sum() - 1.0))
    return err


def gconvexity_counterexample() -> tuple[float, float]:
    """ell(x) = x_1 => f(u) = u_1^2 ao longo de u(t) = cos t e_1 + sin t b (b >= 0, b _|_ e_1)."""
    n = 3
    b = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    t = np.linspace(0.05, np.pi / 4 - 0.05, 200)
    u = np.outer(np.cos(t), np.eye(n)[0]) + np.outer(np.sin(t), b)
    assert np.all(u > 0)  # arco inteiro no octante positivo
    f = u[:, 0] ** 2  # = cos^2 t
    f2 = np.gradient(np.gradient(f, t), t)
    return float(f2.max()), float(f2.min())


if __name__ == "__main__":
    print("== 1. Carta afim ==")
    d0, d1 = affine_chart_explicit_counterexample()
    print(f"  contraexemplo explícito: ||u-u*|| = {d0:.3e},  ||sigma_H(u)-u*|| = {d1:.3e}  (razão {d1/d0:.3f} ~ sqrt 3)")
    print("  excesso de Fejér do GGD com reflexão sigma_H ao longo do benchmark:")
    ex, gap = affine_chart_excess()
    cs = np.cumsum(ex)
    print(f"  excesso > 0 em {np.sum(ex > 0)} de {len(ex)} iterações; gap final = {gap:.2e}")
    for k in (100, 500, 1000, 2000):
        print(f"  soma parcial ate k={k:<5d}: {cs[k - 1]:+.3f}")
    print("  (compare o crescimento com sqrt(k):", [f"{cs[k-1]/np.sqrt(k):.3f}" for k in (100, 500, 1000, 2000)], ")")

    print("\n== 2. Carta esférica ==")
    viol, below, fires, tot = sphere_checks()
    print(f"  violações de Fejér: {viol} em {tot};  ||r||<1: {below};  reflexão/retração acionada: {fires} ({100*fires/tot:.1f}%)")
    print(f"  violações de Fejér geodésico d_S(|u|,u*) <= d_S(u,u*): {sphere_geodesic_fejer()} em 200000")
    print(f"  fórmula fechada vs Exp^S: erro máximo = {sphere_closed_form_check():.2e}")
    fmax, fmin = gconvexity_counterexample()
    print(f"  ell(x)=x_1 na carta esférica: f'' ao longo do arco em [{fmin:.3f}, {fmax:.3f}]  (negativo => não g-convexa)")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        out = Path(__file__).resolve().parents[2] / "figures" / "simplex_affine_fejer_excess.pdf"
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(cs, color="#EB811B"); ax.set_xlabel("k"); ax.set_ylabel(r"$\sum_{j \leq k} e_j$")
        ax.set_title("Affine simplex chart: cumulative Fejér excess"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out); print("  figura salva em", out)
