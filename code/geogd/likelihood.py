"""Família C ("likelihood"): benchmarks de máxima verossimilhança, não construídos na carta.

Ao contrário das famílias A (quadráticas na carta) e B (quadráticas euclidianas com
minimizador prescrito por KKT), aqui o objetivo é uma log-verossimilhança padrão e o
minimizador NÃO é conhecido por construção: é calculado por um método de Newton com
conjunto ativo e certificado por um limitante de dualidade (ver `reference_*`).

    simplex  NPMLE de uma mistura de locação gaussiana numa grade (Kiefer--Wolfowitz):
             ell(x) = -(1/m) sum_j log <a_j, x>,   a_ji = N(y_j; mu_i, 1),  x no simplex.
             O minimizador é esparso (poucos átomos).  Baseline multiplicativa clássica: EM.
    orthant  problema inverso linear de Poisson:
             ell(x) = (1/m) sum_j ( <a_j, x> - b_j log <a_j, x> ),  A >= 0,  x >= 0,
             b ~ Poisson(A x_true), x_true esparso.  Baseline: Richardson--Lucy (EM).
             Dois desenhos: "blur" (deconvolução 1-D, A de banda; grad ell ilimitado na
             fronteira) e "dense" (controle, A denso; grad ell limitado).
    psd      máxima verossimilhança gaussiana (matriz de precisão):
             ell(X) = tr(S X) - log det X,  minimizador X* = S^{-1} (interior).
             Na carta de Cholesky o objetivo levantado é convexo (Obs. 6.4 do artigo).

Todos os métodos recebem o gradiente exato; os de carta usam a regra da cadeia (Lema 2.3).
O valor de ell é +inf fora do domínio (algum <a_j, x> <= 0, ou X não definida positiva), o
que marca a corrida como instável no protocolo de ajuste de passo.
"""

from __future__ import annotations

import numpy as np

from .benchmarks import Problem, _chain_grad_chart, project_psd, project_simplex, random_spd
from .charts import CholeskyChart, OrthantChart, SimplexAffineChart

FAMILY = "likelihood"


def _safe(project):
    """Projeção que devolve NaN (corrida instável) em vez de falhar quando a entrada é não
    finita ou tão grande que o algoritmo de ordenação perde precisão."""
    def wrapped(v):
        if not np.all(np.isfinite(v)):
            return np.full_like(v, np.nan)
        try:
            return project(v)
        except IndexError:
            return np.full_like(v, np.nan)
    return wrapped


# ---------------------------------------------------------------------------
# Soluções de referência (Newton com conjunto ativo + certificado de dualidade)
# ---------------------------------------------------------------------------
def _newton_active_set(A: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray,
                       outer: int = 400, inner: int = 200) -> np.ndarray:
    """min  f(x) = <c,x> - (1/m) sum_j b_j log <a_j,x>   sobre x >= 0   (A >= 0, c > 0).

    Laço interno: Newton na face {x_i = 0, i fora de `free`} com busca de Armijo e teste da
    razão (variáveis que atingem a fronteira saem de `free`).  Laço externo: ao convergir na
    face, libera o índice de multiplicador mais negativo (um por vez, sem ciclagem)."""
    m = A.shape[0]
    pos = b > 0                                      # linhas com b_j = 0 só entram no termo linear <c,x>
    A, b = A[pos], b[pos]
    f = lambda v: float(c @ v - np.sum(b * np.log(A @ v)) / m)
    grad = lambda v: c - A.T @ (b / (A @ v)) / m
    scale = x.max()
    free = x > 1e-9 * scale
    x = np.where(free, x, 0.0)
    for _ in range(outer):
        for _ in range(inner):
            idx = np.flatnonzero(free)
            r = A @ x
            g = grad(x)[idx]
            Af = A[:, idx] * (np.sqrt(b) / r)[:, None]
            H = Af.T @ Af / m
            d = np.linalg.lstsq(H, -g, rcond=None)[0]
            if g @ d >= 0:                               # direção degenerada: usa o gradiente
                d = -g
            dec = -(g @ d)                               # decremento de Newton (>= 0)
            if dec <= 1e-30:
                break
            neg = d < 0
            t_max = np.min(-x[idx][neg] / d[neg]) if np.any(neg) else np.inf
            t = min(1.0, t_max)
            f0 = f(x)
            pure = dec <= 1e-9 * max(1.0, abs(f0))       # perto do ótimo f não resolve: Newton puro
            xn = x
            while t > 1e-16:
                xn = x.copy(); xn[idx] = np.maximum(x[idx] + t * d, 0.0)
                if np.all(A @ xn > 0) and (pure or f(xn) <= f0 - 1e-4 * t * dec):
                    break
                t *= 0.5
            else:
                break                                    # sem progresso possível nesta face
            hit = free & (xn <= 1e-14 * scale)
            xn[hit] = 0.0
            step = np.linalg.norm(xn - x)
            x = xn
            free &= ~hit
            if step <= 1e-15 * scale and not np.any(hit):
                break
        g_all = grad(x)
        cand = np.where(~free, g_all, np.inf)
        j = int(np.argmin(cand))
        if cand[j] >= -1e-14:
            break
        free[j] = True
        x[j] = 1e-9 * scale
    return x


def reference_simplex_loglik(A: np.ndarray, warm_iters: int = 3000):
    """min -(1/m) sum_j log <a_j,x> no simplex.  Devolve (x*, certificado), em que
    certificado = max_i c_i(x*) - 1 >= ell(x*) - min ell  (gap de Frank--Wolfe), com
    c_i(x) = (1/m) sum_j a_ji / <a_j,x> = -d_i ell(x)  (KKT: c_i <= 1, com igualdade no suporte).
    Usa a forma relaxada  min_{x>=0} sum_i x_i - (1/m) sum_j log <a_j,x>,  cujo minimizador
    tem soma 1 e coincide com o do problema no simplex."""
    m, n = A.shape
    x = np.full(n, 1.0 / n)
    for _ in range(warm_iters):                      # EM: monótono, sem passo
        x = x * (A.T @ (1.0 / (A @ x))) / m
    x = _newton_active_set(A, np.ones(m), np.ones(n), x)
    x = x / x.sum()
    cvec = (A.T @ (1.0 / (A @ x))) / m
    return x, float(cvec.max() - 1.0)


def reference_orthant_poisson(A: np.ndarray, b: np.ndarray, warm_iters: int = 3000):
    """min (1/m) sum_j (<a_j,x> - b_j log <a_j,x>) em x >= 0.  Devolve (x*, certificado), com
    certificado = |<g,x*>| + max(-min_i g_i, 0) * sum(b)/min_i (A^T 1)_i  >= ell(x*) - min ell,
    g = grad ell(x*)  (usa sum_j <a_j,x_opt> = sum_j b_j em qualquer minimizador)."""
    m, n = A.shape
    col = A.sum(axis=0)
    x = np.full(n, b.sum() / col.sum())
    pos = b > 0
    Ap, bp = A[pos], b[pos]
    for _ in range(warm_iters):                      # Richardson--Lucy
        x = x * (Ap.T @ (bp / (Ap @ x))) / col
    x = _newton_active_set(A, b, col / m, x)
    g = (col - Ap.T @ (bp / (Ap @ x))) / m
    cert = float(abs(g @ x) + max(-g.min(), 0.0) * b.sum() / col.min())
    return x, cert


# ---------------------------------------------------------------------------
# Simplex: NPMLE de mistura de locação gaussiana numa grade
# ---------------------------------------------------------------------------
def make_simplex_mixture(rng: np.random.Generator, n: int = 60, m: int = 400) -> Problem:
    """Amostras y_j de uma mistura de três gaussianas N(mu,1); componentes candidatas numa
    grade uniforme de n pontos cobrindo a amostra.  a_ji = densidade N(y_j; grade_i, 1)."""
    centers = np.array([-3.0, 0.0, 4.0]); weights = np.array([0.3, 0.5, 0.2])
    comp = rng.choice(3, size=m, p=weights)
    y = centers[comp] + rng.normal(size=m)
    grid = np.linspace(y.min(), y.max(), n)
    A = np.exp(-0.5 * (y[:, None] - grid[None, :]) ** 2) / np.sqrt(2.0 * np.pi)
    x_star, cert = reference_simplex_loglik(A)
    support = x_star > 0

    def ell(x):
        r = A @ x
        return float(-np.mean(np.log(r))) if np.all(r > 0) else np.inf

    def grad_x(x):
        return -(A.T @ (1.0 / (A @ x))) / m

    def mirror(x, g, eta):
        y_ = x * np.exp(-eta * (g - g.min()))
        return y_ / y_.sum()

    def mult(x):                                      # EM (ponto fixo de Cover)
        return x * (A.T @ (1.0 / (A @ x))) / m

    def leakage(x):
        return float(np.sum(x[~support]))

    x0 = rng.uniform(0.5, 1.5, size=n); x0 /= x0.sum()
    prob = Problem(
        name="simplex-mixture", ell=ell, grad_x=grad_x,
        grad_chart=_chain_grad_chart(grad_x, SimplexAffineChart),
        project=_safe(project_simplex), mirror=mirror, chart=SimplexAffineChart,
        x_star=x_star, f_star=ell(x_star), x0=x0, leakage=leakage,
        step_grid=np.logspace(-3, 2, 31), family=FAMILY,
        grad_w=lambda w: 2.0 * w * grad_x(w**2),
    )
    prob.mult = mult
    prob.certificate = cert
    # complementaridade estrita: 1 - c_i(x*) > 0 fora do suporte
    c = -grad_x(x_star)
    prob.info = {"n": n, "m": m, "support": int(support.sum()), "certificate": cert,
                 "strict_compl_margin": float(np.min(1.0 - c[~support])) if np.any(~support) else None,
                 "min_support_mass": float(x_star[support].min())}
    return prob


# ---------------------------------------------------------------------------
# Ortante: problema inverso linear de Poisson
# ---------------------------------------------------------------------------
def _blur_matrix(n: int, width: int, sigma: float) -> np.ndarray:
    """Desfocagem gaussiana truncada (matriz de banda, núcleo normalizado)."""
    offs = np.arange(-width, width + 1)
    k = np.exp(-0.5 * (offs / sigma) ** 2); k /= k.sum()
    A = np.zeros((n, n))
    for t, kv in zip(offs, k):
        i = np.arange(max(0, -t), min(n, n - t))
        A[i, i + t] = kv
    return A


def make_orthant_poisson(rng: np.random.Generator, design: str = "blur") -> Problem:
    """b ~ Poisson(A x_true), x_true >= 0 esparso; ell(x) = (1/m) sum_j (<a_j,x> - b_j log <a_j,x>).

    design="blur"  (padrão): deconvolução 1-D, n = m = 64, A = desfocagem gaussiana truncada
                   (banda), x_true = 5 picos.  É o cenário clássico de Richardson--Lucy.  Há
                   linhas com b_j = 0 e grad ell é ilimitado onde <a_j,x> -> 0 com b_j > 0.
    design="dense" (controle): A denso com entradas uniformes em (0,1), n = 40, m = 120,
                   x_true com 6 entradas não nulas; <a_j,x> > 0 em todo x >= 0 não nulo, logo
                   grad ell é limitado perto do minimizador."""
    if design == "blur":
        n = m = 64
        A = _blur_matrix(n, width=5, sigma=1.5)
        x_true = np.zeros(n)
        idx = rng.choice(n, size=5, replace=False)
        x_true[idx] = rng.uniform(50.0, 200.0, size=5)
    elif design == "dense":
        n, m = 40, 120
        A = rng.uniform(0.0, 1.0, size=(m, n))
        x_true = np.zeros(n)
        idx = rng.choice(n, size=6, replace=False)
        x_true[idx] = rng.uniform(5.0, 20.0, size=6)
    else:
        raise ValueError(design)
    b = rng.poisson(A @ x_true).astype(float)
    col = A.sum(axis=0)
    pos = b > 0
    Ap, bp = A[pos], b[pos]
    x_star, cert = reference_orthant_poisson(A, b)
    supp = x_star > 0

    def ell(x):
        r = Ap @ x
        if np.any(r <= 0) or np.any(x < 0) or not np.all(np.isfinite(x)):
            return np.inf
        return float((col @ x - np.sum(bp * np.log(r))) / m)

    def grad_x(x):
        return (col - Ap.T @ (bp / (Ap @ x))) / m

    def mirror(x, g, eta):
        return x * np.exp(-eta * g)

    def mult(x):                                      # Richardson--Lucy
        return x * (Ap.T @ (bp / (Ap @ x))) / col

    def leakage(x):
        return float(np.sum(x[~supp]))

    x0 = rng.uniform(0.5, 1.5, size=n)
    x0 *= b.sum() / (A @ x0).sum()                    # mesma contagem total dos dados
    prob = Problem(
        name=f"orthant-poisson-{design}", ell=ell, grad_x=grad_x,
        grad_chart=_chain_grad_chart(grad_x, OrthantChart),
        project=lambda v: np.maximum(v, 0.0), mirror=mirror, chart=OrthantChart,
        x_star=x_star, f_star=ell(x_star), x0=x0, leakage=leakage,
        step_grid=np.logspace(-2, 4, 37) if design == "blur" else np.logspace(-3, 2, 31), family=FAMILY,
    )
    prob.mult = mult
    prob.certificate = cert
    g = grad_x(x_star)
    prob.info = {"design": design, "n": n, "m": m, "support": int(supp.sum()), "support_true": int(len(idx)),
                 "rows_with_zero_count": int(np.sum(~pos)), "certificate": cert,
                 "strict_compl_margin": float(np.min(g[~supp])) if np.any(~supp) else None,
                 "min_support_mass": float(x_star[supp].min()), "max_mass": float(x_star.max())}
    return prob


# ---------------------------------------------------------------------------
# PSD: máxima verossimilhança gaussiana  ell(X) = tr(S X) - log det X
# ---------------------------------------------------------------------------
def make_psd_gaussian(rng: np.random.Generator, n: int = 5, cond: float = 100.0) -> Problem:
    """S SPD com número de condição `cond`; minimizador interior X* = S^{-1}."""
    S = random_spd(rng, n, cond=cond)
    X_star = np.linalg.inv(S)

    def ell(X):
        Xs = 0.5 * (X + X.T)
        sign, logdet = np.linalg.slogdet(Xs)
        return float(np.sum(S * Xs) - logdet) if sign > 0 and np.all(np.linalg.eigvalsh(Xs) > 0) else np.inf

    def grad_x(X):
        return S - np.linalg.inv(0.5 * (X + X.T))

    def mirror(X, g, eta):                            # potencial de Burg
        Y = np.linalg.inv(X) + eta * 0.5 * (g + g.T)
        return np.linalg.inv(project_psd(Y, 1e-12))

    def leakage(X):                                   # minimizador interior: erro relativo
        return float(np.linalg.norm(0.5 * (X + X.T) - X_star) / np.linalg.norm(X_star))

    L0 = np.tril(rng.normal(size=(n, n)))
    L0[np.diag_indices(n)] = rng.uniform(0.8, 1.5, size=n)
    X0 = L0 @ L0.T
    prob = Problem(
        name="psd-gaussian", ell=ell, grad_x=grad_x,
        grad_chart=_chain_grad_chart(grad_x, CholeskyChart),
        project=lambda Y: project_psd(Y, 1e-10), mirror=mirror, chart=CholeskyChart,
        x_star=X_star, f_star=ell(X_star), x0=X0, leakage=leakage,
        step_grid=np.logspace(-6, 1, 43), family=FAMILY,
    )
    prob.mult = None
    prob.certificate = 0.0
    prob.info = {"n": n, "cond_S": cond, "interior_minimizer": True}
    return prob


def make_all_likelihood(seed: int = 0) -> dict[str, Problem]:
    """Família C (verossimilhanças).  Cada problema usa um gerador próprio, de modo que
    alterar um construtor não muda as instâncias dos demais."""
    rngs = [np.random.default_rng([seed, 2000, k]) for k in range(4)]
    return {
        "orthant": make_orthant_poisson(rngs[0], design="blur"),
        "simplex": make_simplex_mixture(rngs[1]),
        "psd": make_psd_gaussian(rngs[2]),
        "orthant-dense": make_orthant_poisson(rngs[3], design="dense"),
    }
