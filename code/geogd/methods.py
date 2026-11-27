"""Os seis métodos comparados na Seção 7 do artigo.

Todos retornam um dicionário com as trajetórias `gap` (objetivo - ótimo),
`leak` (vazamento de fronteira) e o iterado final `x`.  Um resultado é
marcado `stable=False` se produziu NaN/inf ou se o gap final excede o inicial.

Convenções:
  * PGD/APGD trabalham em x e usam `problem.project`.
  * MD/AMD trabalham em x e usam `problem.mirror` (entropia ou Burg).
  * GGD/AGGD trabalham na carta: z <- z - eta grad_chart(z), x = phi(z).
    Com `reflect=True`, aplica-se a seleção canônica na fibra z <- sigma(z)
    após cada passo (mecanismo global do ortante, Teorema 5.5(i)).
"""

from __future__ import annotations

import numpy as np

from .benchmarks import Problem


def _finish(problem: Problem, x: np.ndarray, gaps: list, leaks: list) -> dict:
    gaps = np.asarray(gaps, dtype=float)
    leaks = np.asarray(leaks, dtype=float)
    stable = bool(np.all(np.isfinite(gaps)) and gaps[-1] <= gaps[0] * (1.0 + 1e-9))
    return {"gap": gaps, "leak": leaks, "x": x, "stable": stable}


def _record(problem: Problem, x, gaps, leaks):
    gaps.append(problem.gap(x))
    leaks.append(problem.leakage(x))


# ---------------------------------------------------------------------------
# Projetados
# ---------------------------------------------------------------------------
def pgd(problem: Problem, eta: float, iters: int) -> dict:
    x = problem.x0.copy()
    gaps, leaks = [], []
    _record(problem, x, gaps, leaks)
    for _ in range(iters):
        x = problem.project(x - eta * problem.grad_x(x))
        _record(problem, x, gaps, leaks)
    return _finish(problem, x, gaps, leaks)


def apgd(problem: Problem, eta: float, iters: int) -> dict:
    """Nesterov (FISTA) com projeção; o ponto extrapolado também é projetado,
    porque ell só está definido no fecho do conjunto viável."""
    x = problem.x0.copy()
    y = x.copy()
    t = 1.0
    gaps, leaks = [], []
    _record(problem, x, gaps, leaks)
    for _ in range(iters):
        x_new = problem.project(y - eta * problem.grad_x(y))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = problem.project(x_new + ((t - 1.0) / t_new) * (x_new - x))
        x, t = x_new, t_new
        _record(problem, x, gaps, leaks)
    return _finish(problem, x, gaps, leaks)


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------
def md(problem: Problem, eta: float, iters: int) -> dict:
    x = problem.x0.copy()
    gaps, leaks = [], []
    _record(problem, x, gaps, leaks)
    for _ in range(iters):
        x = problem.mirror(x, problem.grad_x(x), eta)
        _record(problem, x, gaps, leaks)
    return _finish(problem, x, gaps, leaks)


def amd(problem: Problem, eta: float, iters: int, r: float = 3.0) -> dict:
    """Mirror descent acelerado (Krichene--Bayen--Bartlett 2015), versão
    discreta por médias:  lambda_k = r/(k+r),
        x_k      = lambda_k z_k + (1-lambda_k) xt_k
        z_{k+1}  = passo de mirror a partir de z_k com passo eta (k+1)/r
        xt_{k+1} = lambda_k z_{k+1} + (1-lambda_k) xt_k .
    Revisar: o artigo original de Krichene et al. também propõe uma variante
    com passo corretor; esta implementação usa apenas a média."""
    z = problem.x0.copy()
    xt = problem.x0.copy()
    gaps, leaks = [], []
    _record(problem, xt, gaps, leaks)
    for k in range(iters):
        lam = r / (k + r)
        xk = lam * z + (1.0 - lam) * xt
        g = problem.grad_x(xk)
        z = problem.mirror(z, g, eta * (k + 1) / r)
        xt = lam * z + (1.0 - lam) * xt
        _record(problem, xt, gaps, leaks)
    return _finish(problem, xt, gaps, leaks)


# ---------------------------------------------------------------------------
# Geodésicos (na carta)
# ---------------------------------------------------------------------------
def ggd(problem: Problem, eta: float, iters: int, reflect: bool = False) -> dict:
    C = problem.chart
    z = C.chi(problem.x0)
    x = C.phi(z)
    gaps, leaks = [], []
    _record(problem, x, gaps, leaks)
    for _ in range(iters):
        z = z - eta * problem.grad_chart(z)
        if reflect:
            z = C.select(z)
        x = C.phi(z)
        _record(problem, x, gaps, leaks)
    return _finish(problem, x, gaps, leaks)


def aggd(problem: Problem, eta: float, iters: int, reflect: bool = False) -> dict:
    """Nesterov na carta (Algoritmo 3.2 com 1/beta substituído por eta)."""
    C = problem.chart
    z = C.chi(problem.x0)
    y = z.copy()
    t = 1.0
    x = C.phi(z)
    gaps, leaks = [], []
    _record(problem, x, gaps, leaks)
    for _ in range(iters):
        z_new = y - eta * problem.grad_chart(y)
        if reflect:
            z_new = C.select(z_new)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = z_new + ((t - 1.0) / t_new) * (z_new - z)
        z, t = z_new, t_new
        x = C.phi(z)
        _record(problem, x, gaps, leaks)
    return _finish(problem, x, gaps, leaks)


def ggd_sphere(problem: Problem, eta: float, iters: int) -> dict:
    """GGD na carta esférica do simplex (Obs. 6.5, eq. simplex-sphere-step):
    w = sqrt(x) na esfera unitária, passo ao longo do círculo máximo.
    Só está definido para problemas com `grad_w`."""
    from .charts import SimplexSphereChart as S
    if problem.grad_w is None:
        raise ValueError("ggd_sphere requer problem.grad_w (benchmarks do simplex)")
    w = np.sqrt(problem.x0)
    x = w**2
    gaps, leaks = [], []
    _record(problem, x, gaps, leaks)
    for _ in range(iters):
        w = S.step_from_grad_w(w, problem.grad_w(w), eta)
        w = np.abs(w)                # reflexão embutida (phi(|w|) = phi(w))
        x = w**2
        _record(problem, x, gaps, leaks)
    return _finish(problem, x, gaps, leaks)


METHODS = {
    "PGD": pgd, "APGD": apgd,
    "MD": md, "AMD": amd,
    "GGD": ggd, "AGGD": aggd,
}
EXTRA_METHODS = {"GGD-S": ggd_sphere}  # só no simplex


TOL_HIT = 1e-6                 # tolerância das tabelas, usada no desempate
TIE_REL, TIE_ABS = 1e-3, 1e-12  # gaps finais que coincidem a 0,1% ou abaixo de 1e-12 são empates


def first_hit(gap: np.ndarray, tol: float = TOL_HIT):
    """Primeira iteração com gap < tol (None se não ocorre)."""
    idx = np.flatnonzero(np.asarray(gap) < tol)
    return int(idx[0]) if idx.size else None


def tune_and_run(problem: Problem, method: str, iters: int = 150, grid=None) -> tuple[dict, float]:
    """Varre a grade logarítmica de passos e devolve a corrida estável com o
    menor gap final (protocolo da Subseção 7.2).

    Desempate.  Vários passos costumam terminar no mesmo piso numérico (gap final exatamente 0,
    ou igual a menos de arredondamento); escolher entre eles pelo menor valor em ponto flutuante
    faria o passo escolhido, e portanto a "primeira iteração abaixo de 1e-6", depender do ruído
    de arredondamento e do ambiente.  Por isso gaps finais que coincidem a menos de TIE_REL
    (relativo) ou que estão abaixo de TIE_ABS contam como iguais, e entre eles fica a corrida
    que atinge TOL_HIT mais cedo (depois: menor gap, depois: maior passo)."""
    grid = problem.step_grid if grid is None else grid
    runs = []
    for eta in grid:
        with np.errstate(all="ignore"):
            try:
                out = {**METHODS, **EXTRA_METHODS}[method](problem, float(eta), iters)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
        if out["stable"]:
            runs.append((out, float(eta)))
    if not runs:  # nenhuma corrida estável: devolve a do menor passo
        return {**METHODS, **EXTRA_METHODS}[method](problem, float(grid[0]), iters), float(grid[0])
    final = lambda out: max(float(out["gap"][-1]), 0.0)
    threshold = min(final(o) for o, _ in runs) * (1.0 + TIE_REL) + TIE_ABS
    tied = [(o, e) for o, e in runs if final(o) <= threshold]

    def key(run):
        out, eta = run
        hit = first_hit(out["gap"])
        return (hit if hit is not None else np.inf, final(out), -eta)

    return min(tied, key=key)
