"""Verificações numéricas da aceleração no ortante (Teorema `thm:orthant-local-accel` e
Observação `rem:orthant-accel-global`).

Objetivo: ell quadrática convexa, minimizador x* numa face própria prescrito por KKT;
levantamento L(z) = ell(z^2/4), z* = 2 sqrt(x*).

(a) Hessiana do levantamento em z*: bloco-diagonal,
        Hess L(z*) = diag( D_S Hess_SS ell(x*) D_S ,  (1/2) diag(d_Z ell(x*)) ),  D_S = diag(sqrt(x*_S)),
    logo lambda_min >= mu_L = min{ mu min_S x*_i , gamma/2 } > 0 sob complementaridade estrita.
(b) Desigualdade de potencial de um passo (forma local), para o método acelerado com momento
    constante q = (sqrt(kappa)-1)/(sqrt(kappa)+1):
        Phi_{k+1} <= (1 - 1/sqrt(kappa)) Phi_k,   Phi_k = L(z_k) - L* + (mu/2) ||v_k - z*||^2,
    e confinamento: partindo de ||z_0 - z*|| < r / (2 sqrt(1+kappa)), nenhum iterado sai de B(z*, r).
(c) Comportamento global do AGGD (momento de Beck--Teboulle, sem reinício, passo 1/beta):
    a cota 4 beta D_chi(x*, x_0) / (N+1)^2 da teoria convexa vale empiricamente, embora L não
    seja convexa e os iterados troquem de sinal.  Evidência, não teorema.

Uso:  python local_acceleration_checks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kl_lyapunov_checks as K  # noqa: E402

rng = np.random.default_rng(5)


def lift(A, xbar):
    g = lambda x: A @ (x - xbar)
    L = lambda z: 0.5 * (z * z / 4 - xbar) @ A @ (z * z / 4 - xbar)
    gL = lambda z: 0.5 * z * g(z * z / 4)
    HL = lambda z: 0.25 * (z[:, None] * A * z[None, :]) + 0.5 * np.diag(g(z * z / 4))
    return g, L, gL, HL


def check_hessian(n_trials: int = 300) -> tuple[float, float]:
    err, slack = 0.0, np.inf
    for _ in range(n_trials):
        n = int(rng.integers(3, 15)); sup = int(rng.integers(1, n))
        A, xbar, xs, _, _ = K.orthant_instance(n, sup)
        g, _, _, HL = lift(A, xbar)
        S = xs > 0
        lam = np.linalg.eigvalsh(HL(2 * np.sqrt(xs)))[0]
        Dh = np.sqrt(xs[S])
        block = np.linalg.eigvalsh(Dh[:, None] * A[np.ix_(S, S)] * Dh[None, :])[0]
        gam = g(xs)[~S].min()
        mu = np.linalg.eigvalsh(A[np.ix_(S, S)])[0]
        err = max(err, abs(lam - min(block, gam / 2)))
        slack = min(slack, lam - min(mu * xs[S].min(), gam / 2))
    return err, slack


def check_potential(n_trials: int = 300, iters: int = 300) -> tuple[float, int, float]:
    viol, exits, rate = -np.inf, 0, -np.inf
    for _ in range(n_trials):
        n = int(rng.integers(3, 15)); sup = int(rng.integers(1, n))
        A, xbar, xs, _, _ = K.orthant_instance(n, sup)
        _, L, gL, HL = lift(A, xbar)
        zs = 2 * np.sqrt(xs)
        muL = np.linalg.eigvalsh(HL(zs))[0]

        def ball_ok(r):
            for _ in range(200):
                d = rng.normal(size=n); d *= r * rng.random() ** (1 / n) / np.linalg.norm(d)
                if np.linalg.eigvalsh(HL(zs + d))[0] < 0.5 * muL:
                    return False
            return True

        r = 1.0
        while not ball_ok(r):
            r *= 0.5
        r *= 0.5                                     # margem: o teste acima é por amostragem
        beta = 1.05 * max(
            np.linalg.eigvalsh(HL(zs + r * d / np.linalg.norm(d)))[-1] for d in rng.normal(size=(300, n))
        )
        mu = 0.5 * muL
        kap = beta / mu
        al = 1 / math.sqrt(kap)
        q = (math.sqrt(kap) - 1) / (math.sqrt(kap) + 1)
        d = rng.normal(size=n)
        z = zs + 0.99 * r / (2 * math.sqrt(1 + kap)) * d / np.linalg.norm(d)
        v, y, Ls = z.copy(), z.copy(), L(zs)
        Phi = Phi0 = L(z) - Ls + 0.5 * mu * np.sum((z - zs) ** 2)
        for k in range(iters):
            gy = gL(y)
            zn = y - gy / beta
            vn = (1 - al) * v + al * y - (al / mu) * gy
            exits += int(max(np.linalg.norm(zn - zs), np.linalg.norm(y - zs)) >= r)
            Phin = L(zn) - Ls + 0.5 * mu * np.sum((vn - zs) ** 2)
            viol = max(viol, (Phin - (1 - al) * Phi) / Phi0)
            rate = max(rate, (L(zn) - Ls - (1 - al) ** (k + 1) * Phi0) / Phi0)
            y = zn + q * (zn - z)
            z, v, Phi = zn, vn, Phin
    return viol, exits, rate


def check_global(n_trials: int = 300, iters: int = 3000) -> dict:
    ratios, worse, sign = [], 0, 0
    for t in range(n_trials):
        n = int(rng.integers(3, 25)); sup = int(rng.integers(1, n))
        A, xbar, xs, ell, _ = K.orthant_instance(n, sup, degenerate=(t % 3 == 0))
        _, L, gL, _ = lift(A, xbar)
        zs = 2 * np.sqrt(xs)
        lam = np.linalg.eigvalsh(A)
        x0 = rng.uniform(0.05, 3.0, size=n)
        ell0 = ell(x0)
        R0 = np.max(np.abs(xbar)) + math.sqrt(2 * ell0 / lam[0])
        G0 = math.sqrt(2 * ell0 * lam[-1])
        beta = lam[-1] * R0 + 0.5 * G0                # cota superior de beta_ell no subnível
        z = 2 * np.sqrt(x0); y = z.copy(); zg = z.copy(); tk = 1.0
        Dchi, Ls, worst, changed = 0.5 * np.sum((z - zs) ** 2), L(zs), 0.0, False
        for k in range(iters):
            zn = y - gL(y) / beta
            tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
            y = zn + ((tk - 1) / tn) * (zn - z)
            z, tk = zn, tn
            changed |= bool(np.min(z) < 0)
            worst = max(worst, (L(z) - Ls) * (k + 2) ** 2 / (4 * beta * Dchi))
            zg = zg - gL(zg) / beta
        ratios.append(worst)
        worse += int(L(z) - Ls > L(zg) - Ls + 1e-12)
        sign += int(changed)
    return dict(max_ratio=max(ratios), median_ratio=float(np.median(ratios)), worse_than_ggd=worse,
                sign_changes=sign, n=n_trials)


def main() -> int:
    err, slack = check_hessian()
    print("(a) Hessiana do levantamento em z*")
    print(f"    erro máx. da fórmula por blocos ......... {err:.2e}")
    print(f"    folga mín. de lambda_min - mu_L ......... {slack:.2e}  (>= 0)")
    viol, exits, rate = check_potential()
    print("(b) Potencial local e confinamento")
    print(f"    maior violação de Phi_(k+1) <= (1-a)Phi_k  {viol:.2e}")
    print(f"    maior violação da taxa acelerada ........ {rate:.2e}")
    print(f"    iterados fora de B(z*, r) ............... {exits}")
    g = check_global()
    print("(c) AGGD global, passo 1/beta (evidência)")
    print(f"    máx. de gap (N+1)^2 / (4 beta D_chi) .... {g['max_ratio']:.3f}  (mediana {g['median_ratio']:.3f})")
    print(f"    instâncias com troca de sinal em z ...... {g['sign_changes']} de {g['n']}")
    print(f"    AGGD pior que GGD ao final .............. {g['worse_than_ggd']} de {g['n']}")
    ok = err < 1e-10 and slack >= -1e-10 and viol <= 1e-8 and rate <= 1e-8 and exits == 0
    print("OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
