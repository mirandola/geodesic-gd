"""Verificações numéricas das identidades usadas na revisão do artigo.

(a) Identidade de transferência do ortante (Lema 5.4):
      <grad Lt(z), z-z*> = <grad ell(x), x-x*> + sum_i d_i ell(x) (sqrt x_i - sqrt x*_i)^2
(b) Contraexemplo 1-D (Observação 5.6): ell(x) = (x-1)^2/2 é fortemente convexa,
    mas a desigualdade estrelada em z* falha para x < 1/9.
(c) Identidade geral com resto de Taylor (Lema 4.7), testada com phi quadrática.
(d) Convexidade de Loewner de phi(L) = L L^T (Observação 4.6).
(e) Regra de composição: ell convexa não decrescente + phi_i convexas => L convexa
    (menor autovalor de Hess L por diferenças finitas).
(f) Termo de correção no cone PSD:  Q_L[D,D] = 2 tr(grad ell(X) D D^T).

Uso:  python check_transfer_identities.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def check_orthant_identity(n_trials: int = 20000) -> float:
    err = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(2, 9))
        x = rng.random(n) * 3 + 1e-3
        xs = rng.random(n) * 3
        g = rng.normal(size=n)  # grad ell(x) arbitrário
        z, zs = 2 * np.sqrt(x), 2 * np.sqrt(xs)
        lhs = np.dot(0.5 * z * g, z - zs)
        rhs = np.dot(g, x - xs) + np.dot(g, (np.sqrt(x) - np.sqrt(xs)) ** 2)
        err = max(err, abs(lhs - rhs))
    return err


def counterexample_1d() -> None:
    # ell(x) = (x-1)^2/2, x* = 1, z = 2 sqrt x.  h(x) = <grad Lt, z-z*> - (ell(x)-ell(x*)).
    print("  x        h(x)       forma fechada 1/2 (t-1)^2 (t+1)(3t-1), t = sqrt x")
    for x in [0.01, 0.05, 1 / 9, 0.2, 0.5, 1.0, 2.0]:
        t = np.sqrt(x)
        lhs = (2 * t) / 2 * (x - 1) * (2 * t - 2)
        h = lhs - 0.5 * (x - 1) ** 2
        print(f"  {x:<8.4f} {h:+.5f}   {0.5 * (t - 1) ** 2 * (t + 1) * (3 * t - 1):+.5f}")


def check_general_identity(n_trials: int = 4000) -> float:
    err = 0.0
    for _ in range(n_trials):
        n, m = int(rng.integers(2, 6)), int(rng.integers(2, 6))  # phi: R^m -> R^n
        Q = [0.5 * (q + q.T) for q in (rng.normal(size=(m, m)) for _ in range(n))]
        b, c = rng.normal(size=(n, m)), rng.normal(size=n)
        phi = lambda z: np.array([0.5 * z @ Q[i] @ z + b[i] @ z + c[i] for i in range(n)])
        J = lambda z: np.array([Q[i] @ z + b[i] for i in range(n)])
        z, zs, g = rng.normal(size=m), rng.normal(size=m), rng.normal(size=n)
        lhs = (J(z).T @ g) @ (z - zs)
        R = np.array([0.5 * (zs - z) @ Q[i] @ (zs - z) for i in range(n)])  # resto exato
        rhs = g @ (phi(z) - phi(zs)) + g @ R
        err = max(err, abs(lhs - rhs))
    return err


def check_loewner(n_trials: int = 20000) -> int:
    viol = 0
    for _ in range(n_trials):
        n = int(rng.integers(2, 6))
        A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        M = 0.5 * (A @ A.T + B @ B.T) - 0.25 * (A + B) @ (A + B).T
        if np.min(np.linalg.eigvalsh(M)) < -1e-10:
            viol += 1
    return viol


def composition_rule_min_eig(n_samples: int = 60) -> float:
    def one(seed):
        r = np.random.default_rng(seed)
        m, n = 4, 3
        P = [p @ p.T for p in (r.normal(size=(m, m)) for _ in range(n))]  # phi_i convexas
        w = r.random(n) + 0.1  # ell(x) = <w,x> + |x|^2/2: convexa e não decrescente em x >= 0

        def L(z):
            x = np.array([0.5 * z @ P[i] @ z for i in range(n)])
            return w @ x + 0.5 * x @ x

        z, h = r.normal(size=m), 1e-4
        H = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                ei, ej = np.zeros(m), np.zeros(m)
                ei[i], ej[j] = h, h
                H[i, j] = (L(z + ei + ej) - L(z + ei - ej) - L(z - ei + ej) + L(z - ei - ej)) / (4 * h * h)
        return np.min(np.linalg.eigvalsh(0.5 * (H + H.T)))

    return min(one(s) for s in range(n_samples))


def check_psd_correction(n_trials: int = 2000) -> float:
    err = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(2, 5))
        G = rng.normal(size=(n, n)); G = 0.5 * (G + G.T)  # grad ell(X) simétrico
        L, D = np.tril(rng.normal(size=(n, n))), np.tril(rng.normal(size=(n, n)))
        h = 1e-4
        acc = ((L + h * D) @ (L + h * D).T - 2 * L @ L.T + (L - h * D) @ (L - h * D).T) / h**2
        err = max(err, abs(np.sum(G * acc) - 2 * np.trace(G @ D @ D.T)))
    return err


if __name__ == "__main__":
    print("(a) identidade do ortante: erro máximo =", check_orthant_identity())
    print("(b) contraexemplo 1-D (negativo para x < 1/9):")
    counterexample_1d()
    print("(c) identidade geral (phi quadrática): erro máximo =", check_general_identity())
    print("(d) violações de convexidade de Loewner de LL^T:", check_loewner(), "em 20000")
    print("(e) regra de composição: menor autovalor de Hess L =", f"{composition_rule_min_eig():.3e}")
    print("(f) termo de correção no PSD vs 2 tr(G D D^T): erro máximo =", f"{check_psd_correction():.2e}")
