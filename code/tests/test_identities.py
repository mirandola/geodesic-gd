"""Testes das identidades matemáticas usadas no artigo (pytest)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geogd.benchmarks import make_all  # noqa: E402
from geogd.charts import CholeskyChart, OrthantChart, SimplexAffineChart, SimplexSphereChart  # noqa: E402
from geogd.methods import ggd, ggd_sphere  # noqa: E402

rng = np.random.default_rng(123)


def test_orthant_transfer_identity():
    # <grad Lt(z), z-z*> = <grad ell(x), x-x*> + sum d_i ell (sqrt x_i - sqrt x*_i)^2
    for _ in range(500):
        n = int(rng.integers(2, 8))
        x, xs, g = rng.random(n) + 1e-3, rng.random(n), rng.normal(size=n)
        z, zs = OrthantChart.chi(x), OrthantChart.chi(xs)
        lhs = OrthantChart.grad_L(z, g) @ (z - zs)
        rhs = g @ (x - xs) + g @ (np.sqrt(x) - np.sqrt(xs)) ** 2
        assert abs(lhs - rhs) < 1e-12


def test_1d_counterexample_threshold():
    # h(x) = 1/2 (t-1)^2 (t+1)(3t-1) muda de sinal em x = 1/9
    t = np.sqrt(np.array([0.05, 1 / 9, 0.2]))
    h = 0.5 * (t - 1) ** 2 * (t + 1) * (3 * t - 1)
    assert h[0] < 0 and abs(h[1]) < 1e-12 and h[2] > 0


def test_charts_are_inverses_in_chart():
    n = 6
    x = rng.random(n) + 0.1
    assert np.allclose(OrthantChart.phi(OrthantChart.chi(x)), x)
    xs = x / x.sum()
    assert np.allclose(SimplexAffineChart.phi(SimplexAffineChart.chi(xs)), xs)
    assert np.allclose(SimplexSphereChart.phi(SimplexSphereChart.chi(xs)), xs)
    A = rng.normal(size=(n, n)); X = A @ A.T + np.eye(n)
    assert np.allclose(CholeskyChart.phi(CholeskyChart.chi(X)), X)


def test_selections_preserve_phi():
    z = rng.normal(size=5)
    assert np.allclose(OrthantChart.phi(OrthantChart.select(z)), OrthantChart.phi(z))
    u = rng.normal(size=5); u /= u.sum()
    assert np.allclose(SimplexAffineChart.phi(SimplexAffineChart.select(u)), SimplexAffineChart.phi(u))
    assert np.allclose(SimplexSphereChart.phi(SimplexSphereChart.select(u)), SimplexSphereChart.phi(u))
    L = np.tril(rng.normal(size=(4, 4)))
    assert np.allclose(CholeskyChart.phi(CholeskyChart.select(L)), CholeskyChart.phi(L))


def test_orthant_selection_nonexpansive():
    for _ in range(1000):
        r, zs = rng.normal(size=5), np.abs(rng.normal(size=5))
        assert np.linalg.norm(np.abs(r) - zs) <= np.linalg.norm(r - zs) + 1e-12


def test_affine_simplex_selection_expands():
    eps = 1e-3
    u_star, u = np.array([1.0, 0, 0]), np.array([1.0, eps, -eps])
    s = SimplexAffineChart.select(u)
    assert np.linalg.norm(s - u_star) > np.linalg.norm(u - u_star)


def test_sphere_closed_form_step():
    for _ in range(300):
        n = int(rng.integers(2, 7))
        x = rng.random(n); x /= x.sum()
        g, eta = rng.normal(size=n), rng.uniform(0.01, 0.4)
        u = np.sqrt(x)
        x_exp = SimplexSphereChart.exp_sphere(u, -eta * SimplexSphereChart.grad_L(u, g)) ** 2
        x_cf = SimplexSphereChart.geodesic_step(x, g, eta)
        assert np.allclose(x_exp, x_cf, atol=1e-12) and abs(x_cf.sum() - 1) < 1e-12 and np.all(x_cf >= 0)


def test_cholesky_selection_not_nonexpansive():
    L_star = np.array([[1.0, 0.0], [2.0, 1.0]])
    L = np.array([[-0.1, 0.0], [2.0, 1.0]])
    assert np.linalg.norm(CholeskyChart.select(L) - L_star) > np.linalg.norm(L - L_star)


@pytest.mark.parametrize("name", ["orthant", "simplex", "psd"])
def test_gradients_vs_finite_differences(name):
    p = make_all(0)[name]
    x = p.x0.copy()
    d = rng.normal(size=x.shape)
    if x.ndim == 2:
        d = d + d.T
    h = 1e-6
    fd = (p.ell(x + h * d) - p.ell(x - h * d)) / (2 * h)
    assert abs(np.sum(p.grad_x(x) * d) - fd) < 1e-6 * max(1, abs(fd))
    z = p.chart.chi(x)
    dz = np.tril(rng.normal(size=z.shape)) if z.ndim == 2 else rng.normal(size=z.shape)
    if name == "simplex":
        dz -= dz.mean()
    fd = (p.ell(p.chart.phi(z + h * dz)) - p.ell(p.chart.phi(z - h * dz))) / (2 * h)
    assert abs(np.sum(p.grad_chart(z) * dz) - fd) < 1e-6 * max(1, abs(fd))


def test_ggd_orthant_monotone_and_feasible():
    p = make_all(0)["orthant"]
    out = ggd(p, 0.05, 100, reflect=True)
    assert np.all(np.diff(out["gap"]) <= 1e-12) and np.all(out["x"] >= 0)


def test_ggd_sphere_feasible_and_descending():
    p = make_all(0)["simplex"]
    out = ggd_sphere(p, 0.05, 100)
    assert abs(out["x"].sum() - 1) < 1e-12 and np.all(out["x"] >= 0)
    assert np.all(np.diff(out["gap"]) <= 1e-12)


def test_sphere_descent_and_chi2_bound():
    # Prop. 6.7 (i)-(ii): descida com eta = 1/beta_S e Delta <= sqrt(1+chi2) D para ell convexa.
    from geogd.benchmarks import project_simplex, random_spd
    r = np.random.default_rng(7)
    for _ in range(20):
        n = int(r.integers(3, 8))
        A = random_spd(r, n); xbar = r.normal(size=n)
        ell = lambda x: 0.5 * (x - xbar) @ A @ (x - xbar)
        grad = lambda x: A @ (x - xbar)
        G = np.array([grad(np.eye(n)[i]) for i in range(n)])
        eta = 1.0 / (4 * np.linalg.norm(A, 2) + 2 * (G.max() - G.min()))
        xs = np.ones(n) / n
        for _ in range(400):
            xs = project_simplex(xs - grad(xs) / np.linalg.norm(A, 2))
        x = r.random(n) + 1e-3; x /= x.sum()
        for _ in range(50):
            d = grad(x) - x @ grad(x); D2 = np.sum(x * d**2)
            chi2 = np.sum(xs**2 / x) - 1.0
            assert ell(x) - ell(xs) <= np.sqrt(1 + chi2) * np.sqrt(D2) + 1e-9
            xn = SimplexSphereChart.geodesic_step(x, grad(x), eta)
            assert ell(xn) <= ell(x) - 2 * eta * D2 + 1e-9 * max(1, abs(ell(x)))
            x = xn


def _load_kl_checks():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "kl_lyapunov_checks.py"
    spec = importlib.util.spec_from_file_location("kl_lyapunov_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.rng = np.random.default_rng(11)
    return mod


def test_orthant_kl_rate():
    # Teo. 5.8: sem reflexão, descida, passo em KL e taxa O(1/N) com constante KL(x*||x_0).
    o = _load_kl_checks().check_orthant(n_trials=25, iters=200)
    assert 0.25 - 1e-10 <= o["factor_min"] and o["factor_max"] <= 2.25 + 1e-10
    assert max(o["descent"], o["kl_step"], o["rate"]) <= 1e-10
    assert o["reflections_old_step"] == 0


def test_sphere_kl_rate():
    # Prop. 6.8: Q <= 4 beta_S Delta, passo em KL e taxa O(1/N) no chart esférico.
    s = _load_kl_checks().check_sphere(n_trials=25, iters=200)
    assert s["factor_min"] > 0
    assert max(s["Q"], s["kl_step"], s["rate"]) <= 1e-10


def test_local_acceleration_ingredients():
    # Teo. thm:orthant-local-accel: Hessiana por blocos em z*, potencial de um passo e confinamento.
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "local_acceleration_checks.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("local_acceleration_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    err, slack = mod.check_hessian(n_trials=40)
    assert err < 1e-10 and slack >= -1e-10
    viol, exits, rate = mod.check_potential(n_trials=15, iters=150)
    assert viol <= 1e-8 and rate <= 1e-8 and exits == 0


def test_likelihood_references_and_gradients():
    # Família C: certificado de dualidade da solução de referência, ponto fixo da baseline
    # multiplicativa, e gradientes (em x e na carta) contra diferenças finitas.
    from geogd.likelihood import make_all_likelihood
    r = np.random.default_rng(5)
    P = make_all_likelihood(0)
    for name in ("orthant", "simplex", "orthant-dense"):
        p = P[name]
        assert abs(p.certificate) < 1e-11
        xs = p.x_star
        assert np.linalg.norm(p.mult(xs) - xs) <= 1e-9 * max(1.0, np.linalg.norm(xs))
        assert p.info["strict_compl_margin"] > 0
        assert p.gap(p.x0) > 0
    for name, p in P.items():
        x = p.x0
        g = p.grad_x(x)
        for _ in range(3):
            v = r.normal(size=x.shape)
            if x.ndim == 2:
                v = v + v.T
            if "simplex" in name:
                v -= v.mean()
            v *= 1e-6 * np.linalg.norm(x) / np.linalg.norm(v)
            fd = (p.ell(x + v) - p.ell(x - v)) / 2.0
            assert abs(fd - np.sum(g * v)) <= 1e-6 * abs(np.sum(g * v)) + 1e-14
        z = p.chart.chi(x)
        gz = p.grad_chart(z)
        for _ in range(3):
            w = r.normal(size=z.shape)
            if z.ndim == 2:
                w = np.tril(w)
            if "simplex" in name:
                w -= w.mean()
            w *= 1e-6 * np.linalg.norm(z) / np.linalg.norm(w)
            fd = (p.ell(p.chart.phi(z + w)) - p.ell(p.chart.phi(z - w))) / 2.0
            assert abs(fd - np.sum(gz * w)) <= 1e-5 * abs(np.sum(gz * w)) + 1e-13
