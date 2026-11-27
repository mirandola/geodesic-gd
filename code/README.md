# Code for "Geodesic Methods for Constraint-Preserving Optimization via Parametrization-Induced Geometry"

A. Bhaya, H. Mirandola, C. Mayrink Verdun

This folder reproduces every figure and table of the numerical section of the paper and checks
numerically the inequalities behind its main theorems. No external data are used: all problem
instances are generated from fixed random seeds.

## Quick start

```bash
pip install -r requirements.txt          # numpy, matplotlib, pytest (jupyter only for the notebook)
python -m pytest tests                   # 18 tests, a few seconds
python -m nbconvert --to notebook --execute --inplace reproduce_paper.ipynb   # about 6 minutes
```

`reproduce_paper.ipynb` is the entry point. It is stored already executed, with all outputs
embedded, and `reproduce_paper.html` is a read-only copy that needs no Python installation.

## Contents

| Path | What it is |
|---|---|
| `reproduce_paper.ipynb` | commented notebook: closed-form geodesic steps, Table 1, Figures 1–2, chart-exit diagnostic, Table 2, numerical checks of the theorems |
| `geogd/charts.py` | the charts: square chart of the orthant, affine and spherical charts of the simplex, Cholesky chart |
| `geogd/benchmarks.py` | Families A (chart-quadratic) and B (Euclidean-quadratic) on the orthant, the simplex and the PSD cone |
| `geogd/likelihood.py` | likelihood objectives: Poisson deconvolution, Poisson with a dense design, Gaussian mixture, Gaussian likelihood |
| `geogd/methods.py` | PGD/APGD, MD/AMD, GGD/AGGD, GGD-S, and the step-size tuning protocol |
| `scripts/generate_boundary_comparative_graphs.py` | Figures 1–2 and Table 1 (writes to `../figures`) |
| `scripts/generate_likelihood_benchmarks.py` | likelihood figures and Table 2 (writes to `../figures`) |
| `scripts/chart_exit_diagnostics.py` | how often the chart iterates leave the chart |
| `scripts/kl_lyapunov_checks.py` | global `O(1/k)` rates with a Kullback–Leibler constant (orthant and spherical chart) |
| `scripts/local_acceleration_checks.py` | Hessian of the lift at the minimizer, local accelerated rate, evidence on the global behaviour of AGGD |
| `scripts/sphere_rate_checks.py`, `check_transfer_identities.py`, `simplex_fejer_excess.py`, `cholesky_selection_counterexample.py` | further identities and counterexamples quoted in the extended version |
| `tests/test_identities.py` | unit tests of the identities and inequalities |

## Protocol used in the experiments

All methods share the initialization. For each benchmark and method a constant step size is tuned
over a logarithmic grid: among the stable runs (finite iterates, final gap not above the initial
one) the run with the smallest final objective gap is kept. Final gaps that coincide up to 0.1% or
lie below `1e-12` are regarded as equal, and such ties are broken by the earliest iteration at
which the gap falls below `1e-6`; this makes the selected step, and hence the reported hitting
iterations, independent of rounding noise. The rule is implemented in `geogd.methods.tune_and_run`.

Comments and docstrings inside the Python sources are written in Portuguese; the notebook and
this file are in English.

## License

MIT (see `LICENSE` in the repository root).
