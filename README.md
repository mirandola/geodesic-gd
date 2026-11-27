# Geodesic Methods for Constraint-Preserving Optimization via Parametrization-Induced Geometry

Code accompanying the paper by Amit Bhaya, Heudson Mirandola and Claudio Mayrink Verdun.

The paper studies first-order methods for constrained convex optimization in which a smooth
parametrization of the feasible set induces a Riemannian metric, and the iteration follows the
geodesics of that metric: a Euclidean gradient step in chart coordinates, mapped back to the
feasible set. This repository reproduces every figure and table of the numerical section and
checks numerically the inequalities behind the main theorems. No external data are used; all
problem instances are generated from fixed random seeds.

## Where to start

* [`code/reproduce_paper.ipynb`](code/reproduce_paper.ipynb) — commented notebook, stored already
  executed with all outputs embedded (GitHub renders it directly). It covers the closed-form
  geodesic steps, the benchmark tables and figures, the chart-exit diagnostic, the likelihood
  benchmarks, and the numerical checks of the theorems.
* [`code/README.md`](code/README.md) — description of the package, the scripts and the
  step-size tuning protocol.

```bash
cd code
pip install -r requirements.txt
python -m pytest tests                                                        # 18 tests
python -m nbconvert --to notebook --execute --inplace reproduce_paper.ipynb   # about 6 minutes
```

## Layout

| Path | Contents |
|---|---|
| `code/geogd/` | charts (orthant, simplex, Cholesky), benchmarks, likelihood objectives, methods |
| `code/scripts/` | generators of the figures and tables, and numerical checks of the theorems |
| `code/tests/` | unit tests of the identities and inequalities |
| `figures/` | figures and numerical summaries (`.json`, `.tex`) produced by the scripts |

## License

MIT, see [`LICENSE`](LICENSE).
