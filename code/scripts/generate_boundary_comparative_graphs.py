"""Gera as figuras e a tabela da Seção 7 do artigo.

Saídas (em ../../figures):
    boundary_examples_chart_quadratic.pdf      (Figura 7.1: família A, gap e vazamento)
    boundary_examples_euclidean_quadratic.pdf  (Figura 7.2: família B, gap e vazamento)
    numerics_summary.tex                       (linhas da Tabela 7.1, dois blocos)
    numerics_summary.json                      (números, passos escolhidos e
                                                robustez em 5 sementes)

Protocolo (Subseção 7.2): semente fixa, mesma inicialização para os seis
métodos, 150 iterações, passo constante escolhido numa grade logarítmica pela
corrida estável de menor gap final.  A robustez é verificada repetindo o
protocolo em 5 sementes e registrando a mediana da primeira iteração em que
o gap cai abaixo de 1e-6.

Família A: objetivos quadráticos NA CARTA (convexos na carta; grad ell(x)
singular na fronteira, ver benchmarks.py).  Família B (controle): quadráticas
euclidianas suaves com minimizador de fronteira construído por KKT, em que
todos os métodos recebem gradientes regulares.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geogd.benchmarks import make_all, make_all_euclid  # noqa: E402
from geogd.methods import EXTRA_METHODS, METHODS, tune_and_run  # noqa: E402

METHODS = dict(METHODS, **EXTRA_METHODS)          # GGD-S só roda no simplex
COLS = ["PGD", "APGD", "MD", "AMD", "GGD", "AGGD", "GGD-S"]

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7})

ITERS = 150
TOL = 1e-6
SEED = 0
# Cada método tem cor (paleta Okabe-Ito), estilo de linha e marcador próprios, fixos nas
# duas figuras, de modo que as curvas se distingam também em impressão preto e branco:
# projetados = círculos, mirror = quadrados, geodésicos = triângulos; acelerados tracejados.
STYLE = {  # cor, estilo de linha, marcador
    "PGD": ("#0072B2", "-", "o"), "APGD": ("#56B4E9", "--", "o"),
    "MD": ("#009E73", "-", "s"), "AMD": ("#CC79A7", "--", "s"),
    "GGD": ("#D55E00", "-", "^"), "AGGD": ("#E69F00", "--", "^"),
    "GGD-S": ("#000000", "-.", "D"),
}
MARK_EVERY = 15   # um marcador a cada 15 iterações, deslocado por método para não sobrepor
TITLES = {"orthant": r"Orthant $\mathbb{R}^{40}_+$", "simplex": r"Simplex $\Delta_{29}$", "psd": r"PSD(5)"}
FAMILIES = {  # rótulo, construtor, arquivo da figura, rótulo de tabela
    "A": ("chart-quadratic", make_all, "boundary_examples_chart_quadratic.pdf", "chart-quadratic"),
    "B": ("euclidean-quadratic", make_all_euclid, "boundary_examples_euclidean_quadratic.pdf", "Euclidean-quadratic"),
}
N_SEEDS = 5


def fmt_sci(v: float) -> str:
    """Formata como $a\\times 10^{b}$ (ou $a$ se b=0)."""
    if v < 1e-16:
        return "$<10^{-16}$"
    b = int(np.floor(np.log10(v)))
    a = v / 10**b
    return f"${a:.2f}$" if b == 0 else f"${a:.2f}\\times 10^{{{b}}}$"


def run_family(make, seed: int) -> dict:
    problems = make(seed)
    res = {name: {} for name in problems}
    for name, prob in problems.items():
        for m in METHODS:
            if m in EXTRA_METHODS and prob.grad_w is None:
                continue
            out, eta = tune_and_run(prob, m, ITERS)
            g = out["gap"]
            hit = int(np.argmax(g < TOL)) if np.any(g < TOL) else None
            res[name][m] = {"eta": eta, "gap": g, "leak": out["leak"], "final": float(g[-1]), "hit": hit}
    return res


def plot_family(res: dict, path_list: list[Path] = ()):
    """Desenha gap e vazamento; salva em cada caminho de `path_list` e devolve a figura
    (quem chama fecha; o notebook reproduce_paper.ipynb a exibe)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    handles = {}
    for j, name in enumerate(("orthant", "simplex", "psd")):
        for idx, m in enumerate(COLS):
            if m not in res[name]:
                continue
            r = res[name][m]
            c, ls, mk = STYLE[m]
            kw = dict(color=c, ls=ls, lw=1.4, marker=mk, ms=4, markevery=(2 * idx, MARK_EVERY),
                      markerfacecolor="white" if ls == "--" else c, markeredgewidth=1.0)
            (h,) = axes[0, j].semilogy(np.maximum(r["gap"], 1e-16), label=m, **kw)
            axes[1, j].semilogy(np.maximum(r["leak"], 1e-16), **kw)
            handles.setdefault(m, h)
        axes[0, j].set_title(TITLES[name]); axes[1, j].set_xlabel("iteration")
        for i in range(2):
            axes[i, j].grid(alpha=0.3)
            axes[i, j].set_ylim(1e-16, 1e3)          # mesma escala nas duas figuras
    axes[0, 0].set_ylabel("objective gap"); axes[1, 0].set_ylabel("boundary leakage")
    fig.legend([handles[m] for m in COLS], COLS, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for d in path_list:
        fig.savefig(d)
    return fig


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dirs = [root / "figures"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    rows, summary = [], {}
    for key, (fam, make, figname, tablabel) in FAMILIES.items():
        res = run_family(make, SEED)
        print(f"\n=== Família {key}: {fam} (seed {SEED}) ===")
        for name in res:
            for m, r in res[name].items():
                print(f"{name:8s} {m:5s} eta={r['eta']:<9.3g} final gap={r['final']:.2e}  k(<1e-6)={r['hit']}")
        plt.close(plot_family(res, [d / figname for d in out_dirs]))

        # robustez: mediana em N_SEEDS sementes da primeira iteração abaixo de TOL
        hits = {name: {m: [] for m in res[name]} for name in res}
        finals = {name: {m: [] for m in res[name]} for name in res}
        for sd in range(N_SEEDS):
            r_sd = run_family(make, sd)
            for name in r_sd:
                for m in r_sd[name]:
                    hits[name][m].append(r_sd[name][m]["hit"] if r_sd[name][m]["hit"] is not None else ITERS + 1)
                    finals[name][m].append(r_sd[name][m]["final"])
        summary[fam] = {}
        for name in res:
            summary[fam][name] = {}
            for m in res[name]:
                med = float(np.median(hits[name][m]))
                summary[fam][name][m] = {
                    "eta_seed0": res[name][m]["eta"],
                    "final_gap_seed0": res[name][m]["final"],
                    "first_k_below_tol_seed0": res[name][m]["hit"],
                    "median_first_k_below_tol_5seeds": None if med > ITERS else med,
                    "n_seeds_reaching_tol": int(sum(h <= ITERS for h in hits[name][m])),
                    "median_final_gap_5seeds": float(np.median(finals[name][m])),
                }

        rows.append(r"\multicolumn{8}{l}{\emph{" + tablabel + r" objectives}}\\")
        for name, label in [("orthant", "Orthant"), ("simplex", "Simplex"), ("psd", "PSD")]:
            cells = []
            for m in COLS:
                if m not in res[name]:
                    cells.append(r"\multicolumn{1}{c}{n/a}")
                    continue
                r = res[name][m]
                k = "--" if r["hit"] is None else str(r["hit"])
                cells.append(f"{fmt_sci(max(r['final'], 0.0))}({k})")
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        if key == "A":
            rows.append(r"\midrule")
    table = "\n".join(rows) + "\n"
    for d in out_dirs:
        (d / "numerics_summary.tex").write_text(table)
        (d / "numerics_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nLinhas da Tabela 7.1:\n" + table)
    print("Robustez (mediana em 5 sementes da 1a iteração com gap < 1e-6):")
    for fam in summary:
        for name in summary[fam]:
            print(f"  {fam:20s} {name:8s} " + "  ".join(
                f"{m}:{summary[fam][name][m]['median_first_k_below_tol_5seeds']}({summary[fam][name][m]['n_seeds_reaching_tol']}/5)"
                for m in summary[fam][name]))


if __name__ == "__main__":
    main()
