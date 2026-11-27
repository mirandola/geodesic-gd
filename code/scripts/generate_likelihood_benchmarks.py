"""Família C: benchmarks de máxima verossimilhança (geogd/likelihood.py).

Saídas (em ../../figures):
    boundary_examples_likelihood.pdf       gap e vazamento, orçamento de 150 iterações
                                           (Poisson/deconvolução, NPMLE de mistura, MLE gaussiana)
    boundary_examples_likelihood_500.pdf   idem, orçamento de 500 iterações
    boundary_examples_likelihood_dense{,_500}.pdf   controle: Poisson com desenho denso
    numerics_likelihood.tex                linhas de tabela no formato de numerics_summary.tex
                                           (mesmas 7 colunas de métodos; orçamento 150)
    numerics_likelihood_mult.tex           idem, com a coluna extra da baseline multiplicativa
    numerics_likelihood.json               números, passos escolhidos, robustez em 5 sementes,
                                           diagnósticos (saídas da carta, complementaridade)

Protocolo idêntico ao de generate_boundary_comparative_graphs.py: semente fixa, mesma
inicialização para todos os métodos, passo constante escolhido numa grade logarítmica pela
corrida estável de menor gap final, robustez em 5 sementes.  Diferenças: (i) o minimizador
não é conhecido por construção; é calculado por Newton com conjunto ativo e certificado por
um limitante de dualidade (< 1e-12, ver `certificate` no JSON); (ii) inclui-se a baseline
multiplicativa clássica sem passo (EM no simplex, Richardson--Lucy no ortante), rótulo "EM/RL";
(iii) roda-se também com orçamento de 500 iterações.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_boundary_comparative_graphs import MARK_EVERY, STYLE, fmt_sci  # noqa: E402
from geogd.likelihood import make_all_likelihood  # noqa: E402
from geogd.methods import EXTRA_METHODS, METHODS, tune_and_run  # noqa: E402

METHODS = dict(METHODS, **EXTRA_METHODS)
COLS = ["PGD", "APGD", "MD", "AMD", "GGD", "AGGD", "GGD-S"]
MULT = "EM/RL"
STYLE = dict(STYLE, **{MULT: ("#7F7F7F", ":", "x")})

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7})

BUDGETS = (150, 500)
TOL = 1e-6
SEED = 0
N_SEEDS = 5
TITLES = {
    "orthant": r"Orthant $\mathbb{R}^{64}_+$: Poisson deconvolution",
    "simplex": r"Simplex $\Delta_{59}$: mixture NPMLE",
    "psd": r"SPD(5): Gaussian likelihood",
    "orthant-dense": r"Orthant $\mathbb{R}^{40}_+$: Poisson, dense design",
}
LEAK_LABEL = {"orthant": "boundary leakage", "simplex": "boundary leakage",
              "psd": "relative error (interior minimizer)", "orthant-dense": "boundary leakage"}
MAIN = ("orthant", "simplex", "psd")
ROWS = [("orthant", "Orthant"), ("simplex", "Simplex"), ("psd", "PSD"), ("orthant-dense", "Orthant (dense)")]


def first_hit(g: np.ndarray):
    return int(np.argmax(g < TOL)) if np.any(g < TOL) else None


def run_mult(prob, iters: int) -> dict:
    x = prob.x0.copy()
    gaps, leaks = [prob.gap(x)], [prob.leakage(x)]
    for _ in range(iters):
        x = prob.mult(x)
        gaps.append(prob.gap(x)); leaks.append(prob.leakage(x))
    return {"gap": np.asarray(gaps), "leak": np.asarray(leaks)}


def chart_exits(prob, method: str, eta: float, iters: int) -> dict:
    """Diagnóstico de permanência na carta aberta V, nos pontos em que o gradiente levantado
    é avaliado (GGD: iterados; AGGD: pontos extrapolados; GGD-S: passo geodésico antes de |w|).

    outside   número de avaliações fora de V (alguma coordenada de carta <= 0);
    crossings número de avaliações em que o padrão de sinais muda em relação à anterior, isto
              é, quantas vezes a trajetória atravessa a fronteira da carta.  No ortante e em
              Cholesky as trocas de sinal são simetrias de phi (cada travessia é uma reflexão
              implícita); na carta afim do simplex não são, e `outside` > 0 significa que o
              método está no regime de continuação viável, fora da teoria geodésica."""
    sign = lambda z: np.sign(np.diag(z)) if z.ndim == 2 else np.sign(z)
    stats = {"outside": 0, "crossings": 0}
    last = [None]

    def visit(z):
        sg = sign(z)
        stats["outside"] += int(np.any(sg <= 0))
        if last[0] is not None and np.any(sg != last[0]):
            stats["crossings"] += 1
        last[0] = sg

    if method == "GGD-S":
        from geogd.charts import SimplexSphereChart as S
        w = np.sqrt(prob.x0)
        for _ in range(iters):
            w = S.step_from_grad_w(w, prob.grad_w(w), eta)
            stats["outside"] += int(np.any(w <= 0))
            stats["crossings"] += int(np.any(w < 0))
            w = np.abs(w)
        return stats
    orig = prob.grad_chart

    def counting(z):
        visit(z)
        return orig(z)

    prob.grad_chart = counting
    try:
        with np.errstate(all="ignore"):
            METHODS[method](prob, eta, iters)
    finally:
        prob.grad_chart = orig
    return stats


def run_family(seed: int, iters: int) -> tuple[dict, dict]:
    problems = make_all_likelihood(seed)
    res = {name: {} for name in problems}
    for name, prob in problems.items():
        for m in COLS:
            if m in EXTRA_METHODS and prob.grad_w is None:
                continue
            out, eta = tune_and_run(prob, m, iters)
            g = out["gap"]
            res[name][m] = {"eta": eta, "gap": g, "leak": out["leak"], "final": float(g[-1]), "hit": first_hit(g),
                            "eta_at_grid_edge": bool(eta in (prob.step_grid[0], prob.step_grid[-1]))}
            if m in ("GGD", "AGGD", "GGD-S"):
                res[name][m]["chart_exits"] = chart_exits(prob, m, eta, iters)
        if prob.mult is not None:
            out = run_mult(prob, iters)
            g = out["gap"]
            res[name][MULT] = {"eta": None, "gap": g, "leak": out["leak"], "final": float(g[-1]), "hit": first_hit(g),
                               "eta_at_grid_edge": False}
    return res, {name: prob.info for name, prob in problems.items()}


def plot_family(res: dict, names: tuple, path: Path | None, iters: int):
    """Desenha gap e vazamento; salva em `path` (se dado) e devolve a figura (quem chama fecha)."""
    ncol = len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(11 * ncol / 3 if ncol > 1 else 4.6, 6.4), squeeze=False)
    handles = {}
    every = MARK_EVERY * iters // 150
    for j, name in enumerate(names):
        for idx, m in enumerate(COLS + [MULT]):
            if m not in res[name]:
                continue
            r = res[name][m]
            c, ls, mk = STYLE[m]
            kw = dict(color=c, ls=ls, lw=1.4, marker=mk, ms=4, markevery=(2 * idx * iters // 150, every),
                      markerfacecolor="white" if ls == "--" else c, markeredgewidth=1.0)
            (h,) = axes[0, j].semilogy(np.maximum(r["gap"], 1e-16), label=m, **kw)
            axes[1, j].semilogy(np.maximum(r["leak"], 1e-16), **kw)
            handles.setdefault(m, h)
        axes[0, j].set_title(TITLES[name]); axes[1, j].set_xlabel("iteration")
        if j > 0 and LEAK_LABEL[name] != LEAK_LABEL[names[0]]:
            axes[1, j].set_ylabel(LEAK_LABEL[name])
        for i in range(2):
            axes[i, j].grid(alpha=0.3)
            axes[i, j].set_ylim(1e-16, 1e3)          # mesma escala das Figuras 7.1 e 7.2
    axes[0, 0].set_ylabel("objective gap"); axes[1, 0].set_ylabel(LEAK_LABEL[names[0]])
    labels = [m for m in COLS + [MULT] if m in handles]
    fig.legend([handles[m] for m in labels], labels, loc="lower center", frameon=False,
               ncol=len(labels) if ncol > 1 else 4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04 if ncol > 1 else 0.08, 1, 1))
    if path is not None:
        fig.savefig(path)
    return fig


def table_rows(res: dict, cols: list[str]) -> str:
    rows = [r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{likelihood objectives}}\\"]
    for name, label in ROWS:
        cells = []
        for m in cols:
            if m not in res[name]:
                cells.append(r"\multicolumn{1}{c}{n/a}")
                continue
            r = res[name][m]
            k = "--" if r["hit"] is None else str(r["hit"])
            cells.append(f"{fmt_sci(max(r['final'], 0.0))}({k})")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows) + "\n"


def burg_md_one_step_gap(seed: int) -> float:
    """No problema gaussiano, ell = tr(SX) + psi_Burg(X): o passo de mirror com eta = 1 dá
    X_1 = S^{-1} exatamente.  Devolve o gap após esse único passo (deve ser ~ 0)."""
    prob = make_all_likelihood(seed)["psd"]
    return prob.gap(prob.mirror(prob.x0, prob.grad_x(prob.x0), 1.0))


def main() -> None:
    fig_dir = Path(__file__).resolve().parents[2] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = {"notes": {
        "tolerance": TOL, "seed_reported": SEED, "n_seeds": N_SEEDS,
        "psd_burg_md_gap_after_one_step_eta_1": [burg_md_one_step_gap(sd) for sd in range(N_SEEDS)],
        "psd_md_note": "com eta = 1 o MD de Burg resolve o problema gaussiano em um passo; o protocolo "
                       "escolhe o passo pelo menor gap final, e muitos passos empatam em ~0, de modo que "
                       "a 'primeira iteração abaixo de 1e-6' registrada para MD/AMD no PSD depende do desempate.",
    }}
    print("MD de Burg, gap após 1 passo com eta=1:", summary["notes"]["psd_burg_md_gap_after_one_step_eta_1"])
    for iters in BUDGETS:
        res, info = run_family(SEED, iters)
        print(f"\n=== Família C: likelihood (seed {SEED}, {iters} iterações) ===")
        for name in res:
            print(f"{name}: {info[name]}")
            for m, r in res[name].items():
                eta = "  --     " if r["eta"] is None else f"{r['eta']:<9.3g}"
                print(f"  {m:6s} eta={eta} final gap={r['final']:.2e}  k(<1e-6)={r['hit']}"
                      + (f"  fora de V={r['chart_exits']['outside']}, travessias={r['chart_exits']['crossings']}"
                         if "chart_exits" in r else "")
                      + ("  [passo na borda da grade]" if r["eta_at_grid_edge"] else ""))
        suffix = "" if iters == BUDGETS[0] else f"_{iters}"
        plt.close(plot_family(res, MAIN, fig_dir / f"boundary_examples_likelihood{suffix}.pdf", iters))
        plt.close(plot_family(res, ("orthant-dense",), fig_dir / f"boundary_examples_likelihood_dense{suffix}.pdf", iters))
        if iters == BUDGETS[0]:
            (fig_dir / "numerics_likelihood.tex").write_text(table_rows(res, COLS))
            (fig_dir / "numerics_likelihood_mult.tex").write_text(table_rows(res, COLS + [MULT]))
            print("\nLinhas de tabela (150 iterações):\n" + table_rows(res, COLS + [MULT]))

        hits = {name: {m: [] for m in res[name]} for name in res}
        finals = {name: {m: [] for m in res[name]} for name in res}
        exits = {name: {m: [] for m in res[name]} for name in res}
        infos = {name: [] for name in res}
        for sd in range(N_SEEDS):
            r_sd, i_sd = (res, info) if sd == SEED else run_family(sd, iters)
            for name in r_sd:
                infos[name].append(i_sd[name])
                for m in r_sd[name]:
                    hits[name][m].append(r_sd[name][m]["hit"] if r_sd[name][m]["hit"] is not None else iters + 1)
                    finals[name][m].append(r_sd[name][m]["final"])
                    exits[name][m].append(r_sd[name][m].get("chart_exits"))
        summary[f"iters_{iters}"] = {}
        for name in res:
            block = {"problem_seed0": info[name], "problem_all_seeds": infos[name], "methods": {}}
            for m in res[name]:
                med = float(np.median(hits[name][m]))
                block["methods"][m] = {
                    "eta_seed0": res[name][m]["eta"],
                    "eta_at_grid_edge_seed0": res[name][m]["eta_at_grid_edge"],
                    "final_gap_seed0": res[name][m]["final"],
                    "first_k_below_tol_seed0": res[name][m]["hit"],
                    "median_first_k_below_tol_5seeds": None if med > iters else med,
                    "n_seeds_reaching_tol": int(sum(h <= iters for h in hits[name][m])),
                    "median_final_gap_5seeds": float(np.median(finals[name][m])),
                    "final_gap_all_seeds": finals[name][m],
                }
                if m in ("GGD", "AGGD", "GGD-S"):
                    block["methods"][m]["chart_exits_all_seeds"] = exits[name][m]
            summary[f"iters_{iters}"][name] = block
        print(f"\nRobustez em {N_SEEDS} sementes, {iters} iterações "
              "(mediana da 1a iteração com gap < 1e-6; sementes que atingem; mediana do gap final):")
        for name in res:
            print(f"  {name:8s} " + "  ".join(
                f"{m}:{summary[f'iters_{iters}'][name]['methods'][m]['median_first_k_below_tol_5seeds']}"
                f"({summary[f'iters_{iters}'][name]['methods'][m]['n_seeds_reaching_tol']}/5,"
                f"{summary[f'iters_{iters}'][name]['methods'][m]['median_final_gap_5seeds']:.1e})"
                for m in res[name]))
    (fig_dir / "numerics_likelihood.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
