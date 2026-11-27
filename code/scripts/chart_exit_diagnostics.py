"""Diagnóstico de permanência na carta nos benchmarks das Famílias A e B (semente 0, passos
ajustados registrados em figures/numerics_summary.json).

Para cada método de carta conta, em 150 iterações:
  fora       avaliações do gradiente levantado fora da carta aberta V;
  travessias avaliações em que o padrão de sinais muda em relação à anterior.

Leitura.  No ortante e na carta de Cholesky as trocas de sinal são simetrias de phi, de modo que,
para o GGD, a iteração sem reflexão produz os mesmos iterados em x que a iteração com reflexão,
e `travessias` é o número de iterações em que a reflexão atua.  Na carta afim do simplex as
trocas de sinal não são simetrias: `fora` > 0 significa continuação viável, fora da teoria
geodésica.  No GGD-S, `fora` conta as iterações em que a reflexão w -> |w| atua.

Uso:  python chart_exit_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_likelihood_benchmarks import chart_exits  # noqa: E402
from geogd.benchmarks import make_all, make_all_euclid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> dict:
    summary = json.load(open(ROOT / "figures" / "numerics_summary.json"))
    out = {}
    for fam, make in (("chart-quadratic", make_all), ("euclidean-quadratic", make_all_euclid)):
        print(fam)
        for name, prob in make(0).items():
            key = name.split("-")[0]
            for m in ("GGD", "AGGD", "GGD-S"):
                if m not in summary[fam][key]:
                    continue
                eta = summary[fam][key][m]["eta_seed0"]
                st = chart_exits(prob, m, eta, 150)
                out[(fam, key, m)] = st
                print(f"  {key:8s} {m:6s} eta={eta:.4g}  fora={st['outside']:3d}  travessias={st['crossings']:3d}")
    return out


if __name__ == "__main__":
    main()
