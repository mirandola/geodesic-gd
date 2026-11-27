"""Gera a ilustração vetorial de uma geodésica parabólica no ortante positivo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Cores próximas às usadas pelo tema Metropolis da apresentação.
TEAL = "#23373B"
ORANGE = "#EB811B"
GREEN = "#2A9D8F"
GRAY = "#6B7375"
LIGHT = "#F7F7F4"


def geodesic(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Imagem por phi(z)=z^(odot 2)/4 da reta z(t)=(2t-2, 2t-6)."""
    x1 = (t - 1.0) ** 2
    x2 = (t - 3.0) ** 2
    return x1, x2


def add_direction_arrow(ax, t0: float, t1: float, color: str) -> None:
    """Adiciona uma seta tangente indicando o sentido crescente do parâmetro."""
    x0, y0 = geodesic(np.array([t0]))
    x1, y1 = geodesic(np.array([t1]))
    ax.annotate(
        "",
        xy=(x1[0], y1[0]),
        xytext=(x0[0], y0[0]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 2.6,
            "mutation_scale": 18,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "geodesic_parabolic_two_bounces.pdf"
    output_png = output_dir / "geodesic_parabolic_two_bounces.png"

    # A trajetória toca x_1=0 em tau_1=1 e x_2=0 em tau_2=3.
    t_first = np.linspace(0.0, 1.0, 250)
    t_middle = np.linspace(1.0, 3.0, 500)
    t_last = np.linspace(3.0, 4.0, 250)
    x1_first, x2_first = geodesic(t_first)
    x1_middle, x2_middle = geodesic(t_middle)
    x1_last, x2_last = geodesic(t_last)

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT)

    # O fundo marca explicitamente o quadrante factível R_+^2.
    ax.axhspan(0.0, 9.7, color=LIGHT, zorder=0)
    ax.plot(
        x1_first,
        x2_first,
        color=TEAL,
        linewidth=4.2,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        x1_middle,
        x2_middle,
        color=ORANGE,
        linewidth=4.2,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        x1_last,
        x2_last,
        color=GREEN,
        linewidth=4.2,
        solid_capstyle="round",
        zorder=3,
    )

    add_direction_arrow(ax, 0.30, 0.48, TEAL)
    add_direction_arrow(ax, 1.72, 1.98, ORANGE)
    add_direction_arrow(ax, 3.42, 3.60, GREEN)

    # Pontos inicial, de contato com cada eixo e final.
    x_start, y_start = geodesic(np.array([0.0]))
    x_bounce_1, y_bounce_1 = geodesic(np.array([1.0]))
    x_bounce_2, y_bounce_2 = geodesic(np.array([3.0]))
    x_end, y_end = geodesic(np.array([4.0]))
    ax.scatter(
        [x_start[0], x_end[0]],
        [y_start[0], y_end[0]],
        s=62,
        color=[TEAL, GREEN],
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
    )
    ax.scatter(
        [x_bounce_1[0], x_bounce_2[0]],
        [y_bounce_1[0], y_bounce_2[0]],
        s=145,
        facecolor="white",
        edgecolor=[ORANGE, GREEN],
        linewidth=3.0,
        zorder=7,
        clip_on=False,
    )

    ax.text(x_start[0] + 0.05, y_start[0] + 0.28, r"$t=0$", fontsize=13, ha="left")
    ax.text(x_end[0] - 0.08, y_end[0] + 0.28, r"$t=4$", fontsize=13, ha="right")

    ax.annotate(
        r"$\tau_1=1$" "\n" r"$x(\tau_1)=(0,4)$",
        xy=(x_bounce_1[0], y_bounce_1[0]),
        xytext=(0.95, 5.05),
        fontsize=13,
        color=TEAL,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 1.5},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D8DDDE"},
        zorder=8,
    )
    ax.annotate(
        r"$\tau_2=3$" "\n" r"$x(\tau_2)=(4,0)$",
        xy=(x_bounce_2[0], y_bounce_2[0]),
        xytext=(5.05, 1.05),
        fontsize=13,
        color=TEAL,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 1.5},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D8DDDE"},
        zorder=8,
    )

    ax.text(0.25, 1.25, r"$x_1=0$", color=GRAY, fontsize=12, ha="left", va="center", rotation=90)
    ax.text(1.25, 0.25, r"$x_2=0$", color=GRAY, fontsize=12, ha="center", va="bottom")

    ax.set_xlim(0.0, 9.7)
    ax.set_ylim(0.0, 9.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$", fontsize=16, color=TEAL, labelpad=8)
    ax.set_ylabel(r"$x_2$", fontsize=16, color=TEAL, labelpad=8, rotation=0)
    ax.yaxis.set_label_coords(-0.045, 1.01)

    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="both", colors=GRAY, labelsize=11, length=0)
    ax.grid(True, color="#D9DEDF", linewidth=0.8, alpha=0.8, zorder=1)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(TEAL)
        ax.spines[side].set_linewidth(1.4)

    # Pequenas pontas de seta deixam clara a orientação dos eixos positivos.
    ax.annotate("", xy=(9.7, 0), xytext=(9.35, 0), arrowprops={"arrowstyle": "-|>", "color": TEAL})
    ax.annotate("", xy=(0, 9.7), xytext=(0, 9.35), arrowprops={"arrowstyle": "-|>", "color": TEAL})

    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
