"""Figures for DESIGN.md / README from committed results (PNG, light surface; each one has a table twin in DESIGN.md).

    python scripts/plots.py sft-curve --log eval/results/train_sft_sft.json --out assets/sft_curve.png
    python scripts/plots.py hardness --a eval/results/val/Qwen2.5-Coder-1.5B-Instruct.json --a-label "base, zero-shot" \
        --b eval/results/val/Qwen2.5-Coder-1.5B-Instruct-sft.json --b-label "SFT" --out assets/ex_by_hardness_sft.png

Style: two categorical slots (blue, orange; validated for CVD and contrast on the light surface), 2 px lines, >= 8 px
markers with a 2 px surface ring, columns <= 24 px with a 4 px rounded end and a square base, hairline solid grid, text in
ink tokens (never the series colour), a legend plus selective direct labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ("#2a78d6", "#eb6834")  # categorical slots 1-2
DPI = 200
PT = 0.75  # points per CSS px (96 px = 72 pt), whatever the dpi


def _style(ax, fig) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, length=0, labelsize=9)


def sft_curve(log_path: Path, out: Path) -> None:
    info = json.loads(log_path.read_text())
    train = [(r["step"], r["train_loss"]) for r in info.get("train_log", []) if r.get("train_loss") is not None]
    val = [(r["step"], r["val_loss"]) for r in info["log"] if r.get("val_loss") is not None]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=DPI)
    _style(ax, fig)
    if train:
        ax.plot(*zip(*train), color=SLOT[0], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label="train loss (mean of each 20 steps)", zorder=2)
    ax.plot(*zip(*val), color=SLOT[1], linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker="o", markersize=8,
            markerfacecolor=SLOT[1], markeredgecolor=SURFACE, markeredgewidth=2, label="val loss (300 val questions)", zorder=3)
    losses = [v for _, v in val] + [v for _, v in train]
    lo, hi = min(losses), max(losses)
    if hi / max(lo, 1e-9) > 4:  # a steep first drop would squash the rest on a linear axis
        ax.set_yscale("log")
        ticks = [t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5) if lo * 0.8 <= t <= hi * 1.25]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
    steps_per_epoch = info["steps"] / max(info.get("epochs_run", 1), 1)
    if info.get("epochs_run", 1) > 1:
        ax.axvline(steps_per_epoch, color=AXIS, linewidth=1, zorder=1)
        ax.text(steps_per_epoch, ax.get_ylim()[1], " epoch 2", color=MUTED, fontsize=8, va="top", ha="left")
    best_step, best = min(((s, v) for s, v in val if s > 0), key=lambda sv: sv[1], default=val[-1])
    ax.annotate(f"saved adapter: lowest val loss, {best:.3f} at step {best_step}", xy=(best_step, best), xytext=(28, 42), textcoords="offset points",
                ha="left", va="bottom", color=INK_2, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1, "shrinkB": 6})
    ax.set_xlabel("optimiser step", color=INK_2, fontsize=10)
    ax.set_ylabel("cross-entropy per target token", color=INK_2, fontsize=10)
    ax.set_title(f"SFT of {info.get('base', 'the base model').split('/')[-1]}: loss by step ({info.get('epochs_run')} epochs, max_len "
                 f"{info['rules']['max_len']:,})", color=INK, fontsize=11, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK_2)  # the empty corner: both curves start high
    leg.set_zorder(4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _per_point(ax, fig) -> tuple[float, float]:
    """Data units per typographic point along x and along y (limits must be set)."""
    box = ax.get_window_extent()
    pts_w, pts_h = box.width * 72 / fig.dpi, box.height * 72 / fig.dpi
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / pts_w, (y1 - y0) / pts_h


def _column(ax, x: float, height: float, width: float, color: str, ux: float, uy: float) -> None:
    """A column with a 4 px rounded top and a square base; ux / uy = data units per point."""
    rx, ry = min(4 * PT * ux, width / 2), 4 * PT * uy
    ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=ry / rx, facecolor=color,
                                edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((x, 0), width, min(height, 2 * ry), facecolor=color, edgecolor="none", zorder=2))  # square the base


def hardness(a: Path, a_label: str, b: Path, b_label: str, out: Path) -> None:
    sa, sb = json.loads(a.read_text()), json.loads(b.read_text())
    levels = [h for h in ("easy", "medium", "hard", "extra") if h in sa["by_hardness"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=DPI)
    _style(ax, fig)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.canvas.draw()
    ux, uy = _per_point(ax, fig)
    width = min(24 * PT * ux, 0.3)  # columns <= 24 px; the rest of the band stays air
    gap = 2 * PT * ux  # the 2 px surface gap between touching columns
    for i, h in enumerate(levels):
        va, vb = 100 * sa["by_hardness"][h]["ex"], 100 * sb["by_hardness"][h]["ex"]
        _column(ax, i - width - gap / 2, va, width, SLOT[0], ux, uy)
        _column(ax, i + gap / 2, vb, width, SLOT[1], ux, uy)
        ax.text(i + gap / 2 + width / 2, vb + 1.5, f"{vb:.0f}", ha="center", va="bottom", color=INK, fontsize=9)  # the story: the trained model
    ax.set_xticks(range(len(levels)), [f"{h}\n(n = {sa['by_hardness'][h]['n']})" for h in levels], color=INK_2)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_title(f"Execution accuracy on val by Spider hardness: {a_label} {100 * sa['accuracy']['ex']:.1f}%, {b_label} {100 * sb['accuracy']['ex']:.1f}%",
                 color=INK, fontsize=11, loc="left", pad=12)
    handles = [Rectangle((0, 0), 1, 1, facecolor=SLOT[0]), Rectangle((0, 0), 1, 1, facecolor=SLOT[1])]
    ax.legend(handles, [a_label, b_label], frameon=False, fontsize=9, loc="upper right", labelcolor=INK_2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("sft-curve")
    c.add_argument("--log", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    h = sub.add_parser("hardness")
    h.add_argument("--a", type=Path, required=True)
    h.add_argument("--a-label", default="base")
    h.add_argument("--b", type=Path, required=True)
    h.add_argument("--b-label", default="SFT")
    h.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.what == "sft-curve":
        sft_curve(a.log, a.out)
    else:
        hardness(a.a, a.a_label, a.b, a.b_label, a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
