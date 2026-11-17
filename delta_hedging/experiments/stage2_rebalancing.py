"""Stage 2: how does hedging error depend on rebalancing frequency?

A dealer sells a 3-month at-the-money call and delta-hedges it. The stock
realises exactly the vol the option was priced at, so any P&L comes only
from hedging at discrete times instead of continuously.

The same 20,000 simulated paths are hedged at every frequency (the fine
grid is subsampled), so the frequencies differ only in how often the
hedger trades and not in the randomness.

Run from the project folder:

    python3 experiments/stage2_rebalancing.py

Writes charts and a summary table to results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hedging import bs  # noqa: E402
from hedging.hedge import delta_hedge  # noqa: E402
from hedging.paths import simulate_gbm  # noqa: E402

OUT = ROOT / "results"

# ---- Set-up --------------------------------------------------------------
S0, K, T, r, SIGMA, MU = 100.0, 100.0, 0.25, 0.03, 0.20, 0.08
TRADING_DAYS = 63                 # in 3 months
FINE_STEPS = 504                  # 8 per trading day, roughly hourly
N_PATHS = 20_000
SEED = 2026

FREQUENCIES = {                   # rebalances over the option's life
    4: "every ~3 weeks",
    6: "every ~2 weeks",
    12: "~weekly",
    21: "every 3 days",
    63: "daily",
    126: "twice daily",
    252: "4x daily",
    504: "~hourly",
}

# ---- Chart style (reference data-viz palette) ----------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
    "legend.frameon": False,
})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    paths = simulate_gbm(S0, MU, SIGMA, T, FINE_STEPS, N_PATHS, seed=SEED,
                         antithetic=True)
    vega0 = float(bs.vega(S0, K, T, r, SIGMA))
    premium = float(bs.price(S0, K, T, r, SIGMA))

    rows, results = [], {}
    for n, label in FREQUENCIES.items():
        res = delta_hedge(paths[:, :: FINE_STEPS // n], K, T, r, SIGMA)
        results[n] = res
        dk = np.sqrt(np.pi / 4) * vega0 * SIGMA / np.sqrt(n)
        rows.append(dict(
            n=n, label=label, mean=res.mean, std=res.std, dk=dk,
            std_pct=100 * res.std / premium,
            corr=float(np.corrcoef(res.pnl, res.gamma_pnl)[0, 1]),
            p01=float(np.percentile(res.pnl, 1)),
        ))

    ns = np.array([row["n"] for row in rows])
    stds = np.array([row["std"] for row in rows])
    slope, intercept = np.polyfit(np.log(ns), np.log(stds), 1)

    write_table(rows, premium, slope)
    chart_error_vs_frequency(ns, stds, vega0, slope)
    chart_distributions(results)
    chart_attribution(results[63])

    print(f"Premium: {premium:.4f}   vega: {vega0:.3f}")
    print(f"log-log slope of std vs N: {slope:.3f} (theory: -0.5)")
    for row in rows:
        print(f"N={row['n']:>4} {row['label']:<15} mean={row['mean']:+.4f} "
              f"std={row['std']:.4f}  Derman-Kamal={row['dk']:.4f}  "
              f"corr={row['corr']:.3f}")


def write_table(rows, premium, slope) -> None:
    lines = [
        "# Stage 2 results: hedging error vs rebalancing frequency",
        "",
        f"Short 3-month ATM call, S0=K={S0:.0f}, r={r:.0%}, sigma={SIGMA:.0%} "
        f"(priced, hedged and realised), real-world drift {MU:.0%}. "
        f"{N_PATHS:,} paths. Premium = {premium:.4f}.",
        "",
        f"Fitted log-log slope of std vs N: **{slope:.3f}** (theory: -0.5).",
        "",
        "| Rebalances | Frequency | Mean P&L | Std P&L | Std % of premium "
        "| Derman-Kamal std | 1st pct P&L | corr(P&L, gamma attribution) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['n']} | {row['label']} | {row['mean']:+.4f} | "
            f"{row['std']:.4f} | {row['std_pct']:.1f}% | {row['dk']:.4f} | "
            f"{row['p01']:+.3f} | {row['corr']:.3f} |")
    (OUT / "stage2_summary.md").write_text("\n".join(lines) + "\n")


def chart_error_vs_frequency(ns, stds, vega0, slope) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    grid = np.geomspace(ns.min(), ns.max(), 100)
    theory = np.sqrt(np.pi / 4) * vega0 * SIGMA / np.sqrt(grid)
    ax.plot(grid, theory, color=ORANGE, lw=2, label="Derman–Kamal: √(π/4)·vega·σ/√N")
    ax.plot(ns, stds, "o", color=BLUE, ms=8, mec=SURFACE, mew=2,
            label=f"Simulated (fitted slope {slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    yticks = [0.15, 0.25, 0.5, 1.0, 1.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks])
    ax.set_xlabel("Rebalances over the option's life (log scale)")
    ax.set_ylabel("Std of hedged P&L (log scale)")
    ax.set_title("Hedging error falls like 1/√N: 4x the trades halves the risk", loc="left")
    ax.annotate("daily", (63, stds[list(ns).index(63)]), textcoords="offset points",
                xytext=(8, 6), color=INK_2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_error_vs_frequency.png", dpi=160)
    plt.close(fig)


def chart_distributions(results) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(-4, 4, 121)
    for n, color in ((12, ORANGE), (63, BLUE), (504, AQUA)):
        ax.hist(results[n].pnl, bins=bins, histtype="step", lw=2, color=color,
                density=True, label=f"{FREQUENCIES[n]} (N={n}), std {results[n].std:.2f}")
    ax.set_xlabel("Hedged P&L at expiry")
    ax.set_ylabel("Density")
    ax.set_title("Same paths, same model: only the hedging frequency differs", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_pnl_distributions.png", dpi=160)
    plt.close(fig)


def chart_attribution(res) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sample = slice(0, 4000)
    ax.scatter(res.gamma_pnl[sample], res.pnl[sample], s=8, color=BLUE,
               alpha=0.35, linewidths=0)
    lim = np.percentile(np.abs(np.concatenate([res.pnl, res.gamma_pnl])), 99.8)
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=1, ls="--")
    ax.text(lim * 0.95, lim * 0.55, "y = x\n(perfect attribution)", color=INK_2,
            ha="right", va="top")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    corr = np.corrcoef(res.pnl, res.gamma_pnl)[0, 1]
    ax.set_xlabel("Gamma attribution: Σ −½ Γ S² (R² − σ²dt)")
    ax.set_ylabel("Actual hedged P&L")
    ax.set_title(f"Daily hedging: gamma explains the P&L (ρ = {corr:.3f})", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_gamma_attribution.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
