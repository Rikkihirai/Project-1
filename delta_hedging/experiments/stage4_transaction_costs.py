"""Stage 4: what does hedging cost, and how should the hedger spend it?

A dealer sells a 3-month at-the-money call at the right vol (20%) and
delta-hedges it, but now every trade costs 5 bp of the notional traded.
Three questions:

  1. Time-based hedging. Stage 2 showed risk falls like 1/sqrt(N). Costs
     grow like sqrt(N):  cost ~ c * vega * sqrt(2N / (pi T)).
     So how often should the hedger trade?

  2. Delta-band hedging. Trade only when the position has drifted more
     than a band away from the model delta. Does that buy more risk
     reduction per unit of cost than trading on a clock?

  3. Leland's vol. Selling and hedging at
     s_L = sigma sqrt(1 + sqrt(8/pi) c / (sigma sqrt(dt)))
     should make the dealer whole for the rebalancing costs.

All settings reuse the same 20,000 paths. Derivations: hedging/hedge.py.

Run from the project folder:

    python3 experiments/stage4_transaction_costs.py

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
from hedging.hedge import delta_hedge, leland_vol  # noqa: E402
from hedging.paths import simulate_gbm  # noqa: E402

OUT = ROOT / "results"

# ---- Set-up --------------------------------------------------------------
S0, K, T, r, SIGMA, MU = 100.0, 100.0, 0.25, 0.03, 0.20, 0.08
COST = 0.0005                     # 5 bp of notional per trade, one way
FINE_STEPS = 504                  # ~hourly grid; bands are checked this often
N_PATHS = 20_000
SEED = 2026

FREQUENCIES = {4: "every ~3 weeks", 6: "every ~2 weeks", 12: "~weekly",
               21: "every 3 days", 63: "daily", 126: "twice daily",
               252: "4x daily", 504: "~hourly"}
BANDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
LELAND_FREQUENCIES = (12, 63, 252, 504)

# ---- Chart style (same as stage 2) ---------------------------------------
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


def predicted_rebalancing_cost(n) -> float:
    """c * vega * sqrt(2N / (pi T)).

    Per rebalance the expected cost is c Gamma S^2 sigma sqrt(2 dt / pi).
    Summing N of them and using vega = Gamma S^2 sigma T (a Black-Scholes
    identity, evaluated at t = 0 as in Derman-Kamal) gives the closed form.
    """
    return COST * float(bs.vega(S0, K, T, r, SIGMA)) * np.sqrt(2 * n / (np.pi * T))


def on_and_off_cost(res, paths) -> float:
    """Mean cost of putting the hedge on at t=0 and unwinding it at T."""
    setup = COST * np.abs(res.deltas[:, 0]) * paths[:, 0] * np.exp(r * T)
    unwind = COST * np.abs(res.deltas[:, -1]) * paths[:, -1]
    return float((setup + unwind).mean())


def summarise(res, paths) -> dict:
    fixed = on_and_off_cost(res, paths)
    return dict(mean=res.mean, std=res.std, cost=float(res.costs.mean()),
                rebal_cost=float(res.costs.mean()) - fixed, fixed=fixed,
                trades=float(res.n_rebalances.mean()),
                p01=float(np.percentile(res.pnl, 1)))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    paths = simulate_gbm(S0, MU, SIGMA, T, FINE_STEPS, N_PATHS, seed=SEED,
                         antithetic=True)
    premium = float(bs.price(S0, K, T, r, SIGMA))

    # 1. Time-based hedging at each frequency.
    clock = []
    for n, label in FREQUENCIES.items():
        P = paths[:, :: FINE_STEPS // n]
        row = summarise(delta_hedge(P, K, T, r, SIGMA, cost_rate=COST), P)
        row.update(n=n, label=label, predicted=predicted_rebalancing_cost(n))
        clock.append(row)
    ns = np.array([row["n"] for row in clock])
    slope = np.polyfit(np.log(ns), np.log([row["rebal_cost"] for row in clock]), 1)[0]

    # 2. Delta-band hedging, checked on the fine grid.
    bands = []
    for band in BANDS:
        row = summarise(delta_hedge(paths, K, T, r, SIGMA, cost_rate=COST,
                                    band=band), paths)
        row.update(band=band, clock_std=clock_std_at_cost(clock, row["cost"]))
        bands.append(row)

    # 3. Leland: sell and hedge at the cost-adjusted vol.
    leland = []
    for n in LELAND_FREQUENCIES:
        P = paths[:, :: FINE_STEPS // n]
        s_L = leland_vol(SIGMA, COST, T / n)
        res = delta_hedge(P, K, T, r, s_L, cost_rate=COST)
        row = summarise(res, P)
        naive = next(c for c in clock if c["n"] == n)
        row.update(n=n, label=FREQUENCIES[n], s_L=s_L, premium=res.premium,
                   naive_mean=naive["mean"], naive_std=naive["std"])
        leland.append(row)

    write_table(clock, bands, leland, premium, slope)
    chart_cost_vs_risk(clock, slope)
    chart_frontier(clock, bands)
    chart_leland(leland)

    print(f"Premium {premium:.4f}, cost {COST * 1e4:.0f} bp. "
          f"log-log slope of rebalancing cost vs N: {slope:.3f} (theory +0.5)")
    for row in clock:
        print(f"N={row['n']:>4} cost={row['cost']:.4f} (rebalancing "
              f"{row['rebal_cost']:.4f}, predicted {row['predicted']:.4f}) "
              f"std={row['std']:.4f}")
    for row in bands:
        print(f"band={row['band']:.3f} trades={row['trades']:6.1f} "
              f"cost={row['cost']:.4f} std={row['std']:.4f} "
              f"(clock at same cost: {row['clock_std']:.4f})")
    for row in leland:
        print(f"N={row['n']:>4} Leland vol={row['s_L']:.4f} mean={row['mean']:+.4f} "
              f"(naive {row['naive_mean']:+.4f})")


def clock_std_at_cost(clock, cost) -> float:
    """Risk a clock hedger gets for the same spend (log-log interpolation)."""
    costs = np.log([row["cost"] for row in clock])
    stds = np.log([row["std"] for row in clock])
    return float(np.exp(np.interp(np.log(cost), costs, stds)))


def write_table(clock, bands, leland, premium, slope) -> None:
    lines = [
        "# Stage 4 results: transaction costs",
        "",
        f"Short 3-month ATM call, S0=K={S0:.0f}, r={r:.0%}, sigma={SIGMA:.0%} "
        f"(priced, hedged and realised), drift {MU:.0%}. Cost {COST * 1e4:.0f} bp "
        f"of notional per trade, one way. {N_PATHS:,} paths, the same for every "
        f"row. Premium = {premium:.4f}.",
        "",
        "## Time-based hedging",
        "",
        f"Fitted log-log slope of rebalancing cost vs N: **{slope:+.3f}** "
        "(theory: +0.5). \"Rebalancing cost\" excludes putting the hedge on "
        "and unwinding it; the prediction is c·vega·√(2N/(πT)).",
        "",
        "| Rebalances | Frequency | Mean P&L | Std P&L | Total cost "
        "| Rebalancing cost | Predicted | Cost % of premium | 1st pct P&L |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in clock:
        lines.append(
            f"| {row['n']} | {row['label']} | {row['mean']:+.4f} | {row['std']:.4f} | "
            f"{row['cost']:.4f} | {row['rebal_cost']:.4f} | {row['predicted']:.4f} | "
            f"{100 * row['cost'] / premium:.1f}% | {row['p01']:+.3f} |")
    lines += [
        "",
        f"## Delta-band hedging (checked {FINE_STEPS} times, ~hourly)",
        "",
        "\"Clock std at same cost\" is the risk a time-based hedger gets for "
        "the same total spend (interpolated from the table above).",
        "",
        "| Band (deltas) | Trades per path | Mean P&L | Std P&L | Total cost "
        "| Clock std at same cost | Risk saved |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bands:
        saved = 1 - row["std"] / row["clock_std"]
        lines.append(
            f"| {row['band']:.3f} | {row['trades']:.1f} | {row['mean']:+.4f} | "
            f"{row['std']:.4f} | {row['cost']:.4f} | {row['clock_std']:.4f} | "
            f"{saved:.0%} |")
    lines += [
        "",
        "## Leland's adjusted vol",
        "",
        "Sold and hedged at sigma versus sold and hedged at Leland's vol, both "
        "paying costs. \"On/off cost\" is the part Leland's formula does not "
        "cover: putting the hedge on and unwinding it.",
        "",
        "| Rebalances | Leland vol | Leland premium | Mean P&L at sigma "
        "| Mean P&L at Leland vol | On/off cost | Std at sigma | Std at Leland vol |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leland:
        lines.append(
            f"| {row['n']} | {row['s_L']:.2%} | {row['premium']:.4f} | "
            f"{row['naive_mean']:+.4f} | {row['mean']:+.4f} | {row['fixed']:.4f} | "
            f"{row['naive_std']:.4f} | {row['std']:.4f} |")
    (OUT / "stage4_summary.md").write_text("\n".join(lines) + "\n")


def chart_cost_vs_risk(clock, slope) -> None:
    ns = np.array([row["n"] for row in clock])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    grid = np.geomspace(ns.min(), ns.max(), 100)
    ax.plot(grid, [predicted_rebalancing_cost(n) for n in grid], color=ORANGE,
            lw=1.5, alpha=0.6)
    ax.plot(ns, [row["rebal_cost"] for row in clock], "o", color=ORANGE, ms=8,
            mec=SURFACE, mew=2,
            label=f"Mean rebalancing cost (slope {slope:+.2f}); line: c·vega·√(2N/πT)")
    ax.plot(ns, [row["std"] for row in clock], "o-", color=BLUE, ms=8,
            mec=SURFACE, mew=2, lw=1.5, label="Std of P&L after costs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    yticks = [0.03, 0.1, 0.3, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks])
    ax.set_xlabel("Rebalances over the option's life (log scale)")
    ax.set_ylabel("P&L units (log scale)")
    ax.set_title("Trading more often swaps risk for cost: 1/√N against √N", loc="left")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "stage4_cost_vs_risk.png", dpi=160)
    plt.close(fig)


def chart_frontier(clock, bands) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([row["cost"] for row in clock], [row["std"] for row in clock], "o-",
            color=BLUE, ms=7, mec=SURFACE, mew=1.5, lw=1.5,
            label="Time-based (label = rebalances)")
    for row in clock:
        ax.annotate(str(row["n"]), (row["cost"], row["std"]), color=BLUE,
                    textcoords="offset points", xytext=(6, 5))
    ax.plot([row["cost"] for row in bands], [row["std"] for row in bands], "o-",
            color=ORANGE, ms=7, mec=SURFACE, mew=1.5, lw=1.5,
            label="Delta band (label = band width)")
    for row in bands:
        ax.annotate(f"{row['band']:g}", (row["cost"], row["std"]), color=ORANGE,
                    textcoords="offset points", xytext=(-8, -13), ha="right")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Mean transaction cost paid")
    ax.set_ylabel("Std of P&L after costs")
    ax.set_title("Cost-risk frontier: the band buys more hedge per unit of cost",
                 loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage4_frontier.png", dpi=160)
    plt.close(fig)


def chart_leland(leland) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(leland))
    width = 0.36
    ax.bar(x - width / 2, [row["naive_mean"] for row in leland], width,
           color=BLUE, label="Sold and hedged at σ = 20%")
    ax.bar(x + width / 2, [row["mean"] for row in leland], width, color=ORANGE,
           label="Sold and hedged at Leland's vol")
    ax.plot(x + width / 2, [-row["fixed"] for row in leland], "_", color=INK,
            ms=22, mew=2, label="Cost of putting the hedge on and off")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['label']}\nN={row['n']}, σ_L={row['s_L']:.1%}"
                        for row in leland])
    ax.set_ylabel("Mean P&L after costs")
    ax.set_title("Leland's vol pays for the rebalancing at every frequency", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "stage4_leland.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
