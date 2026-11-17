"""Stage 3: what happens when implied vol is not the vol the stock realises?

A dealer sells a 3-month at-the-money call at 30% implied vol. The stock
then moves with 20% realised vol, so the option was sold too expensive and
the dealer should make money. How much, and how smoothly, depends on which
vol goes into the hedge delta (Ahmad & Wilmott, 2005):

  implied-vol delta   P&L = integral of 1/2 Gamma_i S^2 (s_i^2 - s_r^2) dt.
                      Accrues smoothly and is never negative, but the total
                      depends on the path (gamma is large only near K).

  realised-vol delta  P&L = (V(s_i) - V(s_r)) e^{rT} on every path, but the
                      mark-to-market wobbles on the way there.

Both hedges are run on the same simulated paths, so the only difference is
the vol used for delta. The derivations are in hedging/hedge.py.

Run from the project folder:

    python3 experiments/stage3_implied_vs_realised.py

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
from hedging.hedge import delta_hedge, mark_to_market, variance_edge_pnl  # noqa: E402
from hedging.paths import simulate_gbm  # noqa: E402

OUT = ROOT / "results"

# ---- Set-up --------------------------------------------------------------
S0, K, T, r, MU = 100.0, 100.0, 0.25, 0.03, 0.08
SIGMA_IMP, SIGMA_REAL = 0.30, 0.20   # sold at 30%, stock realises 20%
FINE_STEPS = 504                     # ~hourly: close to continuous hedging
DAILY_STEPS = 63
N_PATHS = 20_000
SWEEP_PATHS = 5_000                 # per setting in the two sweeps (keeps the run ~1 min)
SEED = 2026

REAL_VOL_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DRIFT_SWEEP = (-0.60, -0.30, 0.0, 0.08, 0.30, 0.60)

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
IMP_LABEL = "Hedge at implied vol"
REAL_LABEL = "Hedge at realised vol"


def locked_in(sigma_imp: float, sigma_real: float) -> float:
    """(V(s_i) - V(s_r)) e^{rT}: the premium over fair value, grown to T."""
    edge = bs.price(S0, K, T, r, sigma_imp) - bs.price(S0, K, T, r, sigma_real)
    return float(edge) * np.exp(r * T)


def both_hedges(paths, sigma_imp=SIGMA_IMP, sigma_real=SIGMA_REAL):
    """Sell at sigma_imp, then hedge the same paths with each delta."""
    implied = delta_hedge(paths, K, T, r, sigma=sigma_imp)
    realised = delta_hedge(paths, K, T, r, sigma=sigma_real,
                           premium_sigma=sigma_imp)
    return implied, realised


def main() -> None:
    OUT.mkdir(exist_ok=True)
    paths = simulate_gbm(S0, MU, SIGMA_REAL, T, FINE_STEPS, N_PATHS, seed=SEED,
                         antithetic=True)
    target = locked_in(SIGMA_IMP, SIGMA_REAL)

    # 1. Headline comparison at hourly and daily hedging (same paths).
    rows = []
    for n, label in ((FINE_STEPS, "~hourly"), (DAILY_STEPS, "daily")):
        P = paths[:, :: FINE_STEPS // n]
        theory = variance_edge_pnl(P, K, T, r, SIGMA_IMP, SIGMA_REAL)
        for name, res in zip((IMP_LABEL, REAL_LABEL), both_hedges(P)):
            rows.append(dict(
                freq=label, name=name, mean=res.mean, std=res.std,
                lo=float(np.percentile(res.pnl, 1)),
                hi=float(np.percentile(res.pnl, 99)),
                win=float(np.mean(res.pnl > 0)),
                corr=(float(np.corrcoef(res.pnl, theory)[0, 1])
                      if name == IMP_LABEL else None)))

    # 2. Vary the realised vol. Same seed = same random draws, rescaled.
    vol_rows = []
    for s_real in REAL_VOL_SWEEP:
        P = simulate_gbm(S0, MU, s_real, T, FINE_STEPS, SWEEP_PATHS, seed=SEED,
                         antithetic=True)
        implied, realised = both_hedges(P, sigma_real=s_real)
        vol_rows.append(dict(
            s_real=s_real, target=locked_in(SIGMA_IMP, s_real),
            vega_approx=float(bs.vega(S0, K, T, r, SIGMA_IMP))
            * (SIGMA_IMP - s_real) * np.exp(r * T),
            imp_mean=implied.mean, imp_std=implied.std,
            real_mean=realised.mean, real_std=realised.std))

    # 3. Vary the real-world drift.
    drift_rows = []
    for mu in DRIFT_SWEEP:
        P = simulate_gbm(S0, mu, SIGMA_REAL, T, FINE_STEPS, SWEEP_PATHS, seed=SEED,
                         antithetic=True)
        implied, realised = both_hedges(P)
        drift_rows.append(dict(mu=mu, imp_mean=implied.mean, imp_std=implied.std,
                               real_mean=realised.mean, real_std=realised.std))

    implied, realised = both_hedges(paths)
    write_table(rows, vol_rows, drift_rows, target)
    chart_mtm_paths(paths, target)
    chart_distributions(implied, realised, target)
    chart_pnl_vs_final_price(paths, implied, realised, target)
    chart_vol_sweep(vol_rows)

    print(f"Sold at {SIGMA_IMP:.0%}, realised {SIGMA_REAL:.0%}. "
          f"Locked-in target (V_i - V_r) e^(rT) = {target:.4f}")
    for row in rows:
        print(f"{row['freq']:<8} {row['name']:<22} mean={row['mean']:.4f} "
              f"std={row['std']:.4f}  1%..99%=[{row['lo']:.2f}, {row['hi']:.2f}]")
    for row in drift_rows:
        print(f"mu={row['mu']:+.0%}  implied-hedge mean={row['imp_mean']:.3f}  "
              f"realised-hedge mean={row['real_mean']:.3f}")


def write_table(rows, vol_rows, drift_rows, target) -> None:
    v_imp = float(bs.price(S0, K, T, r, SIGMA_IMP))
    v_real = float(bs.price(S0, K, T, r, SIGMA_REAL))
    lines = [
        "# Stage 3 results: implied vol vs realised vol",
        "",
        f"Short 3-month ATM call, S0=K={S0:.0f}, r={r:.0%}. Sold at "
        f"{SIGMA_IMP:.0%} implied vol (premium {v_imp:.4f}); the stock realises "
        f"{SIGMA_REAL:.0%} (fair value {v_real:.4f}). Real-world drift {MU:.0%}, "
        f"{N_PATHS:,} paths, the same paths for both hedges.",
        "",
        f"Locked-in target (V_imp - V_real) e^(rT) = **{target:.4f}**.",
        "",
        "## Which vol goes into the delta?",
        "",
        "| Rebalancing | Hedge delta | Mean P&L | Std P&L | 1st pct | 99th pct "
        "| Paths with P&L > 0 | corr with ½∫ΓS²(σi²−σr²)dt |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        corr = "" if row["corr"] is None else f"{row['corr']:.3f}"
        lines.append(
            f"| {row['freq']} | {row['name'].replace('Hedge at ', '')} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | {row['lo']:.3f} | "
            f"{row['hi']:.3f} | {row['win']:.2%} | {corr} |")
    lines += [
        "",
        f"## Realised vol varies (sold at {SIGMA_IMP:.0%}, ~hourly hedging, "
        f"{SWEEP_PATHS:,} paths per row)",
        "",
        "| Realised vol | (V_imp − V_real) e^(rT) | vega × (σi − σr) e^(rT) "
        "| Implied hedge: mean | std | Realised hedge: mean | std |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in vol_rows:
        lines.append(
            f"| {row['s_real']:.0%} | {row['target']:+.4f} | "
            f"{row['vega_approx']:+.4f} | {row['imp_mean']:+.4f} | "
            f"{row['imp_std']:.4f} | {row['real_mean']:+.4f} | "
            f"{row['real_std']:.4f} |")
    lines += [
        "",
        f"## Real-world drift varies (sold at {SIGMA_IMP:.0%}, realised "
        f"{SIGMA_REAL:.0%}, ~hourly hedging, {SWEEP_PATHS:,} paths per row)",
        "",
        "| Drift μ | Implied hedge: mean | std | Realised hedge: mean | std |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in drift_rows:
        lines.append(
            f"| {row['mu']:+.0%} | {row['imp_mean']:.4f} | {row['imp_std']:.4f} | "
            f"{row['real_mean']:.4f} | {row['real_std']:.4f} |")
    (OUT / "stage3_summary.md").write_text("\n".join(lines) + "\n")


def chart_mtm_paths(paths, target) -> None:
    """Mark-to-market P&L through time for a handful of paths."""
    sample = paths[:12]
    days = np.arange(FINE_STEPS + 1) * DAILY_STEPS / FINE_STEPS
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, vol, color, title in (
            (axes[0], SIGMA_IMP, BLUE, f"{IMP_LABEL}: smooth, ends anywhere"),
            (axes[1], SIGMA_REAL, ORANGE, f"{REAL_LABEL}: bumpy, ends near one point")):
        mtm = mark_to_market(sample, K, T, r, sigma=vol, premium_sigma=SIGMA_IMP)
        ax.plot(days, mtm.T, color=color, lw=1.0, alpha=0.75)
        ax.axhline(target, color=INK_2, lw=1, ls="--")
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Trading days since the sale")
    axes[0].set_ylabel("Mark-to-market P&L (option marked at implied vol)")
    axes[1].text(1, target, f"(V_imp − V_real) e^(rT) = {target:.2f}",
                 color=INK_2, va="bottom")
    fig.suptitle("Same 12 paths, ~hourly hedging: only the vol in the delta differs",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "stage3_mtm_paths.png", dpi=160)
    plt.close(fig)


def chart_distributions(implied, realised, target) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 4, 161)
    for res, color, label in ((implied, BLUE, IMP_LABEL), (realised, ORANGE, REAL_LABEL)):
        ax.hist(res.pnl, bins=bins, histtype="step", lw=2, color=color, density=True,
                label=f"{label}: mean {res.mean:.2f}, std {res.std:.2f}")
    ax.axvline(target, color=INK_2, lw=1, ls="--")
    ax.text(target + 0.05, ax.get_ylim()[1] * 0.75,
            f"(V_imp − V_real) e^(rT)\n= {target:.2f}", color=INK_2, va="top")
    ax.set_xlabel("Hedged P&L at expiry")
    ax.set_ylabel("Density")
    ax.set_title("Same average profit, very different certainty (~hourly hedging)",
                 loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage3_pnl_distributions.png", dpi=160)
    plt.close(fig)


def chart_pnl_vs_final_price(paths, implied, realised, target) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sample = slice(0, 5000)
    S_T = paths[sample, -1]
    ax.scatter(S_T, implied.pnl[sample], s=7, color=BLUE, alpha=0.35,
               linewidths=0, label=IMP_LABEL)
    ax.scatter(S_T, realised.pnl[sample], s=7, color=ORANGE, alpha=0.35,
               linewidths=0, label=REAL_LABEL)
    ax.axvline(K, color=INK_2, lw=1, ls="--")
    ax.text(K + 0.5, 0.15, f"strike {K:.0f}", color=INK_2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Stock price at expiry")
    ax.set_ylabel("Hedged P&L at expiry")
    ax.set_title("Implied-vol hedge earns most when the stock stays near the strike",
                 loc="left")
    legend = ax.legend(loc="upper right", markerscale=2.5)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.tight_layout()
    fig.savefig(OUT / "stage3_pnl_vs_final_price.png", dpi=160)
    plt.close(fig)


def chart_vol_sweep(vol_rows) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    grid = np.linspace(REAL_VOL_SWEEP[0], REAL_VOL_SWEEP[-1], 100)
    ax.plot(100 * grid, [locked_in(SIGMA_IMP, s) for s in grid], color=INK_2, lw=1.5,
            label="Theory: (V_imp − V_real) e^(rT)")
    x = 100 * np.array([row["s_real"] for row in vol_rows])
    ax.errorbar(x - 0.4, [row["imp_mean"] for row in vol_rows],
                yerr=[row["imp_std"] for row in vol_rows], fmt="o", color=BLUE,
                ms=6, capsize=3, label=f"{IMP_LABEL} (mean ± 1 std)")
    ax.errorbar(x + 0.4, [row["real_mean"] for row in vol_rows],
                yerr=[row["real_std"] for row in vol_rows], fmt="o", color=ORANGE,
                ms=6, capsize=3, label=f"{REAL_LABEL} (mean ± 1 std)")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axvline(100 * SIGMA_IMP, color=INK_2, lw=1, ls="--")
    ax.text(100 * SIGMA_IMP + 0.5, ax.get_ylim()[0] * 0.92,
            f"sold at {SIGMA_IMP:.0%}", color=INK_2)
    ax.set_xlabel("Realised vol of the stock (%)")
    ax.set_ylabel("Hedged P&L at expiry")
    ax.set_title("A short option is a bet that realised vol ends below implied",
                 loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage3_pnl_vs_realised_vol.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
