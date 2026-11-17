"""Stage 5b: the pricing toolkit, checked against itself and known values.

Four pricing methods that go beyond the Black-Scholes formula:

  1. Binomial tree and Longstaff-Schwartz Monte Carlo (American options)
  2. Finite-difference PDE solver (European, American, barriers)
  3. Heston by Fourier transform (the smile, fast and without noise)
  4. Calibration of Heston to a quoted smile

For each: how accurate is it, how fast, and does it agree with the others?
The methods and their maths are documented in pricing/*.py.

Run from the project folder:

    python3 experiments/stage5b_pricing_toolkit.py

Writes charts and a summary table to results/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hedging import bs  # noqa: E402
from hedging.paths import simulate_gbm, simulate_heston  # noqa: E402
from pricing.american import binomial_price, longstaff_schwartz  # noqa: E402
from pricing.calibration import calibrate_heston  # noqa: E402
from pricing.heston import HestonParams, heston_price, heston_smile  # noqa: E402
from pricing.pde import down_and_out_call, fd_price, fd_solve  # noqa: E402

OUT = ROOT / "results"

# ---- Set-up --------------------------------------------------------------
S0, K, T, r, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.20      # the study's option
# American put from Longstaff & Schwartz (2001), Table 1.
PUT = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.20)
HESTON = HestonParams(v0=0.04, kappa=3.0, theta=0.04, xi=0.4, rho=-0.7)   # as stage 5
SMILE_STRIKES = np.arange(80.0, 121.0, 5.0)

# "Market" for the calibration exercise: quotes generated from known Heston
# parameters plus noise of 0.2 vol points, like a bid-ask spread.
MARKET = HestonParams(v0=0.05, kappa=2.5, theta=0.035, xi=0.45, rho=-0.65)
QUOTE_MATURITIES = (1 / 12, 0.25, 0.5, 1.0)
QUOTE_NOISE = 0.002
SEED = 2026

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


def timed(function, *args, **kwargs):
    """Return (result, seconds taken)."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def convergence() -> dict:
    """Error against the Black-Scholes formula as the grid is refined."""
    exact = float(bs.price(S0, K, T, r, SIGMA))
    sizes = np.array([25, 50, 100, 200, 400, 800, 1600])
    tree = [abs(binomial_price(S0, K, T, r, SIGMA, american=False, n_steps=int(n))
                - exact) for n in sizes]
    pde = [abs(fd_price(S0, K, T, r, SIGMA, n_space=int(n), n_time=int(n)) - exact)
           for n in sizes]
    return dict(sizes=sizes, tree=np.array(tree), pde=np.array(pde))


def american_put_methods() -> list:
    """The same American put priced three ways."""
    c = PUT
    european = float(bs.price(c["S0"], c["K"], c["T"], c["r"], c["sigma"], kind="put"))
    rows = [dict(method="Black-Scholes formula (European, for reference)",
                 price=european, se=None, seconds=None)]
    price, seconds = timed(binomial_price, c["S0"], c["K"], c["T"], c["r"],
                           c["sigma"], kind="put", n_steps=2000)
    rows.append(dict(method="Binomial tree, 2,000 steps", price=price, se=None,
                     seconds=seconds))
    price, seconds = timed(fd_price, c["S0"], c["K"], c["T"], c["r"], c["sigma"],
                           kind="put", american=True)
    rows.append(dict(method="PDE solver, 400 x 400 grid", price=price, se=None,
                     seconds=seconds))
    for n_steps in (50, 250):
        paths = simulate_gbm(c["S0"], c["r"], c["sigma"], c["T"], n_steps, 100_000,
                             seed=SEED, antithetic=True)
        (price, se), seconds = timed(longstaff_schwartz, paths, c["K"], c["T"], c["r"])
        rows.append(dict(
            method=f"Longstaff-Schwartz, 100,000 paths, {n_steps} exercise dates",
            price=price, se=se, seconds=seconds))
    return rows


def barrier_check() -> list:
    rows = []
    for barrier in (80.0, 90.0, 95.0, 99.0):
        exact = down_and_out_call(S0, K, T, r, SIGMA, barrier)
        pde = fd_price(S0, K, T, r, SIGMA, lower_barrier=barrier, n_space=800,
                       n_time=800)
        rows.append(dict(barrier=barrier, exact=exact, pde=pde))
    return rows


def heston_vs_monte_carlo() -> dict:
    """Fourier smile against the Monte Carlo smile of stage 5."""
    fourier, fourier_seconds = timed(heston_smile, S0, SMILE_STRIKES, T, r, HESTON)

    def monte_carlo():
        S_T = simulate_heston(S0, r, HESTON.v0, HESTON.kappa, HESTON.theta,
                              HESTON.xi, HESTON.rho, T, 63, 40_000, seed=SEED + 1,
                              antithetic=True)[0][:, -1]
        vols = []
        for strike in SMILE_STRIKES:
            kind = "put" if strike < S0 else "call"
            price = np.exp(-r * T) * bs.payoff(S_T, strike, kind).mean()
            vols.append(bs.implied_vol(price, S0, strike, T, r, kind=kind))
        return np.array(vols)

    mc, mc_seconds = timed(monte_carlo)
    _, price_seconds = timed(heston_price, S0, SMILE_STRIKES, T, r, HESTON)
    return dict(fourier=fourier, mc=mc, fourier_seconds=fourier_seconds,
                mc_seconds=mc_seconds, price_seconds=price_seconds)


def calibration() -> dict:
    """Fit Heston to noisy quotes generated from known parameters."""
    rng = np.random.default_rng(SEED)
    strikes = np.tile(SMILE_STRIKES, len(QUOTE_MATURITIES))
    maturities = np.repeat(QUOTE_MATURITIES, len(SMILE_STRIKES))
    clean = np.concatenate([heston_smile(S0, SMILE_STRIKES, t, r, MARKET)
                            for t in QUOTE_MATURITIES])
    quotes = clean + rng.normal(0.0, QUOTE_NOISE, clean.shape)
    fit, seconds = timed(calibrate_heston, S0, strikes, maturities, quotes, r)
    return dict(strikes=strikes, maturities=maturities, quotes=quotes, fit=fit,
                seconds=seconds)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    conv = convergence()
    put_rows = american_put_methods()
    barrier_rows = barrier_check()
    smile = heston_vs_monte_carlo()
    calib = calibration()

    write_table(conv, put_rows, barrier_rows, smile, calib)
    chart_convergence(conv)
    chart_early_exercise()
    chart_heston_smile(smile)
    chart_calibration(calib)

    for row in put_rows:
        print(f"{row['method']:<58} {row['price']:.4f}")
    print(f"Heston smile: Fourier {smile['fourier_seconds'] * 1e3:.0f} ms, "
          f"Monte Carlo {smile['mc_seconds']:.1f} s; max gap "
          f"{np.abs(smile['fourier'] - smile['mc']).max():.4f}")
    print(f"Calibration: {calib['fit'].params}, rmse {calib['fit'].rmse_vol:.4f}, "
          f"{calib['seconds']:.2f} s")


def write_table(conv, put_rows, barrier_rows, smile, calib) -> None:
    c = PUT
    lines = [
        "# Stage 5b results: pricing toolkit",
        "",
        "## Convergence to the Black-Scholes formula",
        "",
        f"European call, S0=K={S0:.0f}, 3 months, r={r:.0%}, vol {SIGMA:.0%}. "
        f"Exact price {float(bs.price(S0, K, T, r, SIGMA)):.6f}. For the PDE "
        "solver n is both the number of price steps and of time steps.",
        "",
        "| n | Binomial tree error | PDE solver error |",
        "|---:|---:|---:|",
    ]
    for n, tree, pde in zip(conv["sizes"], conv["tree"], conv["pde"]):
        lines.append(f"| {n} | {tree:.6f} | {pde:.6f} |")
    lines += [
        "",
        "## One American put, three methods",
        "",
        f"1-year put, S0={c['S0']:.0f}, K={c['K']:.0f}, r={c['r']:.0%}, vol "
        f"{c['sigma']:.0%} (Longstaff & Schwartz 2001, Table 1: 4.478 with 50 "
        "exercise dates; about 4.486 with continuous exercise).",
        "",
        "| Method | Price | Std error | Time |",
        "|---|---:|---:|---:|",
    ]
    for row in put_rows:
        se = "" if row["se"] is None else f"{row['se']:.4f}"
        seconds = "" if row["seconds"] is None else f"{row['seconds']:.2f} s"
        lines.append(f"| {row['method']} | {row['price']:.4f} | {se} | {seconds} |")
    lines += [
        "",
        "## Down-and-out call: PDE solver against the closed form",
        "",
        f"The study's call (S0=K={S0:.0f}), knocked out if the stock touches the barrier.",
        "",
        "| Barrier | Closed form | PDE solver | Difference |",
        "|---:|---:|---:|---:|",
    ]
    for row in barrier_rows:
        lines.append(f"| {row['barrier']:.0f} | {row['exact']:.4f} | {row['pde']:.4f} | "
                     f"{row['pde'] - row['exact']:+.4f} |")
    lines += [
        "",
        "## Heston smile: Fourier transform against Monte Carlo",
        "",
        f"Stage 5 parameters, 3 months. Fourier: {smile['fourier_seconds'] * 1e3:.0f} ms "
        f"for the whole smile including the implied-vol inversion "
        f"({smile['price_seconds'] * 1e3:.1f} ms for the nine prices alone). "
        f"Monte Carlo with 40,000 paths: {smile['mc_seconds']:.1f} s.",
        "",
        "| Strike | " + " | ".join(f"{k:.0f}" for k in SMILE_STRIKES) + " |",
        "|---|" + "---:|" * len(SMILE_STRIKES),
        "| Fourier | " + " | ".join(f"{v:.2%}" for v in smile["fourier"]) + " |",
        "| Monte Carlo | " + " | ".join(f"{v:.2%}" for v in smile["mc"]) + " |",
        "",
        "## Calibration to noisy quotes",
        "",
        f"{len(calib['quotes'])} quotes (9 strikes x 4 maturities) generated from "
        f"known parameters plus noise of {QUOTE_NOISE:.1%} vol. Fit error "
        f"{calib['fit'].rmse_vol:.2%} vol, {calib['fit'].n_evaluations} smile "
        f"evaluations, {calib['seconds']:.2f} s.",
        "",
        "| Parameter | Meaning | True | Fitted |",
        "|---|---|---:|---:|",
    ]
    meanings = dict(v0="variance today", kappa="mean-reversion speed",
                    theta="long-run variance", xi="vol of vol", rho="correlation")
    for name, meaning in meanings.items():
        lines.append(f"| {name} | {meaning} | {getattr(MARKET, name):.4f} | "
                     f"{getattr(calib['fit'].params, name):.4f} |")
    (OUT / "stage5b_summary.md").write_text("\n".join(lines) + "\n")


def chart_convergence(conv) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n = conv["sizes"]
    ax.plot(n, conv["tree"], "o-", color=BLUE, ms=7, mec=SURFACE, mew=1.5,
            label="Binomial tree (error ~ 1/n)")
    ax.plot(n, conv["pde"], "o-", color=ORANGE, ms=7, mec=SURFACE, mew=1.5,
            label="PDE solver, Crank-Nicolson (error ~ 1/n²)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlabel("Grid size n (log scale)")
    ax.set_ylabel("Absolute pricing error (log scale)")
    ax.set_title("Both grids converge to Black-Scholes; the PDE solver faster",
                 loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "stage5b_convergence.png", dpi=160)
    plt.close(fig)


def chart_early_exercise() -> None:
    """American and European put values across stock prices (PDE grid)."""
    c = PUT
    s_max = 2.5 * c["K"]
    common = dict(K=c["K"], T=c["T"], r=c["r"], sigma=c["sigma"], s_max=s_max,
                  kind="put", n_space=800, n_time=800)
    S, american = fd_solve(american=True, **common)
    _, european = fd_solve(american=False, **common)
    intrinsic = np.maximum(c["K"] - S, 0.0)
    exercise_below = S[np.flatnonzero(american > intrinsic + 1e-6)[0]]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(S, intrinsic, color=INK_2, lw=1.2, ls="--", label="Exercise now: K − S")
    ax.plot(S, european, color=BLUE, lw=2, label="European put")
    ax.plot(S, american, color=ORANGE, lw=2, label="American put")
    ax.axvline(exercise_below, color=INK_2, lw=0.8)
    ax.text(exercise_below - 0.4, 1.0, f"exercise below {exercise_below:.1f}",
            color=INK_2, ha="right")
    ax.set_xlim(20, 60)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Stock price")
    ax.set_ylabel("Put value (K = 40, 1 year, r = 6%)")
    ax.set_title("A European put can be worth less than exercising; an American can't",
                 loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage5b_early_exercise.png", dpi=160)
    plt.close(fig)


def chart_heston_smile(smile) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fine = np.linspace(SMILE_STRIKES[0], SMILE_STRIKES[-1], 81)
    ax.plot(fine, 100 * heston_smile(S0, fine, T, r, HESTON), color=ORANGE, lw=2,
            label=f"Fourier transform ({smile['price_seconds'] * 1e3:.1f} ms for 9 prices)")
    ax.plot(SMILE_STRIKES, 100 * smile["mc"], "o", color=BLUE, ms=7, mec=SURFACE,
            mew=1.5, label=f"Monte Carlo, 40,000 paths ({smile['mc_seconds']:.1f} s)")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied vol, 3 months (%)")
    ax.set_title("Heston smile: same answer as stage 5's simulation, without the noise",
                 loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage5b_heston_smile.png", dpi=160)
    plt.close(fig)


def chart_calibration(calib) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fine = np.linspace(SMILE_STRIKES[0], SMILE_STRIKES[-1], 81)
    labels = {1 / 12: "1 month", 0.25: "3 months", 0.5: "6 months", 1.0: "1 year"}
    colors = {1 / 12: ORANGE, 0.25: BLUE, 0.5: AQUA, 1.0: INK_2}
    for maturity in QUOTE_MATURITIES:
        rows = calib["maturities"] == maturity
        ax.plot(calib["strikes"][rows], 100 * calib["quotes"][rows], "o",
                color=colors[maturity], ms=6, mec=SURFACE, mew=1.2)
        ax.plot(fine, 100 * heston_smile(S0, fine, maturity, r, calib["fit"].params),
                color=colors[maturity], lw=1.8, label=labels[maturity])
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied vol (%)")
    ax.set_title(f"Five parameters fit {len(calib['quotes'])} quotes to within "
                 f"{calib['fit'].rmse_vol:.2%} vol", loc="left")
    ax.legend(loc="upper right", title="Dots: quotes. Lines: fitted Heston")
    fig.tight_layout()
    fig.savefig(OUT / "stage5b_calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
