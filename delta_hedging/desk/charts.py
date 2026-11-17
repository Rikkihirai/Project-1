"""Matplotlib charts for the desk, in the project's house style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pricing.heston import heston_smile  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA, INK_2, "#8a5fd6", "#c9a227"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "legend.frameon": False,
})


def _figure(height: float = 3.6, rows: int = 1):
    fig, axes = plt.subplots(rows, 1, figsize=(8.0, height), sharex=rows > 1)
    return fig, axes


def smiles(quotes_by_expiry: dict, spot: float, r: float, q: float, fit=None):
    """Market implied vols by strike (dots) and the Heston fit (lines)."""
    fig, ax = _figure(4.2)
    for color, (expiry, quotes) in zip(SERIES, quotes_by_expiry.items()):
        if quotes.empty:
            continue
        ax.plot(quotes["strike"], 100 * quotes["iv"], "o", color=color, ms=4,
                mec=SURFACE, mew=0.6, label=f"{expiry:%d %b %Y}")
        if fit is not None:
            grid = np.linspace(quotes["strike"].min(), quotes["strike"].max(), 60)
            model = heston_smile(spot, grid, quotes["T"].iloc[0], r, fit.params, q)
            ax.plot(grid, 100 * model, color=color, lw=1.4)
    ax.axvline(spot, color=INK_2, lw=0.8, ls="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied vol (%)")
    title = "Implied vol by strike" + (" (dots: market, lines: Heston fit)" if fit else "")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    return fig


def term_structure(term):
    fig, ax = _figure(3.0)
    ax.plot(term["T"] * 365, 100 * term["atm_iv"], "o-", color=BLUE, ms=5,
            mec=SURFACE, mew=1)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("ATM implied vol (%)")
    ax.set_title("At-the-money implied vol by expiry", loc="left")
    fig.tight_layout()
    return fig


def vol_history(close, vols: dict, implied=None, implied_history=None):
    """Price on top; rolling realised vols and implied vol below."""
    fig, (top, bottom) = _figure(5.2, rows=2)
    top.plot(close.index, close, color=INK, lw=1)
    top.set_ylabel("Price")
    top.set_title("Price and realised volatility", loc="left")
    for color, (label, series) in zip(SERIES, vols.items()):
        bottom.plot(series.index, 100 * series, color=color, lw=1.2, label=label)
    if implied_history is not None:
        bottom.plot(implied_history.index, 100 * implied_history, color=AQUA, lw=1.2,
                    label=f"{implied_history.name} (30-day implied)")
    if implied is not None:
        bottom.axhline(100 * implied, color=INK_2, lw=1.0, ls="--",
                       label=f"ATM implied now ({implied:.1%})")
    bottom.set_ylabel("Annualised vol (%)")
    bottom.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    return fig


def cone(cone_table, term=None):
    """Realised-vol percentiles by window, with implied vols on top."""
    fig, ax = _figure(3.8)
    x = cone_table["window"]
    ax.fill_between(x, 100 * cone_table["p05"], 100 * cone_table["p95"], color=BLUE,
                    alpha=0.12, lw=0, label="5th to 95th percentile")
    ax.fill_between(x, 100 * cone_table["p25"], 100 * cone_table["p75"], color=BLUE,
                    alpha=0.25, lw=0, label="25th to 75th percentile")
    ax.plot(x, 100 * cone_table["median"], color=BLUE, lw=1.4, label="Median realised")
    ax.plot(x, 100 * cone_table["current"], "o-", color=INK, ms=4, lw=1,
            label="Realised now")
    if term is not None and not term.empty:
        days = term["T"] * 252
        inside = (days >= x.min()) & (days <= x.max())
        ax.plot(days[inside], 100 * term["atm_iv"][inside], "o-", color=ORANGE, ms=5,
                lw=1.2, label="ATM implied now")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    ax.set_xlabel("Window / time to expiry (trading days, log scale)")
    ax.set_ylabel("Annualised vol (%)")
    ax.set_title("Volatility cone: is implied high or low against history?", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3)
    fig.tight_layout()
    return fig


def single_backtest(days, strike: float):
    fig, (top, middle, bottom) = _figure(6.4, rows=3)
    top.plot(days.index, days["spot"], color=INK, lw=1.2)
    top.axhline(strike, color=INK_2, lw=0.8, ls="--")
    top.set_ylabel("Stock (dashed: strike)")
    top.set_title("One option sold and delta-hedged on real prices", loc="left")
    middle.step(days.index, days["delta"], color=BLUE, lw=1.2, where="post")
    middle.set_ylabel("Shares held (delta)")
    bottom.plot(days.index, days["pnl"], color=ORANGE, lw=1.4)
    bottom.axhline(0, color=INK_2, lw=0.8)
    bottom.set_ylabel("P&L before costs")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def rolling_backtest(trades):
    fig, (top, bottom) = _figure(5.4, rows=2)
    colors = np.where(trades["pnl"] >= 0, AQUA, ORANGE)
    top.bar(trades.index, trades["pnl"], width=4, color=colors)
    top.axhline(0, color=INK_2, lw=0.8)
    top.set_ylabel("P&L per trade (% of spot)")
    top.set_title("Every trade, by the date the option was sold", loc="left")
    bottom.plot(trades.index, trades["pnl"].cumsum(), color=BLUE, lw=1.4)
    bottom.axhline(0, color=INK_2, lw=0.8)
    bottom.set_ylabel("Cumulative P&L (% of spot)")
    fig.tight_layout()
    return fig


def pnl_vs_vol_gap(trades):
    fig, ax = _figure(3.8)
    gap = 100 * (trades["sale_vol"] - trades["realised_vol"])
    ax.scatter(gap, trades["pnl"], s=10, color=BLUE, alpha=0.5, linewidths=0)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Vol sold at minus vol realised afterwards (points)")
    ax.set_ylabel("P&L per trade (% of spot)")
    ax.set_title("Stage 3 on market prices: the P&L is the vol gap", loc="left")
    fig.tight_layout()
    return fig


def replay(table):
    fig, (top, middle, bottom) = _figure(6.4, rows=3)
    top.plot(table.index, table["option"], color=INK, lw=1.2)
    top.set_ylabel("Option close")
    top.set_title("A listed contract sold at its market price, hedged daily", loc="left")
    middle.plot(table.index, 100 * table["iv"], color=BLUE, lw=1.2)
    middle.set_ylabel("Implied vol (%)")
    bottom.plot(table.index, table["pnl"], color=ORANGE, lw=1.4)
    bottom.axhline(0, color=INK_2, lw=0.8)
    bottom.set_ylabel("Hedged P&L per share")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---- interactive charts (hover to read values) -----------------------------------

def hover_lines(wide, x: str, x_title: str, y_title: str, markers: dict | None = None,
                dots=None, height: int = 340):
    """Lines that share an x axis, with a hover ruler that reads off every line.

    wide     DataFrame: one column for x, one column per line.
    markers  {label: x position} drawn as dashed vertical lines.
    dots     optional DataFrame with columns x, "value" and "label" drawn as points
             (market quotes on top of model lines).
    """
    import altair as alt
    names = [c for c in wide.columns if c != x]
    long = wide.melt(x, var_name="curve", value_name="value").dropna()
    colour = alt.Color("curve:N", title=None, sort=names,
                       scale=alt.Scale(domain=names, range=SERIES[:len(names)]),
                       legend=alt.Legend(orient="top"))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("value:Q", title=y_title), color=colour)

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    tips = [alt.Tooltip(f"{x}:Q", title=x_title, format=",.2f")] + [
        alt.Tooltip(f"{name}:Q", format=",.4f") for name in names]
    ruler = alt.Chart(wide).mark_rule(color=INK_2).encode(
        x=f"{x}:Q", tooltip=tips,
        opacity=alt.condition(nearest, alt.value(0.6), alt.value(0.0))).add_params(nearest)
    layers = [lines, ruler]

    for label, position in (markers or {}).items():
        frame = wide.iloc[:1].assign(**{x: position, "label": label})
        layers.append(alt.Chart(frame).mark_rule(color=INK_2, strokeDash=[4, 4]).encode(x=f"{x}:Q"))
        layers.append(alt.Chart(frame).mark_text(align="left", dx=4, dy=-4, color=INK_2).encode(
            x=f"{x}:Q", y=alt.value(12), text="label:N"))
    if dots is not None and not dots.empty:
        layers.append(alt.Chart(dots).mark_circle(size=70, color=INK, opacity=0.9).encode(
            x=f"{x}:Q", y="value:Q",
            tooltip=[alt.Tooltip("label:N", title="Market"),
                     alt.Tooltip("value:Q", title="Mid price", format=",.2f")]))
    return alt.layer(*layers).properties(height=height).configure_view(
        strokeWidth=0).configure(background=SURFACE)
