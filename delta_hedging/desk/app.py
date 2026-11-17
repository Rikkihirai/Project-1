"""The options desk: a dashboard over the pricing and hedging engines.

Run from the project folder:

    python3 -m streamlit run desk/app.py

Four screens:
    Chain & pricer        live option chain, one contract priced as American,
                          Greeks, and a spot/vol scenario grid for a position
    Vol surface           smiles by expiry, Heston calibrated to them
    Realised vs implied   vol history and the volatility cone
    Hedging backtest      sell an option in the past and hedge it on real
                          prices: one trade, the whole history, or a real
                          listed contract

This file only lays out the screens. Data comes from desk/data.py, the
calculations from desk/analytics.py, the charts from desk/charts.py.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desk import analytics, charts  # noqa: E402
from desk.data import (INDEX_NAMES, DataError, SyntheticData, available_sources,  # noqa: E402
                       connect, market_today)
from hedging import bs  # noqa: E402

CONTRACT_SIZE = 100          # shares per listed option contract
HISTORY_START = date(2016, 1, 1)
TODAY = market_today()       # the date in New York, where the options trade

st.set_page_config(page_title="Options desk", layout="wide")


# ---- data, cached so that moving a slider does not refetch ---------------------

@st.cache_resource
def open_source(name: str):
    return connect(name)


def source():
    return open_source(st.session_state["source_name"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_implied_history(symbol: str):
    return source().implied_vol_history(symbol)


@st.cache_data(ttl=20, show_spinner=False)
def load_spot(symbol: str) -> float:
    return source().spot(symbol)


@st.cache_data(ttl=3600, show_spinner="Loading price history...")
def load_bars(symbol: str) -> pd.DataFrame:
    return source().daily_bars(symbol, HISTORY_START)


@st.cache_data(ttl=60, show_spinner="Loading option quotes...")
def load_atm_strip(symbol: str, spot: float) -> pd.DataFrame:
    """Near-the-money quotes for every expiry in the next year (small request)."""
    today = TODAY
    return source().option_chain(symbol, today + timedelta(days=2),
                                 today + timedelta(days=400), 0.98 * spot, 1.02 * spot)


@st.cache_data(ttl=30, show_spinner="Loading option quotes...")
def load_expiry(symbol: str, expiry: date, spot: float, width: float) -> pd.DataFrame:
    """All strikes within +/- width of the spot for one expiry."""
    return source().option_chain(symbol, expiry, expiry, (1 - width) * spot,
                                 (1 + width) * spot)


@st.cache_data(ttl=3600, show_spinner="Loading option history...")
def load_option_bars(contract: str, start: date) -> pd.DataFrame:
    return source().option_bars(contract, start)


@st.cache_data(show_spinner="Calibrating Heston...")
def calibrate(enriched: pd.DataFrame, spot: float, r: float, q: float, expiries: tuple):
    return analytics.fit_heston(enriched, spot, r, q, list(expiries))


@st.cache_data(show_spinner="Running backtest...")
def run_rolling(close: pd.Series, **settings) -> pd.DataFrame:
    return analytics.rolling_backtest(close, **settings)


# ---- sidebar ----------------------------------------------------------------------

with st.sidebar:
    st.header("Options desk")
    chosen_source = st.selectbox("Data source", available_sources())
    if st.session_state.get("source_name") != chosen_source:
        st.session_state["source_name"] = chosen_source
        st.cache_data.clear()            # cached quotes belong to the old source
    symbol = st.text_input("Underlying (SPY = S&P 500 ETF, SPX = the index)",
                           "SPY").strip().upper()
    r = st.number_input("Interest rate (%)", 0.0, 15.0, 4.0, 0.25) / 100
    q = st.number_input("Dividend yield (%)", 0.0, 10.0, 1.2, 0.1) / 100
    refresh_seconds = st.select_slider("Auto-refresh quotes", [0, 15, 30, 60], 0,
                                       format_func=lambda s: "off" if s == 0 else f"{s} s")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
    st.divider()
    st.caption(f"**Data:** {source().name}. {source().quote_delay.capitalize()}.")
    if source().name == SyntheticData.name:
        st.info("Showing made-up demo data, not real quotes.")

# Index options (SPX) are European; stock and ETF options (SPY) are American.
AMERICAN = symbol not in INDEX_NAMES
STYLE = "American" if AMERICAN else "European"

try:
    spot = load_spot(symbol)
    strip = analytics.enrich_chain(load_atm_strip(symbol, spot), spot, r, q, TODAY)
    term = analytics.atm_term_structure(strip, spot)
except DataError as error:
    st.error(f"Could not load market data: {error}")
    st.stop()

expiries = sorted(strip["expiry"].unique())
if not expiries:
    st.error(f"No option quotes found for {symbol}.")
    st.stop()


def nearest_expiry(days: int) -> date:
    """The listed expiry closest to `days` calendar days from now."""
    return min(expiries, key=lambda e: abs((e - TODAY).days - days))


def atm_implied(expiry: date):
    row = term[term["expiry"] == expiry]
    return None if row.empty else float(row["atm_iv"].iloc[0])


# ---- top strip -----------------------------------------------------------------------

@st.fragment(run_every=refresh_seconds or None)
def top_strip():
    price = load_spot(symbol)
    one_month = nearest_expiry(30)
    implied = atm_implied(one_month)
    cells = st.columns(4)
    cells[0].metric(symbol, f"{price:,.2f}")
    cells[1].metric(f"ATM implied vol, {one_month:%d %b}",
                    "n/a" if implied is None else f"{implied:.1%}")
    cells[2].metric("Listed expiries (next year)", len(expiries))
    cells[3].metric("Updated", pd.Timestamp.now().strftime("%H:%M:%S"))


top_strip()
# One screen at a time: Streamlit reruns this whole file on every click, so
# only the screen being looked at is computed.
SCREENS = ["Chain & pricer", "Vol surface", "Realised vs implied", "Hedging backtest"]
screen = st.radio("Screen", SCREENS, horizontal=True, label_visibility="collapsed")
st.divider()


def price_history():
    """Daily closes, or None (with a message) if they cannot be loaded."""
    try:
        bars = load_bars(symbol)
    except DataError as error:
        st.error(f"Could not load price history: {error}")
        return None
    if bars.empty:
        st.warning("No price history available.")
        return None
    return bars["close"]


# ---- 1. chain and pricer ----------------------------------------------------------------

GREEK_HELP = {
    "Delta": "Change in the option price per 1.00 move in the stock. Also the number of "
             "shares that hedges one option.",
    "Gamma": "Change in delta per 1.00 move in the stock. High gamma means the hedge "
             "needs frequent adjusting (stage 2).",
    "Vega": "Change in the option price per 1 vol point (1%) change in implied vol.",
    "Theta": "Change in the option price as one calendar day passes, all else equal. "
             "The price of holding gamma.",
}


@st.cache_data(ttl=60, show_spinner=False)
def load_strike(symbol: str, strike: float) -> pd.DataFrame:
    """One strike across every listed expiry."""
    return source().option_chain(symbol, TODAY + timedelta(days=1), None, strike, strike)


def chain_table(enriched: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Calls on the left, strike in the middle, puts on the right."""
    calls = enriched[enriched["kind"] == "call"].set_index("strike")[columns]
    puts = enriched[enriched["kind"] == "put"].set_index("strike")[columns]
    table = calls.add_prefix("call ").join(puts.add_prefix("put "), how="outer")
    table.insert(len(columns), "strike", table.index)
    return table.reset_index(drop=True)


def show_formula(S, K, T, rate, div, vol, kind, quote) -> None:
    """Black-Scholes written out with this contract's numbers."""
    t = analytics.black_scholes_terms(S, K, T, rate, div, vol, kind)
    st.latex(r"d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad "
             r"d_2=d_1-\sigma\sqrt{T}")
    if kind == "call":
        st.latex(r"C = S\,e^{-qT}N(d_1) - K\,e^{-rT}N(d_2)")
        legs = (rf"C = \underbrace{{{t['stock_leg']:.2f}}}_{{S e^{{-qT}} N(d_1)}}"
                rf" - \underbrace{{{t['strike_leg']:.2f}}}_{{K e^{{-rT}} N(d_2)}}"
                rf" = {t['price']:.2f}")
        cdf = rf"N(d_1)={t['N_d1']:.4f},\quad N(d_2)={t['N_d2']:.4f}"
    else:
        st.latex(r"P = K\,e^{-rT}N(-d_2) - S\,e^{-qT}N(-d_1)")
        legs = (rf"P = \underbrace{{{t['strike_leg']:.2f}}}_{{K e^{{-rT}} N(-d_2)}}"
                rf" - \underbrace{{{t['stock_leg']:.2f}}}_{{S e^{{-qT}} N(-d_1)}}"
                rf" = {t['price']:.2f}")
        cdf = rf"N(-d_1)={t['N_d1']:.4f},\quad N(-d_2)={t['N_d2']:.4f}"
    st.markdown("With this contract's numbers:")
    st.latex(rf"S={S:,.2f},\; K={K:,.2f},\; T={T:.4f}\text{{ years}},\; r={rate:.2%},\; "
             rf"q={div:.2%},\; \sigma={vol:.2%}".replace("%", r"\%"))
    st.latex(rf"d_1={t['d1']:.4f},\quad d_2={t['d2']:.4f},\quad {cdf}")
    st.latex(legs)
    st.markdown(
        "- **N(d₂)** is the probability (in the pricing measure) that the option finishes "
        "in the money; **N(d₁)** is the same with the stock as the unit of account.\n"
        "- The first leg is what the stock side is worth, the second what the strike "
        "side costs, both in today's money.\n"
        + (f"- This is the European price. The option is American, and the PDE solver adds "
           f"**{quote['early_exercise']:.3f}** for the right to exercise early, giving "
           f"{quote['price']:.2f}.\n" if AMERICAN else
           "- Index options are European, so this formula is the model price.\n")
        + "- Greeks: Δ = e^(−qT)·N(d₁) for a call, Γ = e^(−qT)·n(d₁)/(Sσ√T), "
          "vega = S·e^(−qT)·n(d₁)·√T, and θ follows from the PDE "
          "θ + (r−q)SΔ + ½σ²S²Γ = rV.")


@st.fragment(run_every=refresh_seconds or None)
def chain_screen():
    a, b, c = st.columns([2, 2, 3])
    expiry = a.selectbox("Expiry", expiries, index=expiries.index(nearest_expiry(30)),
                         format_func=lambda e: f"{e:%d %b %Y}  ({(e - TODAY).days} days)")
    width = b.slider("Strikes within ± of spot (%)", 2, 25, 8) / 100
    columns = c.multiselect("Columns", ["bid", "ask", "iv", "delta", "gamma", "theta", "vega"],
                            default=["bid", "ask", "iv", "delta", "gamma", "theta"])
    price = load_spot(symbol)
    enriched = analytics.enrich_chain(load_expiry(symbol, expiry, price, width),
                                      price, r, q, TODAY)
    if enriched.empty or not columns:
        st.warning("No quotes for this expiry." if columns else "Pick at least one column.")
        return

    table = chain_table(enriched, columns)
    formats = {"iv": "percent", "gamma": "%.4f", "delta": "%.3f", "theta": "%.3f",
               "vega": "%.3f", "bid": "%.2f", "ask": "%.2f"}
    config = {f"{side} {name}": st.column_config.NumberColumn(format=formats[name])
              for side in ("call", "put") for name in columns}
    config["strike"] = st.column_config.NumberColumn(format="%.1f")
    st.caption("Click a row to price that strike below. Theta is per day, vega per vol point.")
    picked = st.dataframe(table, hide_index=True, height=300, use_container_width=True,
                          column_config=config, on_select="rerun",
                          selection_mode="single-row", key=f"chain_{expiry}")

    strikes = table["strike"].tolist()
    rows = picked.selection.rows
    strike = strikes[rows[0]] if rows else min(strikes, key=lambda k: abs(k - price))

    st.subheader(f"Price one contract: {symbol} {expiry:%d %b %Y} {strike:g}")
    a, b = st.columns([1, 3])
    kind = a.radio("Type", ["call", "put"], horizontal=True)
    row = enriched[(enriched["strike"] == strike) & (enriched["kind"] == kind)]
    if row.empty or np.isnan(row["iv"].iloc[0]):
        st.warning("No usable quote for this contract. Pick another row.")
        return
    row = row.iloc[0]
    quantity = b.number_input("Position (contracts, negative = short)", -1000, 1000, -10)

    # The five inputs of the pricing formula. Keys carry the contract so that
    # picking another contract starts again from the market's values.
    tag = f"{row['contract']}"
    st.markdown("**Scenario inputs** (they start at today's market values)")
    cells = st.columns(6)
    s_spot = cells[0].number_input("Stock price", 0.01, 10 * price, float(round(price, 2)),
                                   max(0.01, round(price * 0.005, 2)), key=f"spot_{tag}")
    s_vol = cells[1].number_input("Vol (%)", 1.0, 200.0, float(round(100 * row["iv"], 2)),
                                  0.25, key=f"vol_{tag}") / 100
    market_days = float(round(365 * row["T"], 1))
    s_days = cells[2].number_input("Days to expiry", 0.5, 3650.0, market_days, 1.0,
                                   key=f"days_{tag}")
    s_r = cells[3].number_input("Rate (%)", 0.0, 15.0, float(100 * r), 0.25, key=f"r_{tag}") / 100
    s_q = cells[4].number_input("Dividend (%)", 0.0, 10.0, float(100 * q), 0.1,
                                key=f"q_{tag}") / 100
    cells[5].write("")
    if cells[5].button("Reset to market", key=f"reset_{tag}"):
        for name in ("spot", "vol", "days", "r", "q"):
            st.session_state.pop(f"{name}_{tag}", None)
        st.rerun(scope="fragment")
    s_T = s_days / 365.0

    # Priced at the same rounded values the inputs start from, so an untouched
    # scenario shows a P&L of exactly zero.
    market = analytics.american_quote(float(round(price, 2)), strike, market_days / 365.0, r, q,
                                      round(100 * row["iv"], 2) / 100, kind, AMERICAN)
    quote = analytics.american_quote(s_spot, strike, s_T, s_r, s_q, s_vol, kind, AMERICAN)
    shares = quantity * CONTRACT_SIZE
    cells = st.columns(4)
    cells[0].metric("Market bid / ask", f"{row['bid']:.2f} / {row['ask']:.2f}")
    cells[1].metric("Implied vol (mid)", f"{row['iv']:.2%}")
    cells[2].metric(f"Scenario price ({STYLE})", f"{quote['price']:.2f}",
                    f"{quote['price'] - market['price']:+.2f} vs today's model price",
                    delta_color="off")
    cells[3].metric("Scenario P&L of the position",
                    f"{shares * (quote['price'] - market['price']):+,.0f}")
    cells = st.columns(4)
    cells[0].metric("Delta", f"{quote['delta']:+.3f}", help=GREEK_HELP["Delta"])
    cells[1].metric("Gamma", f"{quote['gamma']:.4f}", help=GREEK_HELP["Gamma"])
    cells[2].metric("Vega (per vol point)", f"{quote['vega']:.3f}", help=GREEK_HELP["Vega"])
    cells[3].metric("Theta (per day)", f"{quote['theta']:+.3f}", help=GREEK_HELP["Theta"])
    st.caption(
        f"Position of {quantity:+d} contracts in the scenario: value "
        f"{shares * quote['price']:+,.0f}, delta {shares * quote['delta']:+,.0f} shares "
        f"(hedge by trading {-shares * quote['delta']:+,.0f}), gamma "
        f"{shares * quote['gamma']:+,.1f} shares per 1.00 move, vega "
        f"{shares * quote['vega']:+,.0f} per vol point, theta "
        f"{shares * quote['theta']:+,.0f} per day.")

    # -- value against the stock price ---------------------------------------
    st.markdown("**How the option responds to the stock price** (hover to read values)")
    curve = st.radio("Show", analytics.CURVES, horizontal=True, label_visibility="collapsed")
    grid = np.linspace(0.8 * min(price, s_spot), 1.2 * max(price, s_spot), 161)
    lines = {"Stock price": grid,
             "Today (market inputs)": analytics.greek_curve(curve, grid, strike, row["T"], r, q,
                                                            row["iv"], kind),
             "Your scenario": analytics.greek_curve(curve, grid, strike, s_T, s_r, s_q,
                                                    s_vol, kind)}
    if curve == "Price":
        lines["At expiry (payoff)"] = analytics.greek_curve(curve, grid, strike, 0.0, r, q,
                                                            row["iv"], kind)
    markers = {"spot": price, "strike": strike}
    if abs(s_spot - price) > 1e-9:
        markers["scenario"] = s_spot
    st.altair_chart(charts.hover_lines(pd.DataFrame(lines), "Stock price", "Stock price",
                                       f"{curve} per share", markers),
                    use_container_width=True)
    st.caption("Black-Scholes (European) curves. Change the scenario inputs above and the "
               "orange line moves; the gap between the lines is the P&L per share.")

    # -- the same strike across expiries -------------------------------------
    listed = analytics.enrich_chain(load_strike(symbol, strike), price, r, q, TODAY)
    listed = listed[(listed["kind"] == kind) & listed["mid"].notna()]
    if len(listed) >= 2:
        st.markdown(f"**The {strike:g} {kind} across expiries** (hover to read values)")
        days = np.linspace(1.0, 365 * listed["T"].max(), 120)
        model = [float(bs.price(s_spot, strike, d / 365.0, s_r, s_vol, s_q, kind)) for d in days]
        dots = pd.DataFrame({"Days to expiry": 365 * listed["T"], "value": listed["mid"],
                             "label": [f"{e:%d %b %Y}, {v:.1%} vol" for e, v in
                                       zip(listed["expiry"], listed["iv"])]})
        st.altair_chart(charts.hover_lines(
            pd.DataFrame({"Days to expiry": days, f"Model at {s_vol:.1%} vol": model}),
            "Days to expiry", "Days to expiry", "Price per share",
            {"this expiry": 365 * row["T"]}, dots), use_container_width=True)
        st.caption("Dots are market mid prices for each listed expiry; the line is the "
                   "scenario's price if every expiry had the same vol. Dots above the "
                   "line mean the market prices that expiry at a higher vol.")

    with st.expander("The formula behind these numbers (Black-Scholes)", expanded=True):
        show_formula(s_spot, strike, s_T, s_r, s_q, s_vol, kind, quote)

    st.markdown("**Scenario grid:** P&L of the position if spot and vol move today "
                "(Black-Scholes repricing, unhedged)")
    spot_moves = np.array([-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10])
    vol_moves = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
    vol = row["iv"]
    base = float(bs.price(price, strike, row["T"], r, vol, q, kind))
    shocked = bs.price(price * (1 + spot_moves)[None, :], strike, row["T"], r,
                       np.maximum(vol + vol_moves, 0.01)[:, None], q, kind)
    grid = pd.DataFrame(shares * (shocked - base),
                        index=[f"vol {m:+.0%}" for m in vol_moves],
                        columns=[f"spot {m:+.0%}" for m in spot_moves])
    limit = float(np.abs(grid.to_numpy()).max()) or 1.0
    st.dataframe(grid.style.format("{:+,.0f}").background_gradient(
        cmap="RdYlGn", vmin=-limit, vmax=limit), use_container_width=True)


if screen == "Chain & pricer":
    chain_screen()


# ---- 2. vol surface ---------------------------------------------------------------------

if screen == "Vol surface":
    usable = [e for e in expiries if (e - TODAY).days >= 7]
    defaults = sorted({nearest_expiry(d) for d in (30, 60, 120, 240)} & set(usable))
    chosen = st.multiselect("Expiries", usable, default=defaults,
                            format_func=lambda e: f"{e:%d %b %Y}")
    fit_model = st.checkbox("Calibrate Heston to these smiles", value=True)
    if chosen:
        frames = [analytics.enrich_chain(load_expiry(symbol, e, spot, 0.15), spot, r, q,
                                         TODAY) for e in sorted(chosen)]
        enriched = pd.concat(frames, ignore_index=True)
        quotes = {e: analytics.smile(enriched, e, moneyness=(0.85, 1.15))
                  for e in sorted(chosen)}
        fit = None
        if fit_model:
            try:
                fit, _ = calibrate(enriched, spot, r, q, tuple(sorted(chosen)))
            except ValueError as error:
                st.warning(str(error))
        st.pyplot(charts.smiles(quotes, spot, r, q, fit))
        if fit is not None:
            p = fit.params
            cells = st.columns(6)
            cells[0].metric("Fit error", f"{fit.rmse_vol:.2%} vol")
            cells[1].metric("Vol today √v₀", f"{np.sqrt(p.v0):.1%}")
            cells[2].metric("Long-run vol √θ", f"{np.sqrt(p.theta):.1%}")
            cells[3].metric("Mean reversion κ", f"{p.kappa:.2f}")
            cells[4].metric("Vol of vol ξ", f"{p.xi:.2f}")
            cells[5].metric("Correlation ρ", f"{p.rho:+.2f}")
            st.caption("ρ sets the slope of the smile, ξ its curvature, v₀ and θ the "
                       "level at short and long expiries, κ how fast one turns into the other.")
    if not term.empty:
        st.pyplot(charts.term_structure(term))


# ---- 3. realised vs implied ---------------------------------------------------------------

if screen == "Realised vs implied":
    close = price_history()
    if close is not None:
        years = st.radio("History", [1, 3, 5, 10], index=1, horizontal=True,
                         format_func=lambda y: f"{y} year" + ("s" if y > 1 else ""))
        recent = close.iloc[-252 * years:]
        short, long = analytics.realised_vol(close, 21), analytics.realised_vol(close, 63)
        implied = atm_implied(nearest_expiry(30))
        vol_index = load_implied_history(symbol)
        cells = st.columns(4)
        cells[0].metric("Realised vol, 21 days", f"{short.iloc[-1]:.1%}")
        cells[1].metric("Realised vol, 63 days", f"{long.iloc[-1]:.1%}")
        if implied is not None:
            cells[2].metric("ATM implied, ~1 month", f"{implied:.1%}")
            cells[3].metric("Implied minus realised (21 d)",
                            f"{100 * (implied - short.iloc[-1]):+.1f} pts")
        st.pyplot(charts.vol_history(
            recent, {"21-day realised": short.loc[recent.index[0]:],
                     "63-day realised": long.loc[recent.index[0]:]}, implied,
            None if vol_index is None else vol_index.loc[recent.index[0]:]))
        if vol_index is not None:
            # What was implied 21 trading days ago against what then happened.
            gap = (vol_index.reindex(close.index).ffill().shift(21) - short).dropna()
            st.caption(
                f"Since {gap.index[0]:%b %Y}, {vol_index.name} has been above the vol "
                f"realised over the following 21 days on {(gap > 0).mean():.0%} of days, "
                f"by {100 * gap.mean():.1f} points on average. That gap is what option "
                "sellers are paid for carrying the risks of stage 5.")
        st.pyplot(charts.cone(analytics.vol_cone(close), term))
        st.caption(f"Cone built from {len(close):,} daily closes since "
                   f"{close.index[0]:%d %b %Y}.")


# ---- 4. hedging backtest --------------------------------------------------------------------

def trade_inputs(prefix: str):
    """Inputs shared by the single and rolling backtests."""
    a, b, c, d = st.columns(4)
    n_days = a.number_input("Trading days to expiry", 5, 252, 63, key=f"{prefix}_n")
    moneyness = b.number_input("Strike (% of spot)", 70, 130, 100, key=f"{prefix}_m") / 100
    kind = c.radio("Type", ["call", "put"], horizontal=True, key=f"{prefix}_k")
    cost_bp = d.number_input("Cost per trade (bp)", 0.0, 50.0, 1.0, 0.5, key=f"{prefix}_c")
    band = st.slider("Delta band (0 = rebalance every day)", 0.0, 0.3, 0.0, 0.01,
                     key=f"{prefix}_b")
    return n_days, moneyness, kind, cost_bp, band


if screen == "Hedging backtest":
    mode = st.radio("Backtest", ["One trade", "Whole history", "A listed contract"],
                    horizontal=True)
    close = price_history()
    if close is None:
        pass

    elif mode == "One trade":
        n_days, moneyness, kind, cost_bp, band = trade_inputs("one")
        latest_start = close.index[-n_days - 1].date()
        a, b, c = st.columns(3)
        start = a.date_input("Sell the option on", latest_start,
                             min_value=close.index[30].date(), max_value=latest_start)
        trailing = analytics.realised_vol(close, 21).loc[:pd.Timestamp(start)].iloc[-1]
        sale_vol = b.number_input("Sell at vol (%)", 1.0, 200.0,
                                  float(round(100 * trailing + 2, 1)), 0.5) / 100
        hedge_vol = c.number_input("Hedge at vol (%)", 1.0, 200.0,
                                   float(round(100 * sale_vol, 1)), 0.5) / 100
        st.caption(f"Realised vol over the 21 days before the sale: {trailing:.1%}.")
        try:
            run = analytics.hedge_backtest(close, start, n_days, moneyness, kind, sale_vol,
                                           hedge_vol, r, q, cost_bp, band)
        except ValueError as error:
            st.warning(str(error))
        else:
            s = run["summary"]
            cells = st.columns(6)
            cells[0].metric("Premium received", f"{s['premium']:.2f}")
            cells[1].metric("Paid at expiry", f"{s['payoff']:.2f}")
            cells[2].metric("Hedged P&L", f"{s['pnl']:+.2f}",
                            f"{s['pnl'] / s['premium']:+.0%} of premium", delta_color="off")
            cells[3].metric("Costs paid", f"{s['costs']:.2f}")
            cells[4].metric("Vol realised", f"{s['realised_vol']:.1%}",
                            f"{100 * (sale_vol - s['realised_vol']):+.1f} pts vs sale",
                            delta_color="off")
            cells[5].metric("Rebalances", s["trades"])
            st.pyplot(charts.single_backtest(run["days"], s["strike"]))
            st.caption(f"{s['start']:%d %b %Y} to {s['end']:%d %b %Y}. Per share; "
                       f"gamma attribution −½ΓS²(R² − σ²dt) explains {s['gamma_pnl']:+.2f}.")

    elif mode == "Whole history":
        n_days, moneyness, kind, cost_bp, band = trade_inputs("roll")
        vol_index = load_implied_history(symbol)
        choices = ["Past realised vol plus a markup"]
        if vol_index is not None:
            choices.insert(0, f"{vol_index.name} that day (market implied vol)")
        sale_source = st.radio("Sell each option at", choices, horizontal=True)
        use_index = vol_index is not None and sale_source == choices[0]
        a, b, c = st.columns(3)
        step = a.number_input("Sell a new option every (trading days)", 1, 63, 5)
        lookback = b.number_input("Realised-vol lookback (days)", 5, 126, 21,
                                  disabled=use_index)
        markup = c.number_input("Markup on the sale vol (vol points)", -10.0, 20.0,
                                0.0 if use_index else 2.0, 0.5) / 100
        trades = run_rolling(close, n_days=n_days, step=step, moneyness=moneyness, kind=kind,
                             markup=markup, lookback=lookback, r=r, q=q, cost_bp=cost_bp,
                             band=band, sale_vols=vol_index if use_index else None)
        if trades.empty:
            st.warning("Not enough history for these settings.")
        else:
            gap = trades["sale_vol"] - trades["realised_vol"]
            cells = st.columns(6)
            cells[0].metric("Trades", f"{len(trades):,}")
            cells[1].metric("Average P&L", f"{trades['pnl'].mean():+.3f}%")
            cells[2].metric("Std of P&L", f"{trades['pnl'].std():.3f}%")
            cells[3].metric("Profitable", f"{(trades['pnl'] > 0).mean():.0%}")
            cells[4].metric("Worst trade", f"{trades['pnl'].min():+.2f}%")
            cells[5].metric("Average vol gap", f"{100 * gap.mean():+.1f} pts")
            st.pyplot(charts.rolling_backtest(trades))
            st.pyplot(charts.pnl_vs_vol_gap(trades))
            st.caption(
                f"{trades.index[0]:%b %Y} to {trades.index[-1]:%b %Y}. P&L in % of the stock "
                "price at the sale. "
                + (f"Sold at {vol_index.name}, a 30-day implied vol built from the whole "
                   "smile, so it sits a little above the at-the-money vol; use about 21 "
                   "trading days to expiry to match it. " if use_index else
                   "The sale vol is a stand-in built from past realised vol, not a "
                   "historical market quote. ")
                + "Overlapping trades are not independent. "
                f"Correlation of P&L with the vol gap: {np.corrcoef(gap, trades['pnl'])[0, 1]:.2f}.")

    else:
        a, b, c, d = st.columns(4)
        expiry = a.selectbox("Expiry", expiries, index=expiries.index(nearest_expiry(45)),
                             format_func=lambda e: f"{e:%d %b %Y}", key="replay_expiry")
        kind = b.radio("Type", ["call", "put"], horizontal=True, key="replay_kind")
        listed = load_expiry(symbol, expiry, spot, 0.10)
        strikes = sorted(listed.loc[listed["kind"] == kind, "strike"].unique())
        if not strikes:
            st.warning("No listed strikes for this expiry.")
        else:
            strike = c.selectbox("Strike", strikes, key="replay_strike",
                                 index=int(np.argmin(np.abs(np.array(strikes) - spot))))
            start = d.date_input("Sell it on", TODAY - timedelta(days=60),
                                 max_value=TODAY - timedelta(days=3))
            cost_bp = st.number_input("Cost per trade (bp)", 0.0, 50.0, 1.0, 0.5,
                                      key="replay_cost")
            contract = listed.loc[(listed["kind"] == kind) & (listed["strike"] == strike),
                                  "contract"].iloc[0]
            try:
                option_bars = load_option_bars(contract, start)
                table = analytics.contract_replay(option_bars["close"], close, strike, kind,
                                                  expiry, r, q, cost_bp)
            except (DataError, ValueError) as error:
                st.warning(f"{contract}: {error}")
            else:
                st.markdown(f"**{contract}**")
                cells = st.columns(4)
                cells[0].metric("Sold at", f"{table['option'].iloc[0]:.2f}",
                                f"{table['iv'].iloc[0]:.1%} vol", delta_color="off")
                cells[1].metric("Now", f"{table['option'].iloc[-1]:.2f}",
                                f"{table['iv'].iloc[-1]:.1%} vol", delta_color="off")
                cells[2].metric("Hedged P&L per contract",
                                f"{CONTRACT_SIZE * table['pnl'].iloc[-1]:+,.0f}")
                unhedged = table["option"].iloc[0] - table["option"].iloc[-1]
                cells[3].metric("Unhedged P&L per contract", f"{CONTRACT_SIZE * unhedged:+,.0f}")
                st.pyplot(charts.replay(table))
                st.caption(f"{len(table)} trading days, {table.index[0]:%d %b %Y} to "
                           f"{table.index[-1]:%d %b %Y}. Hedged with the Black-Scholes delta "
                           "at each day's implied vol; marked at the option's closing price.")
