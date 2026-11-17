"""Analytics for the desk: everything between raw market data and a screen.

Pure functions on numbers and DataFrames, with no data fetching and no
user interface, so each can be tested on its own.

    implied_vols        vols for a whole chain at once (vectorised bisection)
    enrich_chain        add time to expiry, implied vol and Greeks to a chain
    smile, atm_term_structure, fit_heston
    american_quote      PDE price, early-exercise premium and Greeks
    realised_vol, vol_cone
    hedge_backtest      sell one option on a past date, hedge on real prices
    rolling_backtest    repeat that sale through the whole history
    contract_replay     sell a real listed contract at its real price

Conventions
-----------
Vols are decimals (0.20). Time is in years: calendar days / 365 for option
expiry, trading days / 252 for realised vol and backtests. Theta is per
calendar day and vega per vol point (1% = 0.01), the units a trader quotes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from hedging import bs
from hedging.hedge import delta_hedge, mark_to_market
from pricing.pde import fd_solve
from pricing.calibration import Calibration, calibrate_heston

TRADING_DAYS = 252


def year_fraction(expiry: date, today: date) -> float:
    """Calendar time to expiry in years (at least half a day)."""
    return max((expiry - today).days, 0.5) / 365.0


# ---- implied vol for many options at once ------------------------------------

def implied_vols(prices, S, K, T, r, q=0.0, kinds="call", n_iter: int = 60):
    """Black-Scholes implied vol of each price; NaN where none exists.

    Bisection, vectorised: every option keeps its own bracket [low, high].
    The price rises with vol, so if the price at the midpoint is too low the
    answer is above it. Each pass halves every bracket; 60 passes shrink
    the starting width of 5 to 5 / 2^60, far below any quote precision.
    """
    prices, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (prices, K, T)))
    is_call = np.broadcast_to(np.asarray(kinds) == "call", prices.shape)

    def model(vol):
        calls = bs.price(S, K, T, r, vol, q, "call")
        puts = bs.price(S, K, T, r, vol, q, "put")
        return np.where(is_call, calls, puts)

    low, high = np.full(prices.shape, 1e-4), np.full(prices.shape, 5.0)
    solvable = (prices > model(low)) & (prices < model(high))
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        too_low = model(mid) < prices
        low, high = np.where(too_low, mid, low), np.where(too_low, high, mid)
    return np.where(solvable, 0.5 * (low + high), np.nan)


# ---- option chain --------------------------------------------------------------

def enrich_chain(chain: pd.DataFrame, spot: float, r: float, q: float,
                 today: date) -> pd.DataFrame:
    """Add T, moneyness, implied vol (from the mid price) and Greeks.

    The implied vol is European (Black-Scholes). For the out-of-the-money
    options used to draw the smile the early-exercise premium is tiny, so
    this is accurate there; `american_quote` handles a single contract
    exactly.
    """
    out = chain.copy()
    if out.empty:
        return out
    out["T"] = [year_fraction(expiry, today) for expiry in out["expiry"]]
    out["moneyness"] = out["strike"] / spot
    forward = spot * np.exp((r - q) * out["T"])
    out["otm"] = np.where(out["kind"] == "call", out["strike"] >= forward,
                          out["strike"] < forward)
    out["spread_pct"] = (out["ask"] - out["bid"]) / out["mid"]

    K, T, kinds = out["strike"].to_numpy(), out["T"].to_numpy(), out["kind"].to_numpy()
    iv = implied_vols(out["mid"].to_numpy(), spot, K, T, r, q, kinds)
    out["iv"] = iv
    vol = np.where(np.isnan(iv), 0.2, iv)            # placeholder so Greeks compute
    is_call = kinds == "call"
    delta = np.where(is_call, bs.delta(spot, K, T, r, vol, q, "call"),
                     bs.delta(spot, K, T, r, vol, q, "put"))
    theta = np.where(is_call, bs.theta(spot, K, T, r, vol, q, "call"),
                     bs.theta(spot, K, T, r, vol, q, "put"))
    greeks = {"delta": delta, "gamma": bs.gamma(spot, K, T, r, vol, q),
              "vega": bs.vega(spot, K, T, r, vol, q) / 100.0, "theta": theta / 365.0}
    for name, values in greeks.items():
        out[name] = np.where(np.isnan(iv), np.nan, values)
    return out


def smile(enriched: pd.DataFrame, expiry: date, moneyness=(0.8, 1.2),
          max_spread: float = 0.5) -> pd.DataFrame:
    """Out-of-the-money quotes of one expiry with a usable implied vol."""
    rows = enriched[(enriched["expiry"] == expiry) & enriched["otm"]
                    & enriched["iv"].notna() & (enriched["bid"] > 0)
                    & (enriched["spread_pct"] <= max_spread)
                    & enriched["moneyness"].between(*moneyness)]
    return rows.sort_values("strike")


def atm_term_structure(enriched: pd.DataFrame, spot: float) -> pd.DataFrame:
    """At-the-money implied vol for each expiry (interpolated at the spot)."""
    rows = []
    for expiry in sorted(enriched["expiry"].unique()):
        quotes = smile(enriched, expiry, moneyness=(0.9, 1.1))
        if len(quotes) >= 2 and quotes["strike"].min() <= spot <= quotes["strike"].max():
            rows.append({"expiry": expiry, "T": quotes["T"].iloc[0],
                         "atm_iv": float(np.interp(spot, quotes["strike"], quotes["iv"]))})
    return pd.DataFrame(rows, columns=["expiry", "T", "atm_iv"])


def fit_heston(enriched: pd.DataFrame, spot: float, r: float, q: float,
               expiries: list, moneyness=(0.85, 1.15)):
    """Calibrate Heston to the smiles of the chosen expiries.

    Returns (Calibration, quotes used). Needs at least five quotes because
    there are five parameters.
    """
    quotes = pd.concat([smile(enriched, e, moneyness) for e in expiries])
    if len(quotes) < 5:
        raise ValueError("not enough usable quotes to calibrate")
    fit: Calibration = calibrate_heston(spot, quotes["strike"].to_numpy(),
                                        quotes["T"].to_numpy(), quotes["iv"].to_numpy(), r, q)
    return fit, quotes


# ---- one contract, priced as American ------------------------------------------

def american_quote(spot: float, strike: float, T: float, r: float, q: float,
                   sigma: float, kind: str, american: bool = True,
                   n_space: int = 600, n_time: int = 300) -> dict:
    """Price, early-exercise premium and Greeks of one option.

    american=True for stock and ETF options (SPY); False for index options
    (SPX), which can only be exercised at expiry.

    One PDE solve gives the option value at every stock price on the grid,
    so delta and gamma are read off as the slope and curvature of that
    curve at the spot:

        delta_i = (V[i+1] - V[i-1]) / (2 dS)
        gamma_i = (V[i+1] - 2 V[i] + V[i-1]) / dS^2

    Vega and theta have no such shortcut: the option is repriced with the
    vol nudged by one point, and with one day less to expiry.
    """
    s_max = max(spot, strike) * np.exp(abs(r - q) * T + 5.0 * sigma * np.sqrt(T))

    def solve(vol=sigma, time=T):
        return fd_solve(strike, time, r, vol, s_max, q, kind, american=american,
                        n_space=n_space, n_time=n_time)

    def price_at_spot(vol=sigma, time=T):
        S, V = solve(vol, time)
        return float(np.interp(spot, S, V))

    S, V = solve()
    dS = S[1] - S[0]
    delta = (V[2:] - V[:-2]) / (2.0 * dS)
    gamma = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dS**2
    price = float(np.interp(spot, S, V))
    european = float(bs.price(spot, strike, T, r, sigma, q, kind))
    one_day = min(1.0 / 365.0, 0.5 * T)
    return {
        "price": price,
        "european": european,
        "early_exercise": price - european,
        "intrinsic": float(bs.payoff(spot, strike, kind)),
        "delta": float(np.interp(spot, S[1:-1], delta)),
        "gamma": float(np.interp(spot, S[1:-1], gamma)),
        "vega": (price_at_spot(vol=sigma + 0.01) - price_at_spot(vol=sigma - 0.01)) / 2.0,
        "theta": (price_at_spot(time=T - one_day) - price) * (1.0 / 365.0) / one_day,
    }


# ---- curves and formula terms for the pricer screen ------------------------------

CURVES = ("Price", "Delta", "Gamma", "Vega", "Theta")


def greek_curve(quantity: str, S, K: float, T: float, r: float, q: float,
                sigma: float, kind: str) -> np.ndarray:
    """One Black-Scholes quantity across an array of stock prices.

    quantity is one of CURVES. Vega is per vol point and theta per calendar
    day, as on the rest of the desk. At T = 0 the price is the payoff and
    the Greeks are not defined (NaN).
    """
    S = np.asarray(S, dtype=float)
    if T <= 0:
        return bs.payoff(S, K, kind) if quantity == "Price" else np.full(S.shape, np.nan)
    if quantity == "Price":
        return bs.price(S, K, T, r, sigma, q, kind)
    if quantity == "Delta":
        return bs.delta(S, K, T, r, sigma, q, kind)
    if quantity == "Gamma":
        return bs.gamma(S, K, T, r, sigma, q)
    if quantity == "Vega":
        return bs.vega(S, K, T, r, sigma, q) / 100.0
    if quantity == "Theta":
        return bs.theta(S, K, T, r, sigma, q, kind) / 365.0
    raise ValueError(f"unknown curve: {quantity!r}")


def black_scholes_terms(S: float, K: float, T: float, r: float, q: float,
                        sigma: float, kind: str) -> dict:
    """Every intermediate number of the Black-Scholes formula, for display.

        d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
        call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
        put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    """
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1.0 if kind == "call" else -1.0
    stock_leg = S * np.exp(-q * T) * norm.cdf(sign * d1)
    strike_leg = K * np.exp(-r * T) * norm.cdf(sign * d2)
    return {"d1": float(d1), "d2": float(d2), "N_d1": float(norm.cdf(sign * d1)),
            "N_d2": float(norm.cdf(sign * d2)), "stock_leg": float(stock_leg),
            "strike_leg": float(strike_leg), "price": float(sign * (stock_leg - strike_leg))}


# ---- realised volatility ---------------------------------------------------------

def realised_vol(close: pd.Series, window: int) -> pd.Series:
    """Rolling annualised volatility of daily log returns.

        sqrt(252) * standard deviation of ln(S_t / S_{t-1}) over `window` days
    """
    returns = np.log(close).diff()
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def vol_cone(close: pd.Series, windows=(10, 21, 63, 126, 252)) -> pd.DataFrame:
    """Where realised vol has been, for several window lengths.

    One row per window: percentiles of the rolling realised vol over the
    whole history, plus its latest value. Plotted against window length the
    percentiles form a cone, wide for short windows and narrow for long
    ones. Implied vols drawn on top show whether options are cheap or
    expensive relative to how the stock has actually moved.
    """
    rows = []
    for window in windows:
        series = realised_vol(close, window).dropna()
        if series.empty:
            continue
        low, p25, median, p75, high = np.percentile(series, [5, 25, 50, 75, 95])
        rows.append({"window": window, "p05": low, "p25": p25, "median": median,
                     "p75": p75, "p95": high, "current": float(series.iloc[-1])})
    return pd.DataFrame(rows)


# ---- backtests on real prices ------------------------------------------------------

def hedge_backtest(close: pd.Series, start, n_days: int, moneyness: float, kind: str,
                   sale_vol: float, hedge_vol: float, r: float, q: float = 0.0,
                   cost_bp: float = 0.0, band: float = 0.0) -> dict:
    """Sell one option on `start` and delta-hedge it daily on real closes.

    The option expires `n_days` trading days later with strike
    moneyness x the starting price. It is sold at `sale_vol` and hedged
    with the Black-Scholes delta at `hedge_vol`, through the same engine as
    the simulation stages. Returns a per-day table and a summary.
    """
    window = close.loc[pd.Timestamp(start):].iloc[: n_days + 1]
    if len(window) < n_days + 1:
        raise ValueError("not enough price history after the start date")
    paths = window.to_numpy()[None, :]
    S0, T = float(paths[0, 0]), n_days / TRADING_DAYS
    K = moneyness * S0
    result = delta_hedge(paths, K, T, r, hedge_vol, q, kind, premium_sigma=sale_vol,
                         cost_rate=cost_bp / 1e4, band=band)
    # Daily P&L before costs, with the option marked at the sale vol.
    mtm = mark_to_market(paths, K, T, r, hedge_vol, q, kind, premium_sigma=sale_vol)[0]
    realised = float(np.log(window).diff().std() * np.sqrt(TRADING_DAYS))
    days = pd.DataFrame({"spot": window, "pnl": mtm,
                         "delta": np.append(result.deltas[0], np.nan)})
    summary = {"start": window.index[0], "end": window.index[-1], "spot0": S0,
               "strike": K, "premium": result.premium, "pnl": float(result.pnl[0]),
               "costs": float(result.costs[0]), "gamma_pnl": float(result.gamma_pnl[0]),
               "trades": int(result.n_rebalances[0]), "realised_vol": realised,
               "payoff": float(bs.payoff(paths[0, -1], K, kind))}
    return {"days": days, "summary": summary}


def rolling_backtest(close: pd.Series, n_days: int = 63, step: int = 5,
                     moneyness: float = 1.0, kind: str = "call",
                     markup: float = 0.02, lookback: int = 21, r: float = 0.03,
                     q: float = 0.0, cost_bp: float = 0.0, band: float = 0.0,
                     sale_vols: pd.Series | None = None) -> pd.DataFrame:
    """Sell the same option every `step` days through the whole history.

    The vol each option is sold at is, by default, a stand-in: the realised
    vol of the previous `lookback` days plus `markup`. If `sale_vols` is
    given (a daily implied-vol history such as VIX), each option is instead
    sold at that day's market implied vol plus `markup`.

    The option is hedged at its sale vol. Every window is scaled to start
    at 100, so the P&L is in percent of the stock price and trades from
    different years are comparable.

    One row per trade: sale vol, the vol realised afterwards, and the P&L.
    """
    if sale_vols is None:
        quoted = realised_vol(close, lookback)
    else:
        quoted = sale_vols.reindex(close.index).ffill()
    values, T = close.to_numpy(), n_days / TRADING_DAYS
    rows = []
    for i in range(lookback, len(values) - n_days, step):
        sale_vol = float(quoted.iloc[i]) + markup
        if not sale_vol > 0.01:               # also skips days with no quote (NaN)
            continue
        path = 100.0 * values[i:i + n_days + 1] / values[i]
        res = delta_hedge(path[None, :], 100.0 * moneyness, T, r, sale_vol, q, kind,
                          cost_rate=cost_bp / 1e4, band=band)
        realised = float(np.diff(np.log(path)).std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append({"start": close.index[i], "sale_vol": sale_vol,
                     "realised_vol": realised, "premium": res.premium,
                     "pnl": float(res.pnl[0]), "costs": float(res.costs[0])})
    return pd.DataFrame(rows).set_index("start")


def contract_replay(option_close: pd.Series, stock_close: pd.Series, strike: float,
                    kind: str, expiry: date, r: float, q: float = 0.0,
                    cost_bp: float = 0.0) -> pd.DataFrame:
    """Sell a real listed contract at its real price and hedge it daily.

    On every day both the option and the stock traded:
      1. back out the implied vol from the option's closing price,
      2. hold the Black-Scholes delta at that vol until the next day.

    The book starts with the premium in cash. Its value is tracked with the
    self-financing rule of `hedging.hedge.mark_to_market`, but marked
    against the option's actual market price, so the P&L column is what a
    risk system would have shown each evening.
    """
    both = pd.DataFrame({"option": option_close}).join(
        stock_close.rename("spot"), how="inner").dropna()
    both = both[both.index.date < expiry]
    if len(both) < 2:
        raise ValueError("need at least two days on which the option traded")
    T = np.array([year_fraction(expiry, day.date()) for day in both.index])
    S, option = both["spot"].to_numpy(), both["option"].to_numpy()
    iv = implied_vols(option, S, strike, T, r, q, kind)
    iv = pd.Series(iv, index=both.index).ffill().bfill().to_numpy()   # days with no solution
    delta = bs.delta(S, strike, T, r, iv, q, kind)

    cost_rate = cost_bp / 1e4
    wealth = np.empty(len(both))
    wealth[0] = option[0] - cost_rate * abs(delta[0]) * S[0]
    for k in range(len(both) - 1):
        dt = (both.index[k + 1] - both.index[k]).days / 365.0
        cash = wealth[k] - delta[k] * S[k]
        trade_cost = cost_rate * abs(delta[k + 1] - delta[k]) * S[k + 1]
        wealth[k + 1] = (cash * np.exp(r * dt) + delta[k] * S[k + 1]
                         + q * delta[k] * S[k] * dt - trade_cost)
    both["iv"], both["delta"], both["pnl"] = iv, delta, wealth - option
    return both
