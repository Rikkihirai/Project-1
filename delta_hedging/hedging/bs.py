"""Vectorised Black-Scholes pricer and Greeks.

Every function accepts numpy arrays (or scalars) and broadcasts, so a whole
matrix of (path, time) states can be priced in one call. That is what makes
it practical to delta-hedge tens of thousands of simulated paths.

Model
-----
European option on an asset paying a continuous dividend yield q.

    d1 = (ln(S/K) + (r - q + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    call = S e^{-q tau} N(d1) - K e^{-r tau} N(d2)
    put  = K e^{-r tau} N(-d2) - S e^{-q tau} N(-d1)

tau is time to expiry in years. All Greeks below require tau > 0; the
hedging engine never evaluates them at expiry (it settles the payoff there).

Greeks (N = standard normal CDF, n = standard normal PDF)
---------------------------------------------------------
    delta  call: e^{-q tau} N(d1)          put: e^{-q tau} (N(d1) - 1)
    gamma  e^{-q tau} n(d1) / (S sigma sqrt(tau))
    vega   S e^{-q tau} n(d1) sqrt(tau)            (per 1.00 of vol)
    theta  dV/dt in calendar time, per year (so usually negative):
        call: -S e^{-q tau} n(d1) sigma / (2 sqrt(tau))
              - r K e^{-r tau} N(d2) + q S e^{-q tau} N(d1)
        put:  -S e^{-q tau} n(d1) sigma / (2 sqrt(tau))
              + r K e^{-r tau} N(-d2) - q S e^{-q tau} N(-d1)

These satisfy the Black-Scholes PDE, which the tests check directly:

    theta + (r - q) S delta + 1/2 sigma^2 S^2 gamma - r V = 0

That identity is the whole reason delta hedging works: once delta removes
the first-order exposure, what is left is theta versus gamma, and they
cancel exactly *on average* if the stock moves with volatility sigma.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f'kind must be "call" or "put", got {kind!r}')


def _d1_d2(S, K, tau, r, sigma, q):
    S, K, tau, sigma = (np.asarray(x, dtype=float) for x in (S, K, tau, sigma))
    if np.any(tau <= 0):
        raise ValueError("tau must be > 0 (settle the payoff at expiry instead)")
    if np.any(S <= 0) or np.any(K <= 0) or np.any(sigma <= 0):
        raise ValueError("S, K and sigma must be positive")
    vol_sqrt_t = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def payoff(S, K, kind: str = "call"):
    """Payoff at expiry: max(S-K, 0) for a call, max(K-S, 0) for a put."""
    _check_kind(kind)
    S = np.asarray(S, dtype=float)
    return np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)


def price(S, K, tau, r, sigma, q=0.0, kind: str = "call"):
    _check_kind(kind)
    d1, d2 = _d1_d2(S, K, tau, r, sigma, q)
    spot_disc = S * np.exp(-q * tau)
    strike_disc = K * np.exp(-r * tau)
    if kind == "call":
        return spot_disc * norm.cdf(d1) - strike_disc * norm.cdf(d2)
    return strike_disc * norm.cdf(-d2) - spot_disc * norm.cdf(-d1)


def delta(S, K, tau, r, sigma, q=0.0, kind: str = "call"):
    _check_kind(kind)
    d1, _ = _d1_d2(S, K, tau, r, sigma, q)
    if kind == "call":
        return np.exp(-q * tau) * norm.cdf(d1)
    return np.exp(-q * tau) * (norm.cdf(d1) - 1.0)


def gamma(S, K, tau, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, tau, r, sigma, q)
    return np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def vega(S, K, tau, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, tau, r, sigma, q)
    return S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)


def theta(S, K, tau, r, sigma, q=0.0, kind: str = "call"):
    _check_kind(kind)
    d1, d2 = _d1_d2(S, K, tau, r, sigma, q)
    decay = -S * np.exp(-q * tau) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(tau))
    if kind == "call":
        return (decay - r * K * np.exp(-r * tau) * norm.cdf(d2)
                + q * S * np.exp(-q * tau) * norm.cdf(d1))
    return (decay + r * K * np.exp(-r * tau) * norm.cdf(-d2)
            - q * S * np.exp(-q * tau) * norm.cdf(-d1))


def implied_vol(target_price: float, S: float, K: float, tau: float, r: float,
                q: float = 0.0, kind: str = "call") -> float:
    """The vol at which the Black-Scholes price equals target_price.

    The price is strictly increasing in sigma (vega > 0), so there is one
    root and a bracketing solver (Brent) finds it. A price outside the
    no-arbitrage range has no implied vol and raises ValueError.
    """
    def miss(sigma):
        return float(price(S, K, tau, r, sigma, q, kind)) - target_price

    low, high = 1e-4, 5.0
    if miss(low) > 0 or miss(high) < 0:
        raise ValueError("price is outside the range Black-Scholes can produce")
    return float(brentq(miss, low, high, xtol=1e-10))
