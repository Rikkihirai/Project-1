"""Calibrate Heston to a market smile.

A pricer turns parameters into prices. Calibration runs it backwards: find
the parameters under which the model reproduces the prices the market is
quoting. This is the daily use of a pricer on a desk: once the model
matches the liquid vanilla options, it is used to price and hedge
everything else consistently with them.

Objective
---------
Quotes are implied vols at (strike, maturity) pairs. We want model vols
close to market vols, but inverting to an implied vol inside the optimiser
is slow. A first-order Taylor expansion avoids it:

    model price - market price ~ vega * (model vol - market vol)

so  (model price - market price) / vega  is the error in vol units. The
optimiser minimises the sum of squares of that over all quotes.

What each parameter does to the smile (why the fit is identifiable)
-------------------------------------------------------------------
    v0     level of short-maturity vol
    theta  level of long-maturity vol
    kappa  how fast the level moves from one to the other
    rho    slope of the smile (negative = low strikes more expensive)
    xi     curvature of the smile

One maturity alone cannot separate v0, theta and kappa, so quotes from
several maturities are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from hedging import bs
from .heston import HestonParams, heston_price

#             v0     kappa  theta  xi     rho
LOWER = (1e-4, 0.05, 1e-4, 0.01, -0.99)
UPPER = (1.0, 20.0, 1.0, 3.0, 0.99)


@dataclass
class Calibration:
    params: HestonParams
    rmse_vol: float          # root-mean-square error of the fit, in vol units
    n_evaluations: int       # how many times the smile was repriced


def _model_prices(x, S0, strikes, maturities, r, q):
    """Call prices for every quote, one Fourier inversion per maturity."""
    params = HestonParams(*x)
    prices = np.empty(len(strikes))
    for T in np.unique(maturities):
        rows = maturities == T
        prices[rows] = heston_price(S0, strikes[rows], T, r, params, q, "call")
    return prices


def calibrate_heston(S0: float, strikes, maturities, market_vols, r: float,
                     q: float = 0.0, start: HestonParams | None = None) -> Calibration:
    """Fit the five Heston parameters to quoted implied vols.

    strikes, maturities, market_vols: 1-D arrays of equal length, one entry
    per quote. Returns the fitted parameters and the fit error.
    """
    strikes, maturities, market_vols = (
        np.asarray(a, dtype=float) for a in (strikes, maturities, market_vols))
    market_prices = bs.price(S0, strikes, maturities, r, market_vols, q, "call")
    vegas = bs.vega(S0, strikes, maturities, r, market_vols, q)

    def vol_errors(x):
        return (_model_prices(x, S0, strikes, maturities, r, q) - market_prices) / vegas

    if start is None:                      # flat smile at the average quoted vol
        level = float(np.mean(market_vols)) ** 2
        start = HestonParams(v0=level, kappa=2.0, theta=level, xi=0.5, rho=-0.5)
    x0 = [start.v0, start.kappa, start.theta, start.xi, start.rho]
    fit = least_squares(vol_errors, x0, bounds=(LOWER, UPPER))
    return Calibration(params=HestonParams(*fit.x),
                       rmse_vol=float(np.sqrt(np.mean(fit.fun**2))),
                       n_evaluations=int(fit.nfev))
