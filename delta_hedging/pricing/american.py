"""American options: binomial tree and Longstaff-Schwartz Monte Carlo.

An American option can be exercised at any time up to expiry, so its value
at every moment is the larger of two numbers:

    value = max(exercise now, continuation value)

    exercise now       = payoff(S)
    continuation value = discounted expected value of holding one more step

There is no closed formula, because the holder's best exercise rule has to
be found at the same time as the price. Both methods below work backwards
from expiry, where the value is known, applying that max at every step.

When does early exercise pay?
  - Put: yes when it is deep in the money. Exercising gives K now, and K
    earns interest; waiting only keeps a little optionality.
  - Call with no dividends: never. Exercising pays K early and gives up
    optionality, so American call = European call.
  - Call with dividends: sometimes, to capture the dividend.

Cox-Ross-Rubinstein (1979) binomial tree
----------------------------------------
Over each step dt the stock goes up by u or down by d = 1/u:

    u = e^{sigma sqrt(dt)}             (matches the stock's variance)
    p = (e^{(r-q) dt} - d) / (u - d)   (risk-neutral probability of "up":
                                        makes the stock earn r - q on average)

    V(node) = e^{-r dt} (p V(up) + (1 - p) V(down))

As dt -> 0 the tree converges to Black-Scholes, with error ~ 1/n_steps.

Longstaff-Schwartz (2001) least-squares Monte Carlo
---------------------------------------------------
Monte Carlo runs forwards, but the exercise decision needs the continuation
value, which looks backwards. The trick: estimate the continuation value by
regression. At each date, across the paths that are in the money, regress

    y = discounted cash flow the path receives later if it does not exercise
    x = S / K (as a polynomial)

The fitted curve is the expected value of continuing, given S. A path is
exercised when its payoff beats the fitted value. Only in-the-money paths
are used, since exercising an out-of-the-money option is never right.
"""

from __future__ import annotations

import numpy as np

from hedging import bs


def binomial_price(S0: float, K: float, T: float, r: float, sigma: float,
                   q: float = 0.0, kind: str = "call", american: bool = True,
                   n_steps: int = 500) -> float:
    """Price a European or American option on a CRR binomial tree."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)
    if not 0.0 < p < 1.0:
        raise ValueError("n_steps too small: risk-neutral probability outside (0, 1)")
    discount = np.exp(-r * dt)

    # Prices at expiry, lowest (all downs) to highest (all ups).
    S = S0 * u ** (2.0 * np.arange(n_steps + 1) - n_steps)
    V = bs.payoff(S, K, kind)
    for _ in range(n_steps):
        V = discount * (p * V[1:] + (1.0 - p) * V[:-1])   # continuation value
        S = S[:-1] * u                                    # prices one step earlier
        if american:
            V = np.maximum(V, bs.payoff(S, K, kind))      # exercise if better
    return float(V[0])


def longstaff_schwartz(paths: np.ndarray, K: float, T: float, r: float,
                       kind: str = "put", degree: int = 3):
    """Price an American option on simulated paths; returns (price, std error).

    paths   (n_paths, n_steps + 1) risk-neutral paths (drift r - q), e.g.
            from `hedging.paths.simulate_gbm`. Exercise is allowed at every
            grid date, so more steps = closer to a true American option.
    degree  degree of the polynomial in S/K used for the regression.

    `cash` holds, for every path, the value today of what the path will
    receive under the exercise rule found so far.
    """
    paths = np.asarray(paths, dtype=float)
    n_steps = paths.shape[1] - 1
    discount = np.exp(-r * T / n_steps)

    cash = bs.payoff(paths[:, -1], K, kind)
    for k in range(n_steps - 1, 0, -1):
        cash = cash * discount
        exercise = bs.payoff(paths[:, k], K, kind)
        in_the_money = np.flatnonzero(exercise > 0)
        if len(in_the_money) <= degree + 1:
            continue
        x = paths[in_the_money, k] / K
        fit = np.polyfit(x, cash[in_the_money], degree)
        continuation = np.polyval(fit, x)
        stop = in_the_money[exercise[in_the_money] > continuation]
        cash[stop] = exercise[stop]

    cash = cash * discount
    hold = float(cash.mean())
    exercise_now = float(bs.payoff(paths[0, 0], K, kind))
    return max(hold, exercise_now), float(cash.std(ddof=1) / np.sqrt(len(cash)))
