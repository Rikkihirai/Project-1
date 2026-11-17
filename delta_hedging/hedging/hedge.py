"""Discrete delta-hedging engine.

Set-up: a dealer SELLS one European option to a client, collects the
premium, and delta-hedges it with the stock, rebalancing at each grid date
of the supplied paths. Cash earns r. The book is unwound at expiry. The
function returns the P&L at expiry of every path.

To study a long option, negate the P&L.

The P&L of one hedging step
----------------------------
Over a step dt the book (short option, +delta shares, cash = V - delta S)
changes by

    -dV + delta dS + r (V - delta S) dt + q delta S dt.

A Taylor expansion gives dV ~ theta dt + delta dS + 1/2 gamma dS^2, and the
Black-Scholes PDE removes theta, so what is left is

    step P&L  ~  -1/2 gamma S^2 (R^2 - sigma^2 dt),     R = dS / S.

In words, the short-gamma book loses when the realised squared return R^2
beats the variance it was priced for, sigma^2 dt, and wins when it falls
short. With continuous rebalancing these terms average out to zero. With
discrete rebalancing each term is a random number of size ~ gamma S^2
sigma^2 dt. Summing N independent terms gives a total whose standard
deviation shrinks like 1/sqrt(N). The engine reports this "gamma
attribution" next to the true P&L so the two can be compared.

When implied vol is not realised vol (stage 3)
----------------------------------------------
Now the option is sold at implied vol s_i but the stock moves with realised
vol s_r. The hedger must choose which vol to put in the delta formula.
Write V_i, Delta_i, Gamma_i for Black-Scholes values computed at s_i and
V_r, Delta_r for values computed at s_r.

(a) Hedge with the IMPLIED delta. Mark the option at s_i. Ito's lemma uses
    the stock's true quadratic variation, dS^2 = s_r^2 S^2 dt:

        dV_i = theta_i dt + Delta_i dS + 1/2 Gamma_i s_r^2 S^2 dt.

    The BS PDE at s_i gives theta_i = r V_i - (r-q) S Delta_i
    - 1/2 s_i^2 S^2 Gamma_i. Substituting into the book's change
    -dV_i + Delta_i dS + r (V_i - Delta_i S) dt + q Delta_i S dt, the dS
    terms cancel and

        dP&L = 1/2 Gamma_i S^2 (s_i^2 - s_r^2) dt.

    No dW term: the P&L accrues smoothly and its sign is known (gamma > 0).
    But the total, the integral of that over the life of the option,
    depends on where the path goes, because gamma is large only near the
    strike. `variance_edge_pnl` computes this integral.

(b) Hedge with the REALISED delta. Had the option been sold at V_r, this
    hedge would replicate it exactly. It was sold at V_i instead, so the
    difference is banked on day one and the final P&L is the same on every
    path:

        P&L at T = (V_i - V_r) e^{rT}.

    The price of that certainty is a noisy mark-to-market along the way.
    The book holds Delta_r shares against an option the market marks with
    Delta_i, so it carries a residual stock position:

        dP&L = 1/2 Gamma_i S^2 (s_i^2 - s_r^2) dt
               + (Delta_r - Delta_i) ((mu - r + q) S dt + s_r S dW).

    `mark_to_market` returns this daily P&L path for either hedge.

Reference: Ahmad & Wilmott (2005), "Which Free Lunch Would You Like Today,
Sir?".

Transaction costs (stage 4)
---------------------------
Every trade of n shares costs c * |n| * S (c = `cost_rate`, one way, e.g.
0.0005 = 5 bp). At each rebalance the hedger trades the change in delta,
which to first order is gamma times the stock move:

    |trade| ~ Gamma |dS| = Gamma S sigma sqrt(dt) |Z|,    E|Z| = sqrt(2/pi)

    expected cost of one rebalance ~ c Gamma S^2 sigma sqrt(2 dt / pi).

Each rebalance costs sqrt(dt) but there are T/dt of them, so the total
grows like 1/sqrt(dt), i.e. like sqrt(N). Hedging error falls like
1/sqrt(N) (stage 2). Trading more often swaps risk for cost one-for-one.

Leland (1985) noticed that this cost has the same shape as the gamma term
1/2 Gamma S^2 sigma^2 dt in the BS PDE, so it can be absorbed into the vol.
Setting 1/2 Gamma S^2 (s_L^2 - sigma^2) dt equal to the expected cost gives

    s_L^2 = sigma^2 (1 + sqrt(8/pi) c / (sigma sqrt(dt)))     (short option)

A dealer who sells AND hedges at s_L collects, through the higher premium,
what the rebalancing is expected to cost. See `leland_vol`.

Instead of trading on a clock, the hedger can trade only when the hedge has
drifted: keep the current position until it is more than `band` away from
the model delta, then trade back to it. That skips the many small trades
that cost money but remove little risk. See `band_deltas`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bs


@dataclass
class HedgeResult:
    pnl: np.ndarray        # P&L at expiry per path (short option + hedge)
    gamma_pnl: np.ndarray  # -1/2 Gamma S^2 (R^2 - sigma^2 dt), summed, valued at T
    costs: np.ndarray      # transaction costs paid per path, valued at T
    deltas: np.ndarray     # (n_paths, n_steps) shares held over each interval
    premium: float         # premium received at t = 0

    @property
    def n_rebalances(self) -> np.ndarray:
        """Trades per path after the initial hedge (the unwind not counted)."""
        return (np.diff(self.deltas, axis=1) != 0).sum(axis=1)

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))


def delta_hedge(paths: np.ndarray, K: float, T: float, r: float,
                sigma: float, q: float = 0.0, kind: str = "call",
                premium_sigma: float | None = None,
                cost_rate: float = 0.0, band: float = 0.0) -> HedgeResult:
    """Delta-hedge a short option along each simulated path.

    paths          (n_paths, n_steps + 1) prices on an even grid from 0 to T;
                   the hedge is rebalanced at every grid date except T.
    sigma          vol used to compute the hedge ratio (the hedger's model).
    premium_sigma  vol at which the option was sold; defaults to sigma.
    cost_rate      proportional cost per unit traded, e.g. 0.0005 = 5 bp of
                   notional, charged on every rebalance and on the unwind.
    band           0 rebalances at every grid date (time-based hedging).
                   band > 0 checks at every grid date but trades only when
                   the position is more than `band` deltas from the model
                   delta (move-based hedging).
    """
    paths = np.asarray(paths, dtype=float)
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    if n_steps < 1:
        raise ValueError("paths need at least two columns (t = 0 and t = T)")
    premium_sigma = sigma if premium_sigma is None else premium_sigma

    dt = T / n_steps
    t = np.arange(n_steps) * dt            # rebalance dates t_0 .. t_{N-1}
    tau = T - t                            # time to expiry at each date
    S = paths[:, :-1]                      # price at each rebalance
    S_next = paths[:, 1:]
    S_T = paths[:, -1]

    deltas = band_deltas(bs.delta(S, K, tau, r, sigma, q, kind), band)
    gammas = bs.gamma(S, K, tau, r, sigma, q)
    premium = float(bs.price(paths[0, 0], K, T, r, premium_sigma, q, kind))

    # Value every cash flow at T by compounding at r.
    grow_from_t = np.exp(r * (T - t))           # flows at t_k
    grow_from_next = np.exp(r * (T - t - dt))   # flows at t_{k+1}

    trades = np.diff(deltas, axis=1, prepend=0.0)        # shares bought at t_k
    trade_costs = cost_rate * np.abs(trades) * S
    unwind_cost = cost_rate * np.abs(deltas[:, -1]) * S_T
    costs = (trade_costs * grow_from_t).sum(axis=1) + unwind_cost

    stock_purchases = (trades * S * grow_from_t).sum(axis=1)
    dividends = (q * deltas * S * dt * grow_from_next).sum(axis=1)

    pnl = (premium * np.exp(r * T)
           - stock_purchases
           + dividends
           + deltas[:, -1] * S_T            # sell the remaining shares
           - bs.payoff(S_T, K, kind)        # pay the client
           - costs)

    R = S_next / S - 1.0
    gamma_pnl = (-0.5 * gammas * S**2 * (R**2 - sigma**2 * dt)
                 * grow_from_next).sum(axis=1)

    return HedgeResult(pnl=pnl, gamma_pnl=gamma_pnl, costs=costs,
                       deltas=deltas, premium=premium)


def _rebalance_grid(paths: np.ndarray, T: float):
    """Split paths into (S at rebalance dates, S one step later, t, dt)."""
    paths = np.asarray(paths, dtype=float)
    n_steps = paths.shape[1] - 1
    if n_steps < 1:
        raise ValueError("paths need at least two columns (t = 0 and t = T)")
    dt = T / n_steps
    return paths[:, :-1], paths[:, 1:], np.arange(n_steps) * dt, dt


def variance_edge_pnl(paths: np.ndarray, K: float, T: float, r: float,
                      sigma_hedge: float, sigma_real: float,
                      q: float = 0.0) -> np.ndarray:
    """Theoretical P&L of a short option hedged with the implied-vol delta.

        sum over steps of  1/2 Gamma_i S^2 (s_i^2 - s_r^2) dt,  valued at T

    where Gamma_i is computed at sigma_hedge (= the implied vol the option
    was sold at) and sigma_real is the vol the stock actually has. This is
    the continuous-hedging limit, so unlike `HedgeResult.gamma_pnl` it uses
    the expected squared return s_r^2 dt and not each step's realised R^2.
    """
    S, _, t, dt = _rebalance_grid(paths, T)
    gammas = bs.gamma(S, K, T - t, r, sigma_hedge, q)
    grow_from_next = np.exp(r * (T - t - dt))
    edge = 0.5 * gammas * S**2 * (sigma_hedge**2 - sigma_real**2) * dt
    return (edge * grow_from_next).sum(axis=1)


def mark_to_market(paths: np.ndarray, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0, kind: str = "call",
                   premium_sigma: float | None = None,
                   mark_sigma: float | None = None) -> np.ndarray:
    """P&L of the hedged book at every grid date, not only at expiry.

    Returns (n_paths, n_steps + 1). Entry k is, in time-t_k money,

        hedge wealth W_k  -  option value V(S_k, t_k; mark_sigma)

    sigma          vol used for the hedge delta.
    premium_sigma  vol the option was sold at; defaults to sigma.
    mark_sigma     vol the market uses to value the option before expiry;
                   defaults to premium_sigma (implied vol stays put).

    The hedge wealth is self-financing. Start with the premium, hold
    Delta_k shares over [t_k, t_k+1], keep the rest in cash at r:

        W_{k+1} = (W_k - Delta_k S_k) e^{r dt} + Delta_k S_{k+1}
                  + q Delta_k S_k dt

    The last column equals `delta_hedge(...).pnl` with no transaction
    costs, because at expiry the option is worth its payoff whatever vol
    it was marked at.
    """
    S, S_next, t, dt = _rebalance_grid(paths, T)
    premium_sigma = sigma if premium_sigma is None else premium_sigma
    mark_sigma = premium_sigma if mark_sigma is None else mark_sigma
    tau = T - t
    deltas = bs.delta(S, K, tau, r, sigma, q, kind)

    wealth = np.empty((S.shape[0], S.shape[1] + 1))
    wealth[:, 0] = bs.price(S[0, 0], K, T, r, premium_sigma, q, kind)
    for k in range(S.shape[1]):
        cash = wealth[:, k] - deltas[:, k] * S[:, k]
        wealth[:, k + 1] = (cash * np.exp(r * dt)
                            + deltas[:, k] * S_next[:, k]
                            + q * deltas[:, k] * S[:, k] * dt)

    option = np.empty_like(wealth)
    option[:, :-1] = bs.price(S, K, tau, r, mark_sigma, q, kind)
    option[:, -1] = bs.payoff(S_next[:, -1], K, kind)
    return wealth - option


def band_deltas(target: np.ndarray, band: float) -> np.ndarray:
    """Shares actually held when the hedger only trades outside a band.

    target is the model delta at each rebalance date, (n_paths, n_steps).
    The initial hedge is always put on. After that the position is left
    alone while |target - held| <= band, and reset to the target once the
    gap is larger. band = 0 returns the target unchanged.
    """
    if band < 0:
        raise ValueError("band must be >= 0")
    if band == 0:
        return target
    held = np.empty_like(target)
    held[:, 0] = target[:, 0]
    for k in range(1, target.shape[1]):
        previous = held[:, k - 1]
        outside = np.abs(target[:, k] - previous) > band
        held[:, k] = np.where(outside, target[:, k], previous)
    return held


def leland_vol(sigma: float, cost_rate: float, dt: float,
               short: bool = True) -> float:
    """Leland's (1985) cost-adjusted volatility.

        s_L = sigma * sqrt(1 +/- sqrt(8/pi) * c / (sigma * sqrt(dt)))

    c is the one-way proportional cost and dt the time between rebalances.
    Use + for a short option (costs make it worth more to the seller) and
    - for a long one. The dimensionless ratio c / (sigma sqrt(dt)) compares
    the cost of a trade with the size of a typical stock move between
    trades; the adjustment is only sensible while it is small.
    """
    leland_number = np.sqrt(8.0 / np.pi) * cost_rate / (sigma * np.sqrt(dt))
    variance = sigma**2 * (1.0 + leland_number if short else 1.0 - leland_number)
    if variance <= 0:
        raise ValueError("costs too large for this dt: adjusted variance <= 0")
    return float(np.sqrt(variance))
