"""Finite-difference solver for the Black-Scholes PDE (Crank-Nicolson).

The option value V(S, t) satisfies

    dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0

with V known at expiry (the payoff). The solver puts V on a grid of stock
prices S_0 .. S_M and steps backwards in time from expiry to today.

Space. On the grid, derivatives become differences of neighbours:

    dV/dS   ~ (V[i+1] - V[i-1]) / (2 dS)
    d2V/dS2 ~ (V[i+1] - 2 V[i] + V[i-1]) / dS^2

so the PDE at node i reads  dV/dtau = lower_i V[i-1] + diag_i V[i] + upper_i V[i+1]
(tau = time to expiry), with

    lower_i = 1/2 sigma^2 S_i^2 / dS^2 - (r - q) S_i / (2 dS)
    diag_i  =    -sigma^2 S_i^2 / dS^2 - r
    upper_i = 1/2 sigma^2 S_i^2 / dS^2 + (r - q) S_i / (2 dS)

Time. Call that right-hand side L V. Three ways to take a step d_tau:

    explicit        V_new = V_old + d_tau L V_old        cheap, but unstable
                                                         unless d_tau is tiny
    implicit        V_new = V_old + d_tau L V_new        stable, error ~ d_tau
    Crank-Nicolson  average of the two                   stable, error ~ d_tau^2

The implicit part means solving a linear system each step, but the matrix
is tridiagonal (each node only touches its two neighbours), which costs
O(M). The first two steps are fully implicit ("Rannacher smoothing"),
because Crank-Nicolson rings when started from the kink in the payoff.

What the grid buys over a closed formula:
    American   after each step, V = max(V, payoff)
    barriers   end the grid at the barrier and hold V = 0 there
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from hedging import bs


def _edge_values(kind, american, K, r, q, tau, s_max, lower_barrier, upper_barrier):
    """Option value at the bottom and top of the grid, tau before expiry."""
    if lower_barrier is not None:
        low = 0.0                                   # knocked out
    elif kind == "call":
        low = 0.0                                   # S = 0 stays at 0
    else:
        low = K if american else K * np.exp(-r * tau)   # put is certain to pay K

    if upper_barrier is not None:
        high = 0.0                                  # knocked out
    elif kind == "put":
        high = 0.0                                  # far out of the money
    else:
        high = s_max * np.exp(-q * tau) - K * np.exp(-r * tau)   # deep ITM call
        if american:
            high = max(high, s_max - K)
    return low, high


def fd_solve(K: float, T: float, r: float, sigma: float, s_max: float,
             q: float = 0.0, kind: str = "call", american: bool = False,
             lower_barrier: float | None = None,
             upper_barrier: float | None = None,
             n_space: int = 400, n_time: int = 400):
    """Solve the PDE; returns (S grid, option values today on that grid)."""
    s_min = 0.0 if lower_barrier is None else lower_barrier
    s_max = s_max if upper_barrier is None else upper_barrier
    S = np.linspace(s_min, s_max, n_space + 1)
    dS = S[1] - S[0]
    d_tau = T / n_time

    payoff = bs.payoff(S, K, kind)
    if lower_barrier is not None:
        payoff[0] = 0.0
    if upper_barrier is not None:
        payoff[-1] = 0.0

    inner = S[1:-1]
    diffusion = 0.5 * sigma**2 * inner**2 / dS**2
    drift = (r - q) * inner / (2.0 * dS)
    lower, diag, upper = diffusion - drift, -2.0 * diffusion - r, diffusion + drift

    V = payoff.copy()
    for step in range(n_time):
        w = 1.0 if step < 2 else 0.5        # weight on the implicit side
        low, high = _edge_values(kind, american, K, r, q, (step + 1) * d_tau,
                                 s_max, lower_barrier, upper_barrier)

        # Explicit half: V_old + (1 - w) d_tau L V_old
        rhs = V[1:-1] + (1.0 - w) * d_tau * (
            lower * V[:-2] + diag * V[1:-1] + upper * V[2:])
        # Known edge values of V_new move to the right-hand side.
        rhs[0] += w * d_tau * lower[0] * low
        rhs[-1] += w * d_tau * upper[-1] * high

        # Implicit half: (I - w d_tau L) V_new = rhs, stored as three diagonals.
        bands = np.zeros((3, len(inner)))
        bands[0, 1:] = -w * d_tau * upper[:-1]
        bands[1, :] = 1.0 - w * d_tau * diag
        bands[2, :-1] = -w * d_tau * lower[1:]
        V[1:-1] = solve_banded((1, 1), bands, rhs)
        V[0], V[-1] = low, high

        if american:
            V = np.maximum(V, payoff)
    return S, V


def fd_price(S0: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call", american: bool = False,
             lower_barrier: float | None = None,
             upper_barrier: float | None = None,
             n_space: int = 400, n_time: int = 400) -> float:
    """Price one option: European, American, or with knock-out barriers.

    lower_barrier / upper_barrier: the option dies worthless if the stock
    ever touches that level (down-and-out / up-and-out).
    """
    if lower_barrier is not None and S0 <= lower_barrier:
        return 0.0
    if upper_barrier is not None and S0 >= upper_barrier:
        return 0.0
    # Far enough up that the top edge does not affect the price at S0.
    s_max = max(S0, K) * np.exp(abs(r - q) * T + 5.0 * sigma * np.sqrt(T))
    S, V = fd_solve(K, T, r, sigma, s_max, q, kind, american, lower_barrier,
                    upper_barrier, n_space, n_time)
    return float(np.interp(S0, S, V))


def down_and_out_call(S0: float, K: float, T: float, r: float, sigma: float,
                      barrier: float, q: float = 0.0) -> float:
    """Closed form for a down-and-out call with barrier <= K.

    Reflection principle: paths that touch the barrier H are cancelled by
    an "image" option started at H^2 / S,

        C_do = C(S) - (S / H)^{1 - 2 (r - q) / sigma^2} C(H^2 / S)

    Used to check the PDE solver.
    """
    if barrier > K:
        raise ValueError("this formula needs barrier <= K")
    if S0 <= barrier:
        return 0.0
    power = 1.0 - 2.0 * (r - q) / sigma**2
    image = bs.price(barrier**2 / S0, K, T, r, sigma, q, "call")
    return float(bs.price(S0, K, T, r, sigma, q, "call")
                 - (S0 / barrier) ** power * image)
