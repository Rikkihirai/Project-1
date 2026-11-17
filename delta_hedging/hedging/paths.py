"""Stock-price path simulation.

Geometric Brownian motion (GBM) has an exact solution over any step dt:

    S(t + dt) = S(t) * exp((mu - sigma^2 / 2) dt + sigma sqrt(dt) Z),  Z ~ N(0, 1)

mu is the *real-world* drift of the price. A hedger does not need to know mu:
the hedged P&L does not depend on it (one of the experiments demonstrates
this). Under the risk-neutral measure mu = r - q.

Two models that break the Black-Scholes assumptions (stage 5)
-------------------------------------------------------------
Heston (1993): the variance v is itself random and mean-reverting.

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

    theta  long-run variance (0.04 = 20% vol)
    kappa  speed of mean reversion; E[v_t] = theta + (v0 - theta) e^{-kappa t}
    xi     vol of vol
    rho    usually negative for equities: prices fall, vol rises

Merton (1976): GBM plus jumps that arrive at random (Poisson) times.

    dS / S = (mu - lam k) dt + sigma dW + (e^J - 1) dN

    N      Poisson process, lam jumps per year on average
    J      log jump size, Normal(jump_mean, jump_std^2)
    k      E[e^J - 1] = exp(jump_mean + jump_std^2 / 2) - 1; subtracting
           lam k from the drift keeps E[S_T] = S0 e^{mu T}

    variance of log returns per year = sigma^2 + lam (jump_mean^2 + jump_std^2)

A stock position can only hedge moves in S that are small and driven by
dW1. Neither the second random factor (dW2) nor a jump can be hedged with
the stock alone, however often the hedger trades.
"""

from __future__ import annotations

import numpy as np


def simulate_gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int,
                 n_paths: int, seed: int | None = None,
                 antithetic: bool = False) -> np.ndarray:
    """Simulate GBM paths on an evenly spaced grid.

    Returns an array of shape (n_paths, n_steps + 1); column 0 is S0 and
    column n_steps is the price at T.

    antithetic=True pairs every draw Z with -Z (n_paths must be even), which
    cancels odd-order noise and cuts Monte Carlo error for the same cost.
    """
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even n_paths")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.vstack([half, -half])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def _normals(rng, n_paths: int, antithetic: bool) -> np.ndarray:
    """One standard normal per path; second half mirrors the first if antithetic."""
    if not antithetic:
        return rng.standard_normal(n_paths)
    half = rng.standard_normal(n_paths // 2)
    return np.concatenate([half, -half])


def simulate_heston(S0: float, mu: float, v0: float, kappa: float,
                    theta: float, xi: float, rho: float, T: float,
                    n_steps: int, n_paths: int, seed: int | None = None,
                    antithetic: bool = False):
    """Simulate Heston paths with the full-truncation Euler scheme.

    Returns (prices, variances), each of shape (n_paths, n_steps + 1).

    Per step, with v+ = max(v, 0) (a discretised v can dip below zero, so
    only its positive part is used: "full truncation"):

        ln S += (mu - v+ / 2) dt + sqrt(v+ dt) Z1
        v    += kappa (theta - v+) dt + xi sqrt(v+ dt) Z2
        Z2    = rho Z1 + sqrt(1 - rho^2) Z_independent

    The price step is exact for a variance held constant over the step, so
    E[S_T] = S0 e^{mu T} holds exactly. The reported variances are v+.
    """
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even n_paths")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    log_S = np.zeros((n_paths, n_steps + 1))
    var = np.empty((n_paths, n_steps + 1))
    v = np.full(n_paths, float(v0))
    var[:, 0] = max(v0, 0.0)
    for k in range(n_steps):
        v_plus = np.maximum(v, 0.0)
        Z1 = _normals(rng, n_paths, antithetic)
        Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * _normals(rng, n_paths, antithetic)
        log_S[:, k + 1] = (log_S[:, k] + (mu - 0.5 * v_plus) * dt
                           + np.sqrt(v_plus * dt) * Z1)
        v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * Z2
        var[:, k + 1] = np.maximum(v, 0.0)
    return S0 * np.exp(log_S), var


def simulate_merton(S0: float, mu: float, sigma: float, lam: float,
                    jump_mean: float, jump_std: float, T: float, n_steps: int,
                    n_paths: int, seed: int | None = None,
                    antithetic: bool = False) -> np.ndarray:
    """Simulate Merton jump-diffusion paths (exact on the grid).

    Over a step dt the log price moves by

        (mu - sigma^2 / 2 - lam k) dt + sigma sqrt(dt) Z + (sum of n jumps)

    with n ~ Poisson(lam dt). A sum of n Normal(m, s^2) jumps is
    Normal(n m, n s^2), so it is drawn as n m + s sqrt(n) Z_J.

    With lam = 0 this returns exactly the paths of `simulate_gbm` for the
    same seed. antithetic mirrors the diffusion draws only.
    """
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even n_paths")
    if lam < 0 or jump_std < 0:
        raise ValueError("lam and jump_std must be >= 0")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.vstack([half, -half])
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
    Z_J = rng.standard_normal((n_paths, n_steps))
    jumps = n_jumps * jump_mean + jump_std * np.sqrt(n_jumps) * Z_J

    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    log_increments = ((mu - 0.5 * sigma**2 - lam * k) * dt
                      + sigma * np.sqrt(dt) * Z + jumps)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)
