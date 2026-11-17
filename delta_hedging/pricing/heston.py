"""Heston option prices by Fourier transform.

Stage 5 priced options under Heston by simulating 40,000 paths. That is
slow and noisy. Heston (1993) showed the price is available almost in
closed form, through the characteristic function of the log price.

Characteristic function
-----------------------
For x = ln S_T,   phi(u) = E[e^{i u x}].  It is the Fourier transform of
the density of x: knowing phi for all u is the same as knowing the whole
distribution. For Heston it has a closed form (below, in the numerically
stable form of Albrecher et al. 2007):

    d = sqrt((rho xi i u - kappa)^2 + xi^2 (i u + u^2))
    g = (kappa - rho xi i u - d) / (kappa - rho xi i u + d)

    phi(u) = exp(i u (ln S0 + (r - q) T))
           * exp(kappa theta / xi^2 * ((kappa - rho xi i u - d) T
                                       - 2 ln((1 - g e^{-d T}) / (1 - g))))
           * exp(v0 / xi^2 * (kappa - rho xi i u - d)
                           * (1 - e^{-d T}) / (1 - g e^{-d T}))

Two sanity checks: phi(0) = 1 (probabilities sum to one) and
phi(-i) = E[S_T] = S0 e^{(r-q)T} (the stock earns r - q).

From phi to a price
-------------------
Exactly as in Black-Scholes, a call is

    C = S0 e^{-qT} P1 - K e^{-rT} P2

where P2 is the risk-neutral probability of finishing in the money and P1
is the same probability measured with the stock as the unit of account.
Black-Scholes has P1 = N(d1), P2 = N(d2). Here they come from inverting
the characteristic function (Gil-Pelaez):

    P2 = 1/2 + 1/pi * integral_0^inf Re[ e^{-i u ln K} phi(u) / (i u) ] du
    P1 = 1/2 + 1/pi * integral_0^inf Re[ e^{-i u ln K} phi(u - i)
                                         / (i u phi(-i)) ] du

The integrals are one-dimensional and smooth, so Gauss-Legendre quadrature
with a few hundred points gives ~1e-8 accuracy in well under a millisecond.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

from hedging import bs

# Quadrature nodes on (0, U_MAX); the integrand has decayed to nothing by then.
U_MAX, N_NODES = 400.0, 512
_x, _w = np.polynomial.legendre.leggauss(N_NODES)
NODES, WEIGHTS = 0.5 * U_MAX * (_x + 1.0), 0.5 * U_MAX * _w


@dataclass
class HestonParams:
    v0: float      # variance today
    kappa: float   # speed of mean reversion
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # correlation between stock and variance moves


def characteristic_function(u, S0: float, T: float, r: float,
                            params: HestonParams, q: float = 0.0):
    """phi(u) = E[exp(i u ln S_T)] under the risk-neutral measure."""
    v0, kappa, theta, xi, rho = astuple(params)
    u = np.asarray(u, dtype=complex)
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b**2 + xi**2 * (1j * u + u**2))
    g = (b - d) / (b + d)
    decay = np.exp(-d * T)
    drift = 1j * u * (np.log(S0) + (r - q) * T)
    long_run = kappa * theta / xi**2 * (
        (b - d) * T - 2.0 * np.log((1.0 - g * decay) / (1.0 - g)))
    initial = v0 / xi**2 * (b - d) * (1.0 - decay) / (1.0 - g * decay)
    return np.exp(drift + long_run + initial)


def heston_price(S0: float, K, T: float, r: float, params: HestonParams,
                 q: float = 0.0, kind: str = "call"):
    """European option price under Heston. K may be an array of strikes."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    forward = S0 * np.exp((r - q) * T)                  # = phi(-i)
    phi = characteristic_function(NODES, S0, T, r, params, q)
    phi_shifted = characteristic_function(NODES - 1j, S0, T, r, params, q)

    # One row per strike, one column per quadrature node.
    wave = np.exp(-1j * np.outer(np.log(K), NODES)) / (1j * NODES)
    P2 = 0.5 + (wave * phi).real @ WEIGHTS / np.pi
    P1 = 0.5 + (wave * phi_shifted / forward).real @ WEIGHTS / np.pi

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    if kind == "call":
        price = call
    elif kind == "put":                                 # put-call parity
        price = call - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    else:
        raise ValueError(f'kind must be "call" or "put", got {kind!r}')
    return price if price.size > 1 else float(price[0])


def heston_smile(S0: float, strikes, T: float, r: float, params: HestonParams,
                 q: float = 0.0) -> np.ndarray:
    """Black-Scholes implied vol of the Heston price at each strike.

    Uses the out-of-the-money option at each strike (put below the forward,
    call above), where the implied vol is best determined numerically.
    """
    forward = S0 * np.exp((r - q) * T)
    vols = []
    for strike in np.atleast_1d(strikes):
        kind = "put" if strike < forward else "call"
        price = heston_price(S0, strike, T, r, params, q, kind)
        vols.append(bs.implied_vol(price, S0, strike, T, r, q, kind))
    return np.array(vols)
