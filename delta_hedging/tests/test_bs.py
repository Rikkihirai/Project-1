"""Black-Scholes pricer and Greeks: known values, parity, finite differences, PDE."""

import itertools

import numpy as np
import pytest

from hedging import bs

GRID = list(itertools.product(
    [80.0, 100.0, 125.0],   # S
    [0.1, 1.0, 3.0],        # tau
    [0.0, 0.05],            # r
    [0.1, 0.4],             # sigma
    [0.0, 0.03],            # q
))
K = 100.0


def test_textbook_values():
    # S=K=100, T=1, r=5%, sigma=20%: Hull's standard example.
    assert bs.price(100, 100, 1, 0.05, 0.2, kind="call") == pytest.approx(10.4506, abs=1e-4)
    assert bs.price(100, 100, 1, 0.05, 0.2, kind="put") == pytest.approx(5.5735, abs=1e-4)
    assert bs.delta(100, 100, 1, 0.05, 0.2) == pytest.approx(0.63683, abs=1e-5)
    assert bs.gamma(100, 100, 1, 0.05, 0.2) == pytest.approx(0.018762, abs=1e-6)
    assert bs.vega(100, 100, 1, 0.05, 0.2) == pytest.approx(37.524, abs=1e-3)


@pytest.mark.parametrize("S,tau,r,sigma,q", GRID)
def test_put_call_parity(S, tau, r, sigma, q):
    c = bs.price(S, K, tau, r, sigma, q, "call")
    p = bs.price(S, K, tau, r, sigma, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * tau) - K * np.exp(-r * tau), abs=1e-10)


@pytest.mark.parametrize("S,tau,r,sigma,q", GRID)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(S, tau, r, sigma, q, kind):
    h, hv, ht = 1e-2, 1e-5, 1e-6
    V = lambda S_=S, tau_=tau, sig=sigma: bs.price(S_, K, tau_, r, sig, q, kind)
    fd_delta = (V(S + h) - V(S - h)) / (2 * h)
    fd_gamma = (V(S + h) - 2 * V() + V(S - h)) / h**2
    fd_vega = (V(sig=sigma + hv) - V(sig=sigma - hv)) / (2 * hv)
    fd_theta = -(V(tau_=tau + ht) - V(tau_=tau - ht)) / (2 * ht)   # dV/dt = -dV/dtau

    assert bs.delta(S, K, tau, r, sigma, q, kind) == pytest.approx(fd_delta, abs=1e-6)
    assert bs.gamma(S, K, tau, r, sigma, q) == pytest.approx(fd_gamma, abs=1e-5)
    assert bs.vega(S, K, tau, r, sigma, q) == pytest.approx(fd_vega, rel=1e-5, abs=1e-6)
    assert bs.theta(S, K, tau, r, sigma, q, kind) == pytest.approx(fd_theta, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("S,tau,r,sigma,q", GRID)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_black_scholes_pde(S, tau, r, sigma, q, kind):
    V = bs.price(S, K, tau, r, sigma, q, kind)
    lhs = (bs.theta(S, K, tau, r, sigma, q, kind)
           + (r - q) * S * bs.delta(S, K, tau, r, sigma, q, kind)
           + 0.5 * sigma**2 * S**2 * bs.gamma(S, K, tau, r, sigma, q)
           - r * V)
    assert lhs == pytest.approx(0.0, abs=1e-10)


def test_broadcasting_over_paths_and_times():
    S = np.array([[90.0, 100.0, 110.0], [95.0, 105.0, 115.0]])
    tau = np.array([0.5, 0.25, 0.1])
    out = bs.delta(S, K, tau, 0.02, 0.2)
    assert out.shape == (2, 3)
    assert out[0, 1] == pytest.approx(float(bs.delta(100.0, K, 0.25, 0.02, 0.2)))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bs.price(100, 100, 0.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs.price(100, 100, 1.0, 0.05, -0.2)
    with pytest.raises(ValueError):
        bs.price(100, 100, 1.0, 0.05, 0.2, kind="straddle")
