"""Pricing toolkit: binomial tree, Longstaff-Schwartz, PDE solver, Heston
by Fourier transform, and calibration.

Each method is checked against something known independently: the
Black-Scholes formula, a no-arbitrage argument, a published reference
value, or one of the other methods.
"""

import numpy as np
import pytest

from hedging import bs
from hedging.paths import simulate_gbm, simulate_heston
from pricing.american import binomial_price, longstaff_schwartz
from pricing.calibration import calibrate_heston
from pricing.heston import (HestonParams, characteristic_function, heston_price,
                            heston_smile)
from pricing.pde import down_and_out_call, fd_price

S0, K, T, r, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.2
HESTON = HestonParams(v0=0.04, kappa=3.0, theta=0.04, xi=0.4, rho=-0.7)
# Longstaff & Schwartz (2001), Table 1: 1-year put, S=36, K=40, r=6%, vol 20%.
LS_CASE = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
LS_AMERICAN, LS_50_DATES = 4.486, 4.478     # continuous exercise / 50 dates a year


# ---- binomial tree ----------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_european_tree_converges_to_black_scholes(kind, strike):
    exact = float(bs.price(S0, strike, T, r, SIGMA, kind=kind))
    tree = binomial_price(S0, strike, T, r, SIGMA, kind=kind, american=False, n_steps=2000)
    assert tree == pytest.approx(exact, abs=0.003)


def test_tree_error_shrinks_with_more_steps():
    exact = float(bs.price(S0, K, T, r, SIGMA))
    errors = [abs(binomial_price(S0, K, T, r, SIGMA, american=False, n_steps=n) - exact)
              for n in (50, 200, 800)]
    assert errors[0] > errors[1] > errors[2]


def test_american_call_without_dividends_is_never_exercised_early():
    american = binomial_price(S0, K, T, r, SIGMA, kind="call", american=True)
    european = binomial_price(S0, K, T, r, SIGMA, kind="call", american=False)
    assert american == pytest.approx(european, abs=1e-10)


def test_dividends_make_early_exercise_of_a_call_worth_something():
    american = binomial_price(S0, 90.0, 1.0, r, SIGMA, q=0.08, kind="call", american=True)
    european = binomial_price(S0, 90.0, 1.0, r, SIGMA, q=0.08, kind="call", american=False)
    assert american > european + 0.05


def test_american_put_is_worth_more_than_european_and_at_least_intrinsic():
    american = binomial_price(S0, K, T, r, SIGMA, kind="put", american=True)
    european = binomial_price(S0, K, T, r, SIGMA, kind="put", american=False)
    assert american > european + 0.01
    deep = binomial_price(70.0, K, T, r, SIGMA, kind="put", american=True)
    assert deep == pytest.approx(30.0)          # exercise immediately


def test_early_exercise_premium_grows_with_the_interest_rate():
    def premium(rate):
        return (binomial_price(S0, K, 1.0, rate, SIGMA, kind="put", american=True)
                - binomial_price(S0, K, 1.0, rate, SIGMA, kind="put", american=False))
    assert premium(0.08) > premium(0.03) > premium(0.0) == pytest.approx(0.0, abs=1e-10)


def test_tree_matches_published_american_put():
    c = LS_CASE
    tree = binomial_price(c["S0"], c["K"], c["T"], c["r"], c["sigma"], kind="put",
                          n_steps=2000)
    assert tree == pytest.approx(LS_AMERICAN, abs=0.003)


# ---- Longstaff-Schwartz ------------------------------------------------------

@pytest.fixture(scope="module")
def ls_paths():
    c = LS_CASE
    return simulate_gbm(c["S0"], c["r"], c["sigma"], c["T"], 50, 40_000, seed=1,
                        antithetic=True)


def test_longstaff_schwartz_matches_the_paper(ls_paths):
    price, se = longstaff_schwartz(ls_paths, LS_CASE["K"], LS_CASE["T"], LS_CASE["r"])
    assert price == pytest.approx(LS_50_DATES, abs=4 * se + 0.01)


def test_longstaff_schwartz_sits_between_european_and_american(ls_paths):
    c = LS_CASE
    price, se = longstaff_schwartz(ls_paths, c["K"], c["T"], c["r"])
    european = float(bs.price(c["S0"], c["K"], c["T"], c["r"], c["sigma"], kind="put"))
    assert european + 0.3 < price < LS_AMERICAN + 4 * se


def test_longstaff_schwartz_call_without_dividends_is_european():
    paths = simulate_gbm(S0, r, SIGMA, T, 25, 40_000, seed=3, antithetic=True)
    price, se = longstaff_schwartz(paths, K, T, r, kind="call")
    assert price == pytest.approx(float(bs.price(S0, K, T, r, SIGMA)), abs=4 * se + 0.02)


# ---- finite-difference PDE solver -------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_pde_matches_black_scholes(kind, strike):
    exact = float(bs.price(S0, strike, T, r, SIGMA, kind=kind))
    assert fd_price(S0, strike, T, r, SIGMA, kind=kind) == pytest.approx(exact, abs=0.003)


def test_pde_with_dividends_matches_black_scholes():
    exact = float(bs.price(S0, K, 1.0, r, SIGMA, q=0.04, kind="put"))
    assert fd_price(S0, K, 1.0, r, SIGMA, q=0.04, kind="put") == pytest.approx(exact, abs=0.005)


def test_pde_error_shrinks_on_a_finer_grid():
    exact = float(bs.price(S0, K, T, r, SIGMA))
    coarse = abs(fd_price(S0, K, T, r, SIGMA, n_space=50, n_time=50) - exact)
    fine = abs(fd_price(S0, K, T, r, SIGMA, n_space=800, n_time=800) - exact)
    assert fine < coarse / 10


def test_pde_american_put_agrees_with_the_tree():
    tree = binomial_price(S0, K, T, r, SIGMA, kind="put", n_steps=2000)
    assert fd_price(S0, K, T, r, SIGMA, kind="put", american=True) == pytest.approx(tree, abs=0.003)
    c = LS_CASE
    pde = fd_price(c["S0"], c["K"], c["T"], c["r"], c["sigma"], kind="put", american=True)
    assert pde == pytest.approx(LS_AMERICAN, abs=0.003)


@pytest.mark.parametrize("barrier", [85.0, 95.0])
def test_pde_down_and_out_call_matches_closed_form(barrier):
    exact = down_and_out_call(S0, K, T, r, SIGMA, barrier)
    pde = fd_price(S0, K, T, r, SIGMA, lower_barrier=barrier, n_space=800, n_time=800)
    assert pde == pytest.approx(exact, abs=0.003)


def test_knock_out_barriers_only_remove_value():
    vanilla = float(bs.price(S0, K, T, r, SIGMA))
    near = fd_price(S0, K, T, r, SIGMA, upper_barrier=110.0)
    far = fd_price(S0, K, T, r, SIGMA, upper_barrier=120.0)
    assert 0 < near < far < vanilla
    assert fd_price(S0, K, T, r, SIGMA, upper_barrier=400.0) == pytest.approx(vanilla, abs=0.005)
    assert fd_price(S0, K, T, r, SIGMA, upper_barrier=100.0) == 0.0
    assert down_and_out_call(S0, K, T, r, SIGMA, 1e-6) == pytest.approx(vanilla)


# ---- Heston by Fourier transform --------------------------------------------

def test_characteristic_function_basics():
    # phi(0) = 1: probabilities sum to one. phi(-i) = E[S_T] = forward.
    assert characteristic_function(0.0, S0, T, r, HESTON) == pytest.approx(1.0)
    forward = S0 * np.exp(r * T)
    assert characteristic_function(-1j, S0, T, r, HESTON) == pytest.approx(forward)


def test_heston_matches_published_value():
    # Broadie & Kaya (2006): exact price 6.8061.
    params = HestonParams(v0=0.010201, kappa=6.21, theta=0.019, xi=0.61, rho=-0.7)
    assert heston_price(100.0, 100.0, 1.0, 0.0319, params) == pytest.approx(6.8061, abs=1e-4)


@pytest.mark.parametrize("strike", [85.0, 100.0, 115.0])
def test_heston_without_vol_of_vol_is_black_scholes(strike):
    flat = HestonParams(v0=0.04, kappa=3.0, theta=0.04, xi=1e-3, rho=0.0)
    exact = float(bs.price(S0, strike, T, r, 0.2))
    assert heston_price(S0, strike, T, r, flat) == pytest.approx(exact, abs=1e-4)


def test_heston_put_call_parity():
    strikes = np.array([80.0, 100.0, 120.0])
    call = heston_price(S0, strikes, T, r, HESTON, q=0.01, kind="call")
    put = heston_price(S0, strikes, T, r, HESTON, q=0.01, kind="put")
    np.testing.assert_allclose(call - put,
                               S0 * np.exp(-0.01 * T) - strikes * np.exp(-r * T), atol=1e-10)


def test_heston_fourier_agrees_with_monte_carlo():
    S_T = simulate_heston(S0, r, HESTON.v0, HESTON.kappa, HESTON.theta, HESTON.xi,
                          HESTON.rho, T, 126, 40_000, seed=1, antithetic=True)[0][:, -1]
    for strike, kind in ((90.0, "put"), (100.0, "call"), (110.0, "call")):
        payoff = np.exp(-r * T) * bs.payoff(S_T, strike, kind)
        se = payoff.std(ddof=1) / np.sqrt(len(payoff))
        assert heston_price(S0, strike, T, r, HESTON, kind=kind) == pytest.approx(
            payoff.mean(), abs=4 * se + 0.01)


def test_correlation_sets_the_slope_of_the_smile():
    strikes = np.array([90.0, 110.0])
    down = heston_smile(S0, strikes, T, r, HESTON)                      # rho = -0.7
    up = heston_smile(S0, strikes, T, r, HestonParams(0.04, 3.0, 0.04, 0.4, +0.7))
    assert down[0] > down[1] + 0.03
    assert up[1] > up[0] + 0.03


def test_vol_of_vol_sets_the_curvature_of_the_smile():
    def curvature(xi):
        low, mid, high = heston_smile(S0, [85.0, 100.0, 115.0], T, r,
                                      HestonParams(0.04, 3.0, 0.04, xi, 0.0))
        return low + high - 2 * mid
    assert curvature(0.8) > curvature(0.4) > curvature(0.1) > 0


# ---- calibration -------------------------------------------------------------

TRUE = HestonParams(v0=0.05, kappa=2.5, theta=0.035, xi=0.45, rho=-0.65)
QUOTE_MATURITIES = (1 / 12, 0.25, 0.5, 1.0)
QUOTE_STRIKES = np.arange(80.0, 121.0, 5.0)


@pytest.fixture(scope="module")
def quotes():
    strikes = np.tile(QUOTE_STRIKES, len(QUOTE_MATURITIES))
    maturities = np.repeat(QUOTE_MATURITIES, len(QUOTE_STRIKES))
    vols = np.concatenate([heston_smile(S0, QUOTE_STRIKES, t, r, TRUE)
                           for t in QUOTE_MATURITIES])
    return strikes, maturities, vols


def as_array(p):
    return np.array([p.v0, p.kappa, p.theta, p.xi, p.rho])


def test_calibration_recovers_the_parameters_that_made_the_quotes(quotes):
    fit = calibrate_heston(S0, *quotes, r)
    assert fit.rmse_vol < 1e-6
    np.testing.assert_allclose(as_array(fit.params), as_array(TRUE), rtol=1e-3)


def test_calibration_does_not_depend_on_the_starting_guess(quotes):
    start = HestonParams(v0=0.02, kappa=6.0, theta=0.08, xi=1.0, rho=0.2)
    fit = calibrate_heston(S0, *quotes, r, start=start)
    np.testing.assert_allclose(as_array(fit.params), as_array(TRUE), rtol=1e-3)


def test_calibration_with_noisy_quotes_fits_to_the_noise_level(quotes):
    strikes, maturities, vols = quotes
    noise = np.random.default_rng(0).normal(0.0, 0.002, vols.shape)   # 0.2 vol points
    fit = calibrate_heston(S0, strikes, maturities, vols + noise, r)
    assert 0.001 < fit.rmse_vol < 0.003
    assert fit.params.rho == pytest.approx(TRUE.rho, abs=0.05)
    assert fit.params.xi == pytest.approx(TRUE.xi, rel=0.1)
    assert fit.params.v0 == pytest.approx(TRUE.v0, rel=0.05)
