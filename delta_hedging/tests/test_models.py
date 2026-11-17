"""Stage 5: Heston and Merton simulators, and BS hedging when BS is wrong.

Simulator checks use moments known in closed form. Hedging checks use the
main prediction of the theory: risk from a second random factor or from
jumps cannot be removed by trading the stock more often.
"""

import numpy as np
import pytest

from hedging import bs
from hedging.hedge import delta_hedge
from hedging.paths import simulate_gbm, simulate_heston, simulate_merton

S0, K, T, r, MU, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.08, 0.2
HESTON = dict(v0=0.04, kappa=3.0, theta=0.04, xi=0.4, rho=-0.7)
LAM, JUMP_MEAN, JUMP_STD = 1.0, -0.08, 0.08
# Diffusive vol chosen so total variance matches SIGMA^2.
SIGMA_D = np.sqrt(SIGMA**2 - LAM * (JUMP_MEAN**2 + JUMP_STD**2))
N_PATHS = 10_000


def std_error(x):
    return x.std(ddof=1) / np.sqrt(len(x))


@pytest.fixture(scope="module")
def gbm():
    return simulate_gbm(S0, MU, SIGMA, T, 128, N_PATHS, seed=4, antithetic=True)


@pytest.fixture(scope="module")
def heston():
    return simulate_heston(S0, MU, T=T, n_steps=128, n_paths=N_PATHS, seed=4,
                           antithetic=True, **HESTON)


@pytest.fixture(scope="module")
def merton():
    return simulate_merton(S0, MU, SIGMA_D, LAM, JUMP_MEAN, JUMP_STD, T, 128,
                           N_PATHS, seed=4, antithetic=True)


# ---- Heston simulator ------------------------------------------------------

def test_heston_shapes_and_start(heston):
    prices, var = heston
    assert prices.shape == var.shape == (N_PATHS, 129)
    np.testing.assert_allclose(prices[:, 0], S0)
    np.testing.assert_allclose(var[:, 0], HESTON["v0"])
    assert var.min() >= 0


def test_heston_price_grows_at_mu(heston):
    S_T = heston[0][:, -1]
    assert S_T.mean() == pytest.approx(S0 * np.exp(MU * T), abs=4 * std_error(S_T))


def test_heston_variance_mean_reverts():
    # E[v_t] = theta + (v0 - theta) e^{-kappa t}
    v0, kappa, theta = 0.09, 3.0, 0.04
    _, var = simulate_heston(S0, MU, v0, kappa, theta, 0.4, -0.7, 1.0, 256,
                             N_PATHS, seed=8)
    for col in (64, 128, 256):
        expected = theta + (v0 - theta) * np.exp(-kappa * col / 256)
        assert var[:, col].mean() == pytest.approx(expected, rel=0.03)


def test_heston_without_vol_of_vol_is_gbm():
    prices, var = simulate_heston(S0, MU, 0.04, 3.0, 0.04, 0.0, 0.0, T, 64,
                                  N_PATHS, seed=2)
    np.testing.assert_allclose(var, 0.04)
    log_ret = np.log(prices[:, -1] / S0)
    assert log_ret.std() == pytest.approx(SIGMA * np.sqrt(T), rel=0.03)
    assert log_ret.mean() == pytest.approx((MU - 0.5 * SIGMA**2) * T,
                                           abs=4 * std_error(log_ret))


def test_heston_returns_and_variance_moves_have_correlation_rho(heston):
    prices, var = heston
    returns = np.diff(np.log(prices), axis=1).ravel()
    var_moves = np.diff(var, axis=1).ravel()
    assert np.corrcoef(returns, var_moves)[0, 1] == pytest.approx(HESTON["rho"], abs=0.03)


def test_heston_rejects_bad_rho():
    with pytest.raises(ValueError):
        simulate_heston(S0, MU, 0.04, 3.0, 0.04, 0.4, -1.5, T, 4, 10)


# ---- Merton simulator ------------------------------------------------------

def test_merton_without_jumps_is_exactly_gbm():
    a = simulate_merton(S0, MU, SIGMA, 0.0, JUMP_MEAN, JUMP_STD, T, 32, 1000,
                        seed=6, antithetic=True)
    b = simulate_gbm(S0, MU, SIGMA, T, 32, 1000, seed=6, antithetic=True)
    np.testing.assert_allclose(a, b)


def test_merton_price_grows_at_mu(merton):
    # The -lam k drift correction offsets the average effect of the jumps.
    S_T = merton[:, -1]
    assert S_T.mean() == pytest.approx(S0 * np.exp(MU * T), abs=4 * std_error(S_T))


def test_merton_total_variance(merton):
    # var of log return = (sigma^2 + lam (m^2 + s^2)) T = SIGMA^2 T by construction
    log_ret = np.log(merton[:, -1] / S0)
    assert log_ret.var() == pytest.approx(SIGMA**2 * T, rel=0.08)


def test_merton_jump_frequency(merton):
    # A jump is far outside what the diffusion can do in one step.
    step_returns = np.diff(np.log(merton), axis=1)
    big = np.abs(step_returns) > 6 * SIGMA_D * np.sqrt(T / 128)
    jumps_per_year = big.sum() / (N_PATHS * T)
    assert 0.5 * LAM < jumps_per_year < 1.05 * LAM   # small jumps go undetected


def test_downward_jumps_give_a_fat_left_tail(merton, gbm):
    def skew(x):
        return float(((x - x.mean())**3).mean() / x.std()**3)
    assert skew(np.log(merton[:, -1] / S0)) < -0.3
    assert abs(skew(np.log(gbm[:, -1] / S0))) < 0.1


# ---- implied vol and the smile ----------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("sigma", [0.1, 0.35])
def test_implied_vol_round_trip(kind, strike, sigma):
    price = float(bs.price(S0, strike, T, r, sigma, kind=kind))
    assert bs.implied_vol(price, S0, strike, T, r, kind=kind) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_impossible_price():
    with pytest.raises(ValueError):
        bs.implied_vol(S0 + 1.0, S0, K, T, r)       # a call can't cost more than S


def test_downward_jumps_make_low_strikes_expensive():
    # Price a 90 put and a 110 call by Monte Carlo under the risk-neutral
    # drift; the put's implied vol must be higher (the equity "skew").
    S_T = simulate_merton(S0, r, SIGMA_D, LAM, JUMP_MEAN, JUMP_STD, T, 1,
                          40_000, seed=1, antithetic=True)[:, -1]
    def iv(strike, kind):
        price = np.exp(-r * T) * bs.payoff(S_T, strike, kind).mean()
        return bs.implied_vol(price, S0, strike, T, r, kind=kind)
    assert iv(90.0, "put") > iv(100.0, "call") + 0.01 > iv(110.0, "call") + 0.01


# ---- hedging with the wrong model -------------------------------------------

def hedge(paths, n):
    return delta_hedge(paths[:, :: 128 // n], K, T, r, SIGMA)


def test_gbm_error_halves_with_4x_the_trades_but_others_do_not(gbm, heston, merton):
    assert hedge(gbm, 32).std / hedge(gbm, 128).std == pytest.approx(2.0, rel=0.1)
    assert hedge(heston[0], 32).std / hedge(heston[0], 128).std < 1.4
    assert hedge(merton, 32).std / hedge(merton, 128).std < 1.15


def test_wrong_model_leaves_more_risk_and_fatter_tails(gbm, heston, merton):
    base = hedge(gbm, 128)
    for paths in (heston[0], merton):
        res = hedge(paths, 128)
        assert res.std > 2 * base.std
        assert np.percentile(res.pnl, 1) < 2 * np.percentile(base.pnl, 1)


def test_heston_pnl_is_a_bet_on_realised_variance(heston):
    prices, _ = heston
    res = hedge(prices, 128)
    realised_var = (np.diff(np.log(prices), axis=1)**2).sum(axis=1) / T
    assert np.corrcoef(res.pnl, realised_var)[0, 1] < -0.6
    assert res.pnl[realised_var > SIGMA**2].mean() < 0 < res.pnl[realised_var < SIGMA**2].mean()


def test_jump_paths_carry_the_losses(merton):
    res = hedge(merton, 128)
    step_returns = np.diff(np.log(merton), axis=1)
    jumped = (np.abs(step_returns) > 6 * SIGMA_D * np.sqrt(T / 128)).any(axis=1)
    # No jump: the stock realises less than the 20% charged, so the dealer wins.
    assert res.pnl[~jumped].mean() > 0.3
    assert res.pnl[jumped].mean() < -1.0
    assert res.pnl.min() < -8.0


def test_jump_loss_is_the_convexity_the_delta_misses():
    # Stock gaps from 100 to 90 before the hedger can trade. The short option
    # book loses  V(90) - V(100) - delta * (90 - 100), and the second-order
    # estimate 1/2 Gamma dS^2 is the right size.
    before = float(bs.price(100.0, K, T, 0.0, SIGMA))
    after = float(bs.price(90.0, K, T, 0.0, SIGMA))
    delta = float(bs.delta(100.0, K, T, 0.0, SIGMA))
    loss = after - before - delta * (90.0 - 100.0)
    gamma_estimate = 0.5 * float(bs.gamma(100.0, K, T, 0.0, SIGMA)) * 10.0**2
    assert loss > 0
    assert loss == pytest.approx(gamma_estimate, rel=0.25)
