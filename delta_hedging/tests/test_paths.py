"""GBM simulator: shape, starting value, and the moments of the distribution."""

import numpy as np
import pytest

from hedging.paths import simulate_gbm


def test_shape_and_start():
    P = simulate_gbm(100, 0.05, 0.2, 1.0, n_steps=12, n_paths=50, seed=0)
    assert P.shape == (50, 13)
    assert np.all(P[:, 0] == 100)
    assert np.all(P > 0)


def test_reproducible_with_seed():
    a = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 5, seed=42)
    b = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 5, seed=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("antithetic", [False, True])
def test_terminal_mean(antithetic):
    # E[S_T] = S0 e^{mu T}; check within 4 standard errors.
    S0, mu, sigma, T = 100, 0.08, 0.3, 1.0
    S_T = simulate_gbm(S0, mu, sigma, T, 4, 200_000, seed=1, antithetic=antithetic)[:, -1]
    se = S_T.std() / np.sqrt(len(S_T))
    assert abs(S_T.mean() - S0 * np.exp(mu * T)) < 4 * se


def test_log_return_volatility():
    sigma, T, n = 0.25, 1.0, 50
    P = simulate_gbm(100, 0.0, sigma, T, n, 20_000, seed=2)
    log_rets = np.diff(np.log(P), axis=1)
    assert log_rets.std() == pytest.approx(sigma * np.sqrt(T / n), rel=0.01)


def test_antithetic_needs_even_paths():
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.0, 0.2, 1.0, 5, 3, antithetic=True)
