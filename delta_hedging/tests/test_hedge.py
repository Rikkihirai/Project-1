"""Hedging engine: the properties delta-hedging theory predicts."""

import numpy as np
import pytest

from hedging import bs
from hedging.hedge import delta_hedge
from hedging.paths import simulate_gbm

S0, K, T, r, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.2


@pytest.fixture(scope="module")
def fine_paths():
    return simulate_gbm(S0, 0.08, SIGMA, T, n_steps=512, n_paths=20_000, seed=7)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_unbiased_when_vols_match(fine_paths, kind):
    # Sold at the vol the stock actually realises: expected P&L ~ 0.
    res = delta_hedge(fine_paths, K, T, r, SIGMA, kind=kind)
    se = res.std / np.sqrt(len(res.pnl))
    assert abs(res.mean) < 4 * se
    assert res.std < 0.05 * res.premium


def test_error_shrinks_like_one_over_sqrt_n(fine_paths):
    coarse = delta_hedge(fine_paths[:, ::16], K, T, r, SIGMA)   # 32 rebalances
    fine = delta_hedge(fine_paths[:, ::4], K, T, r, SIGMA)      # 128 rebalances
    assert coarse.std / fine.std == pytest.approx(2.0, rel=0.08)


def test_matches_derman_kamal(fine_paths):
    # std of hedging error ~ sqrt(pi/4) * vega * sigma / sqrt(N)
    N = 128
    res = delta_hedge(fine_paths[:, ::512 // N], K, T, r, SIGMA)
    predicted = np.sqrt(np.pi / 4) * bs.vega(S0, K, T, r, SIGMA) * SIGMA / np.sqrt(N)
    assert res.std == pytest.approx(predicted, rel=0.05)


def test_pnl_does_not_depend_on_drift():
    for mu in (-0.3, 0.0, 0.3):
        P = simulate_gbm(S0, mu, SIGMA, T, 128, 20_000, seed=3)
        res = delta_hedge(P, K, T, r, SIGMA)
        assert abs(res.mean) < 4 * res.std / np.sqrt(len(res.pnl))


def test_gamma_attribution_explains_pnl(fine_paths):
    res = delta_hedge(fine_paths[:, ::2], K, T, r, SIGMA)
    assert np.corrcoef(res.pnl, res.gamma_pnl)[0, 1] > 0.98


def test_transaction_costs_reduce_pnl(fine_paths):
    P = fine_paths[:, ::8]
    free = delta_hedge(P, K, T, r, SIGMA)
    costly = delta_hedge(P, K, T, r, SIGMA, cost_rate=0.0005)
    assert np.all(costly.costs > 0)
    np.testing.assert_allclose(free.pnl - costly.pnl, costly.costs)


def test_selling_below_realised_vol_loses_money(fine_paths):
    # Option sold and hedged at 15% vol while the stock realises 20%.
    res = delta_hedge(fine_paths[:, ::4], K, T, r, sigma=0.15)
    assert res.mean < -10 * res.std / np.sqrt(len(res.pnl))


def test_single_step_is_static_hedge():
    P = np.array([[100.0, 110.0], [100.0, 90.0]])
    res = delta_hedge(P, K, T, 0.0, SIGMA)
    d0 = float(bs.delta(100.0, K, T, 0.0, SIGMA))
    expected = res.premium + d0 * (P[:, 1] - 100.0) - np.maximum(P[:, 1] - K, 0)
    np.testing.assert_allclose(res.pnl, expected)
