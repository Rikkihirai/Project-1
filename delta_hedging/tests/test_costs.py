"""Stage 4: transaction costs, delta-band hedging and Leland's adjusted vol.

Theory being checked:
  - rebalancing cost ~ c * vega * sqrt(2N / (pi T)): grows like sqrt(N)
  - a band trades less, pays less, and leaves more risk
  - selling and hedging at Leland's vol pays for the rebalancing
"""

import numpy as np
import pytest

from hedging import bs
from hedging.hedge import band_deltas, delta_hedge, leland_vol
from hedging.paths import simulate_gbm

S0, K, T, r, SIGMA, COST = 100.0, 100.0, 0.25, 0.03, 0.2, 0.0005


@pytest.fixture(scope="module")
def paths():
    return simulate_gbm(S0, 0.08, SIGMA, T, n_steps=256, n_paths=10_000, seed=21)


def rebalancing_cost(res, paths):
    """Mean cost per path, excluding putting the hedge on and taking it off."""
    setup = COST * abs(res.deltas[:, 0]) * paths[:, 0] * np.exp(r * T)
    unwind = COST * abs(res.deltas[:, -1]) * paths[:, -1]
    return float((res.costs - setup - unwind).mean())


# ---- cost of time-based hedging ------------------------------------------

@pytest.mark.parametrize("n", [64, 256])
def test_rebalancing_cost_matches_closed_form(paths, n):
    P = paths[:, :: 256 // n]
    res = delta_hedge(P, K, T, r, SIGMA, cost_rate=COST)
    predicted = COST * bs.vega(S0, K, T, r, SIGMA) * np.sqrt(2 * n / (np.pi * T))
    assert rebalancing_cost(res, P) == pytest.approx(predicted, rel=0.05)


def test_cost_grows_like_sqrt_n(paths):
    coarse, fine = paths[:, ::16], paths[:, ::4]          # 16 and 64 rebalances
    cost = [rebalancing_cost(delta_hedge(P, K, T, r, SIGMA, cost_rate=COST), P)
            for P in (coarse, fine)]
    assert cost[1] / cost[0] == pytest.approx(2.0, rel=0.08)


def test_cost_is_proportional_to_cost_rate(paths):
    P = paths[:2000, ::4]
    one = delta_hedge(P, K, T, r, SIGMA, cost_rate=COST)
    three = delta_hedge(P, K, T, r, SIGMA, cost_rate=3 * COST)
    np.testing.assert_allclose(three.costs, 3 * one.costs)


# ---- delta-band hedging ----------------------------------------------------

def test_zero_band_is_time_based_hedging(paths):
    P = paths[:500, ::4]
    np.testing.assert_array_equal(delta_hedge(P, K, T, r, SIGMA, band=0.0).pnl,
                                  delta_hedge(P, K, T, r, SIGMA).pnl)


def test_position_never_ends_a_date_outside_the_band(paths):
    P = paths[:2000]
    band = 0.05
    target = delta_hedge(P, K, T, r, SIGMA).deltas
    held = delta_hedge(P, K, T, r, SIGMA, band=band).deltas
    assert np.abs(held - target).max() <= band
    np.testing.assert_array_equal(held[:, 0], target[:, 0])   # initial hedge is on


def test_band_only_trades_to_the_model_delta():
    target = np.array([[0.50, 0.53, 0.57, 0.56, 0.40]])
    np.testing.assert_allclose(band_deltas(target, 0.05),
                               [[0.50, 0.50, 0.57, 0.57, 0.40]])


def test_wider_band_means_fewer_trades_lower_cost_more_risk(paths):
    results = [delta_hedge(paths, K, T, r, SIGMA, cost_rate=COST, band=b)
               for b in (0.0, 0.03, 0.10)]
    trades = [res.n_rebalances.mean() for res in results]
    costs = [res.costs.mean() for res in results]
    stds = [res.std for res in results]
    assert trades[0] > trades[1] > trades[2]
    assert costs[0] > costs[1] > costs[2]
    assert stds[0] < stds[1] < stds[2]


def test_band_wider_than_any_delta_is_a_static_hedge(paths):
    P = paths[:200, ::8]
    res = delta_hedge(P, K, T, r, SIGMA, band=1.0)
    assert np.all(res.n_rebalances == 0)
    np.testing.assert_allclose(res.deltas, bs.delta(S0, K, T, r, SIGMA))


def test_band_beats_the_clock_at_equal_cost(paths):
    # Hedging 64 times on a clock costs about the same as a 0.06 band
    # checked 256 times, but the band leaves less risk.
    clock = delta_hedge(paths[:, ::4], K, T, r, SIGMA, cost_rate=COST)
    band = delta_hedge(paths, K, T, r, SIGMA, cost_rate=COST, band=0.06)
    assert band.costs.mean() <= clock.costs.mean()
    assert band.std < clock.std


def test_negative_band_rejected():
    with pytest.raises(ValueError):
        band_deltas(np.zeros((1, 3)), -0.1)


# ---- Leland's adjusted volatility -----------------------------------------

def test_leland_vol_formula():
    dt = T / 64
    expected = SIGMA * np.sqrt(1 + np.sqrt(8 / np.pi) * COST / (SIGMA * np.sqrt(dt)))
    assert leland_vol(SIGMA, COST, dt) == pytest.approx(expected)
    assert leland_vol(SIGMA, 0.0, dt) == pytest.approx(SIGMA)


def test_leland_vol_rises_with_cost_and_frequency():
    assert leland_vol(SIGMA, 2 * COST, T / 64) > leland_vol(SIGMA, COST, T / 64) > SIGMA
    assert leland_vol(SIGMA, COST, T / 256) > leland_vol(SIGMA, COST, T / 64)


def test_leland_vol_is_lower_for_a_long_option():
    assert leland_vol(SIGMA, COST, T / 64, short=False) < SIGMA


def test_leland_vol_rejects_costs_that_swamp_the_moves():
    with pytest.raises(ValueError):
        leland_vol(SIGMA, 0.05, 1e-4, short=False)


@pytest.mark.parametrize("n", [64, 256])
def test_leland_premium_pays_for_the_rebalancing(paths, n):
    # Sold and hedged at sigma: the dealer loses the costs on average.
    # Sold and hedged at Leland's vol: the extra premium covers rebalancing,
    # leaving only the cost of putting the hedge on and taking it off.
    P = paths[:, :: 256 // n]
    naive = delta_hedge(P, K, T, r, SIGMA, cost_rate=COST)
    s_L = leland_vol(SIGMA, COST, T / n)
    leland = delta_hedge(P, K, T, r, s_L, cost_rate=COST)
    on_and_off = leland.costs.mean() - rebalancing_cost(leland, P)
    se = leland.std / np.sqrt(len(leland.pnl))
    assert naive.mean == pytest.approx(-naive.costs.mean(), abs=4 * naive.std / np.sqrt(len(naive.pnl)))
    assert leland.mean == pytest.approx(-on_and_off, abs=4 * se + 0.1 * rebalancing_cost(naive, P))
