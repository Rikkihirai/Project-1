"""Stage 3: option sold at implied vol, stock realises a different vol.

Theory (Ahmad & Wilmott 2005), for a dealer SHORT the option:
  - hedge at realised vol -> P&L = (V_imp - V_real) e^{rT} on every path
  - hedge at implied vol  -> P&L = integral of 1/2 Gamma_i S^2 (s_i^2 - s_r^2) dt,
    always the same sign, but its size depends on the path
"""

import numpy as np
import pytest

from hedging import bs
from hedging.hedge import delta_hedge, mark_to_market, variance_edge_pnl
from hedging.paths import simulate_gbm

S0, K, T, r = 100.0, 100.0, 0.25, 0.03
IMP, REAL = 0.30, 0.20          # sold at 30% vol, stock realises 20%


def locked_in(imp=IMP, real=REAL, kind="call"):
    """(V(imp) - V(real)) e^{rT}: the edge from selling at the wrong vol."""
    edge = bs.price(S0, K, T, r, imp, kind=kind) - bs.price(S0, K, T, r, real, kind=kind)
    return float(edge) * np.exp(r * T)


@pytest.fixture(scope="module")
def paths():
    return simulate_gbm(S0, 0.08, REAL, T, n_steps=256, n_paths=10_000, seed=11)


@pytest.fixture(scope="module")
def implied_hedge(paths):
    return delta_hedge(paths, K, T, r, sigma=IMP)


@pytest.fixture(scope="module")
def realised_hedge(paths):
    return delta_hedge(paths, K, T, r, sigma=REAL, premium_sigma=IMP)


# ---- hedging with the realised-vol delta ---------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
def test_realised_hedge_locks_in_the_price_difference(paths, kind):
    res = delta_hedge(paths, K, T, r, sigma=REAL, premium_sigma=IMP, kind=kind)
    target = locked_in(kind=kind)
    assert res.mean == pytest.approx(target, abs=4 * res.std / np.sqrt(len(res.pnl)))
    assert res.std < 0.2 * target       # tight around it: only discrete-hedging noise


def test_realised_hedge_is_fair_hedge_plus_a_constant(paths, realised_hedge):
    # Same deltas as selling at the right vol; only the premium differs, so
    # the P&L is shifted by exactly (V_imp - V_real) e^{rT} on every path.
    fair = delta_hedge(paths, K, T, r, sigma=REAL)
    np.testing.assert_allclose(realised_hedge.pnl - fair.pnl, locked_in())


# ---- hedging with the implied-vol delta ----------------------------------

def test_implied_hedge_matches_the_gamma_integral(paths, implied_hedge):
    theory = variance_edge_pnl(paths, K, T, r, sigma_hedge=IMP, sigma_real=REAL)
    assert np.corrcoef(implied_hedge.pnl, theory)[0, 1] > 0.93
    assert implied_hedge.mean == pytest.approx(theory.mean(), rel=0.02)


def test_implied_hedge_sign_follows_implied_minus_realised(paths):
    rich = delta_hedge(paths, K, T, r, sigma=0.30)     # sold above realised
    cheap = delta_hedge(paths, K, T, r, sigma=0.15)    # sold below realised
    assert np.mean(rich.pnl > 0) > 0.99
    assert np.mean(cheap.pnl < 0) > 0.99
    assert np.all(variance_edge_pnl(paths, K, T, r, 0.30, REAL) > 0)
    assert np.all(variance_edge_pnl(paths, K, T, r, 0.15, REAL) < 0)


def test_implied_hedge_pnl_is_path_dependent(implied_hedge, realised_hedge):
    assert implied_hedge.std > 2.5 * realised_hedge.std


def test_implied_hedge_earns_most_when_stock_ends_at_the_strike(paths, implied_hedge):
    # Gamma is concentrated at the strike, so paths that finish there collect
    # the most 1/2 Gamma S^2 (s_i^2 - s_r^2) dt.
    S_T = paths[:, -1]
    near = np.abs(S_T / K - 1) < 0.02
    far = np.abs(S_T / K - 1) > 0.15
    assert implied_hedge.pnl[near].mean() > 2 * implied_hedge.pnl[far].mean()


def test_variance_edge_is_zero_when_vols_match(paths):
    np.testing.assert_allclose(variance_edge_pnl(paths[:100], K, T, r, REAL, REAL), 0.0)


# ---- both hedges compared -------------------------------------------------

def test_same_expected_pnl_under_risk_neutral_drift():
    # With mu = r the discounted stock is a martingale, so no choice of delta
    # can change the EXPECTED P&L: both hedges average (V_imp - V_real) e^{rT}.
    P = simulate_gbm(S0, r, REAL, T, 256, 40_000, seed=5, antithetic=True)
    for hedge_vol in (IMP, REAL):
        res = delta_hedge(P, K, T, r, sigma=hedge_vol, premium_sigma=IMP)
        assert res.mean == pytest.approx(locked_in(), abs=4 * res.std / np.sqrt(len(res.pnl)))


def test_only_the_implied_hedge_depends_on_drift():
    means = {IMP: [], REAL: []}
    for mu in (0.0, 0.6):
        P = simulate_gbm(S0, mu, REAL, T, 256, 20_000, seed=9)
        for hedge_vol in (IMP, REAL):
            means[hedge_vol].append(
                delta_hedge(P, K, T, r, sigma=hedge_vol, premium_sigma=IMP).mean)
    # A strong drift carries the stock away from the strike, where gamma is small.
    assert means[IMP][1] < 0.8 * means[IMP][0]
    assert means[REAL][1] == pytest.approx(means[REAL][0], abs=0.03)


# ---- mark-to-market paths --------------------------------------------------

@pytest.mark.parametrize("hedge_vol", [IMP, REAL])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mark_to_market_starts_at_zero_and_ends_at_final_pnl(paths, hedge_vol, kind):
    P = paths[:500, ::4]
    mtm = mark_to_market(P, K, T, r, sigma=hedge_vol, premium_sigma=IMP, kind=kind)
    final = delta_hedge(P, K, T, r, sigma=hedge_vol, premium_sigma=IMP, kind=kind)
    assert mtm.shape == P.shape
    np.testing.assert_allclose(mtm[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(mtm[:, -1], final.pnl)


def test_mark_to_market_with_dividends_matches_engine(paths):
    P = paths[:500, ::4]
    mtm = mark_to_market(P, K, T, r, sigma=REAL, q=0.02, premium_sigma=IMP)
    final = delta_hedge(P, K, T, r, sigma=REAL, q=0.02, premium_sigma=IMP)
    np.testing.assert_allclose(mtm[:, -1], final.pnl)


def test_implied_hedge_has_the_smoother_mark_to_market(paths):
    # Implied delta: dP&L has no dW term. Realised delta: it carries
    # (Delta_r - Delta_i) s_r S dW. Compare the size of the step-to-step P&L moves.
    P = paths[:4000]
    step_moves = {
        vol: np.diff(mark_to_market(P, K, T, r, sigma=vol, premium_sigma=IMP), axis=1).std()
        for vol in (IMP, REAL)}
    assert step_moves[REAL] > 2.5 * step_moves[IMP]
