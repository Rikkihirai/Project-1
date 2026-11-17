"""The desk: market-data layer and the analytics behind the dashboard.

No test touches the network. Alpaca responses are faked, and everything
else runs on simulated prices where the right answer is known.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from desk import analytics
from desk.data import (AlpacaData, DataError, PublicData, SyntheticData,
                       available_sources, connect, contract_symbol, load_keys,
                       parse_contract)
from hedging import bs
from hedging.hedge import delta_hedge
from hedging.paths import simulate_gbm

R, Q, SIGMA = 0.03, 0.0, 0.2


# ---- contract symbols ----------------------------------------------------------

def test_contract_symbol_round_trip():
    symbol = contract_symbol("SPY", date(2026, 12, 18), "call", 750.0)
    assert symbol == "SPY261218C00750000"
    parsed = parse_contract(symbol)
    assert (parsed.root, parsed.expiry, parsed.kind, parsed.strike) == (
        "SPY", date(2026, 12, 18), "call", 750.0)
    assert parse_contract(contract_symbol("AAPL", date(2027, 1, 15), "put", 187.5)).strike == 187.5


def test_bad_contract_symbol_rejected():
    with pytest.raises(ValueError):
        parse_contract("SPY")


# ---- Alpaca client, with a fake network -----------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self.payload


class FakeSession:
    """Replays canned responses in order and records every request."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0)


def bar(day, close):
    return {"t": f"{day}T05:00:00Z", "o": close, "h": close, "l": close, "c": close, "v": 10}


def snapshot(bid, ask, iv=0.2):
    return {"latestQuote": {"bp": bid, "ap": ask}, "latestTrade": {"p": bid},
            "impliedVolatility": iv, "dailyBar": {"v": 5}}


def test_chain_follows_pages_and_computes_mids():
    session = FakeSession([
        FakeResponse({"snapshots": {"SPY261218C00750000": snapshot(17.0, 17.4)},
                      "next_page_token": "abc"}),
        FakeResponse({"snapshots": {"SPY261218P00750000": snapshot(0.0, 0.0)},
                      "next_page_token": None}),
    ])
    chain = AlpacaData("id", "secret", session=session, cache_dir=None).option_chain(
        "SPY", date(2026, 12, 18), date(2026, 12, 18), 700, 800)
    assert len(session.calls) == 2
    assert session.calls[1]["params"]["page_token"] == "abc"
    assert session.calls[0]["params"]["feed"] == "indicative"
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == "id"
    call = chain[chain["kind"] == "call"].iloc[0]
    assert (call["strike"], call["mid"], call["expiry"]) == (750.0, 17.2, date(2026, 12, 18))
    assert np.isnan(chain[chain["kind"] == "put"]["mid"].iloc[0])      # no two-sided quote


def test_http_error_becomes_data_error():
    session = FakeSession([FakeResponse({"message": "forbidden"}, status=403)])
    with pytest.raises(DataError):
        AlpacaData("id", "secret", session=session, cache_dir=None).spot("SPY")


def test_daily_bars_are_cached_and_only_recent_days_refetched(tmp_path):
    first = FakeSession([FakeResponse({"bars": {"SPY": [
        bar("2026-01-05", 100.0), bar("2026-01-20", 101.0), bar("2026-01-21", 102.0)]}})])
    AlpacaData("id", "secret", session=first, cache_dir=tmp_path).daily_bars(
        "SPY", date(2026, 1, 1), date(2026, 1, 21))
    assert (tmp_path / "SPY_1Day.csv").exists()

    # Second request: only the days from (last cached day - 5) are asked for.
    second = FakeSession([FakeResponse({"bars": {"SPY": [
        bar("2026-01-20", 101.0), bar("2026-01-21", 102.5), bar("2026-01-22", 103.0)]}})])
    bars = AlpacaData("id", "secret", session=second, cache_dir=tmp_path).daily_bars(
        "SPY", date(2026, 1, 1), date(2026, 1, 22))
    assert len(second.calls) == 1
    assert second.calls[0]["params"]["start"] == "2026-01-16"
    assert second.calls[0]["params"]["adjustment"] == "split"
    assert list(bars["close"]) == [100.0, 101.0, 102.5, 103.0]     # 21 Jan was revised


def test_keys_come_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert load_keys(tmp_path / ".env") is None
    (tmp_path / ".env").write_text("# comment\nAPCA_API_KEY_ID=abc\nAPCA_API_SECRET_KEY='xyz'\n")
    assert load_keys(tmp_path / ".env") == ("abc", "xyz")


# ---- public sources (Cboe + Yahoo), with a fake network ---------------------------

class TextResponse(FakeResponse):
    def __init__(self, text):
        super().__init__(None)
        self.text = text


def cboe_quote(symbol, bid, ask, volume=10.0, iv=0.2):
    return {"option": symbol, "bid": bid, "ask": ask, "iv": iv, "volume": volume,
            "last_trade_price": bid}


def cboe_file(quotes, price=740.0):
    return FakeResponse({"data": {"current_price": price, "options": quotes}})


def test_cboe_file_gives_spot_and_chain_from_one_download():
    session = FakeSession([cboe_file([
        cboe_quote("SPY261218C00740000", 17.0, 17.2),
        cboe_quote("SPY261218P00740000", 15.0, 15.4),
        cboe_quote("SPY270115C00800000", 0.0, 0.05),          # no bid: no mid
        cboe_quote("B1SPY261120C00710000", 1.0, 1.2),         # non-standard: skipped
    ])])
    source = PublicData(session=session, cache_dir=None)
    assert source.spot("SPY") == 740.0
    chain = source.option_chain("SPY")
    assert len(session.calls) == 1                            # reused within quote_ttl
    assert session.calls[0]["url"].endswith("/delayed_quotes/options/SPY.json")
    assert list(chain["contract"]) == ["SPY261218C00740000", "SPY261218P00740000",
                                       "SPY270115C00800000"]
    assert chain["mid"].iloc[0] == pytest.approx(17.1)
    assert np.isnan(chain["mid"].iloc[2])

    near = source.option_chain("SPY", expiry_to=date(2026, 12, 31), strike_from=700,
                               strike_to=750)
    assert len(near) == 2 and set(near["kind"]) == {"call", "put"}


def test_a_dropped_connection_is_retried_once_then_reported():
    import requests

    class Flaky(FakeSession):
        def __init__(self, failures, responses):
            super().__init__(responses)
            self.failures = failures

        def get(self, url, **kwargs):
            if self.failures:
                self.failures -= 1
                raise requests.ConnectionError("connection reset by peer")
            return super().get(url, **kwargs)

    recovers = Flaky(1, [cboe_file([cboe_quote("SPY261218C00740000", 17.0, 17.2)])])
    assert PublicData(session=recovers, cache_dir=None).spot("SPY") == 740.0
    with pytest.raises(DataError, match="network problem"):
        PublicData(session=Flaky(2, []), cache_dir=None).spot("SPY")


def test_index_uses_its_own_names_and_drops_duplicate_listings():
    session = FakeSession([cboe_file([
        cboe_quote("SPX261218C07400000", 100.0, 102.0, volume=5.0),
        cboe_quote("SPXW261218C07400000", 100.5, 101.5, volume=50.0),   # same option
    ], price=7400.0)])
    chain = PublicData(session=session, cache_dir=None).option_chain("SPX")
    assert session.calls[0]["url"].endswith("/_SPX.json")
    assert list(chain["contract"]) == ["SPXW261218C07400000"]   # the more traded one


def test_yahoo_daily_bars_parse_and_skip_days_without_a_close():
    payload = {"chart": {"result": [{
        "timestamp": [1767623400, 1767709800, 1767796200],      # 5, 6, 7 Jan 2026
        "indicators": {"quote": [{"open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0],
                                  "low": [1.0, 2.0, 3.0], "close": [100.0, None, 102.0],
                                  "volume": [10, 0, 30]}]}}]}}
    session = FakeSession([FakeResponse(payload)])
    bars = PublicData(session=session, cache_dir=None).daily_bars(
        "SPX", date(2026, 1, 1), date(2026, 1, 31))
    assert session.calls[0]["url"].endswith("/%5EGSPC") or session.calls[0]["url"].endswith("/^GSPC")
    assert list(bars["close"]) == [100.0, 102.0]
    assert [d.date() for d in bars.index] == [date(2026, 1, 5), date(2026, 1, 7)]


def test_vix_history_is_parsed_into_decimal_vols():
    csv = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n11/13/2026,18.78,20.92,17.64,19.30\n"
    session = FakeSession([TextResponse(csv)])
    vix = PublicData(session=session, cache_dir=None).implied_vol_history("SPY")
    assert vix.name == "VIX" and vix.iloc[-1] == pytest.approx(0.193)
    assert vix.index[0] == pd.Timestamp("1990-01-02")
    assert PublicData(session=FakeSession([]), cache_dir=None).implied_vol_history("AAPL") is None


def test_public_source_is_the_default_and_needs_no_keys():
    assert available_sources()[0] == PublicData.name
    assert SyntheticData.name in available_sources()
    assert isinstance(connect(), PublicData)
    with pytest.raises(ValueError):
        connect("nonsense")


# ---- synthetic market -------------------------------------------------------------

@pytest.fixture(scope="module")
def market():
    return SyntheticData(today=date(2026, 11, 13))


@pytest.fixture(scope="module")
def enriched(market):
    spot = market.spot("SPY")
    return analytics.enrich_chain(market.option_chain("SPY"), spot, market.r, market.q,
                                  market.today)


def test_synthetic_market_shape(market):
    bars = market.daily_bars("SPY", date(2016, 1, 1))
    assert len(bars) == 2520 and bars.index[-1].date() == date(2026, 11, 13)
    assert (bars["high"] >= bars["close"]).all() and (bars["low"] <= bars["close"]).all()
    chain = market.option_chain("SPY")
    assert chain["expiry"].nunique() == 6
    assert (chain["ask"] > chain["bid"]).all()


def test_synthetic_option_history_ends_at_todays_quote(market):
    chain = market.option_chain("SPY")
    row = chain.iloc[len(chain) // 2]
    history = market.option_bars(row["contract"], date(2026, 9, 1))
    assert history["close"].iloc[-1] == pytest.approx(row["mid"], rel=1e-6)


# ---- implied vol and the chain ----------------------------------------------------

def test_implied_vols_round_trip_for_calls_and_puts():
    strikes = np.array([80.0, 100.0, 120.0, 90.0, 110.0])
    kinds = np.array(["call", "call", "call", "put", "put"])
    vols = np.array([0.35, 0.2, 0.15, 0.3, 0.25])
    prices = [float(bs.price(100.0, k, 0.5, R, v, Q, kind))
              for k, v, kind in zip(strikes, vols, kinds)]
    np.testing.assert_allclose(
        analytics.implied_vols(prices, 100.0, strikes, 0.5, R, Q, kinds), vols, atol=1e-8)


def test_implied_vol_is_nan_for_impossible_prices():
    out = analytics.implied_vols([150.0, 0.0], 100.0, [100.0, 100.0], 0.5, R, Q, "call")
    assert np.isnan(out).all()


def test_enriched_chain_matches_the_quoted_vols_and_sign_conventions(enriched):
    valid = enriched.dropna(subset=["iv"])
    np.testing.assert_allclose(valid["iv"], valid["quote_iv"], atol=1e-6)
    calls, puts = valid[valid["kind"] == "call"], valid[valid["kind"] == "put"]
    assert calls["delta"].between(0, 1).all() and puts["delta"].between(-1, 0).all()
    assert (valid["gamma"] > 0).all() and (valid["vega"] > 0).all()
    assert (valid.loc[valid["otm"], "theta"] < 0).all()   # deep ITM puts can have theta > 0


def test_smile_keeps_only_out_of_the_money_quotes(enriched, market):
    expiry = sorted(enriched["expiry"].unique())[1]
    quotes = analytics.smile(enriched, expiry)
    forward = market.spot("SPY") * np.exp((market.r - market.q) * quotes["T"].iloc[0])
    assert (quotes.loc[quotes["kind"] == "put", "strike"] < forward).all()
    assert (quotes.loc[quotes["kind"] == "call", "strike"] >= forward).all()
    assert quotes["strike"].is_monotonic_increasing
    assert quotes["iv"].iloc[0] > quotes["iv"].iloc[-1]      # rho < 0: downward skew


def test_calibration_recovers_the_model_behind_the_synthetic_market(enriched, market):
    expiries = sorted(enriched["expiry"].unique())[:4]
    fit, quotes = analytics.fit_heston(enriched, market.spot("SPY"), market.r, market.q,
                                       expiries)
    truth = market._params_on(market.bars.index[-1])
    assert fit.rmse_vol < 1e-5
    assert fit.params.rho == pytest.approx(truth.rho, abs=0.01)
    assert fit.params.xi == pytest.approx(truth.xi, rel=0.02)
    assert fit.params.v0 == pytest.approx(truth.v0, rel=0.02)


def test_term_structure_has_one_point_per_expiry(enriched, market):
    term = analytics.atm_term_structure(enriched, market.spot("SPY"))
    assert len(term) == enriched["expiry"].nunique()
    assert term["T"].is_monotonic_increasing


# ---- American quote -----------------------------------------------------------------

def test_american_call_without_dividends_matches_black_scholes_greeks():
    quote = analytics.american_quote(100.0, 100.0, 0.25, R, 0.0, SIGMA, "call")
    assert quote["early_exercise"] == pytest.approx(0.0, abs=0.005)
    assert quote["delta"] == pytest.approx(float(bs.delta(100, 100, 0.25, R, SIGMA)), abs=0.01)
    assert quote["gamma"] == pytest.approx(float(bs.gamma(100, 100, 0.25, R, SIGMA)), rel=0.02)
    assert quote["vega"] == pytest.approx(float(bs.vega(100, 100, 0.25, R, SIGMA)) / 100, rel=0.02)
    assert quote["theta"] == pytest.approx(float(bs.theta(100, 100, 0.25, R, SIGMA)) / 365, rel=0.05)


def test_european_style_quote_is_black_scholes():
    quote = analytics.american_quote(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, "put", american=False)
    assert quote["early_exercise"] == pytest.approx(0.0, abs=0.005)


def test_american_put_carries_an_early_exercise_premium():
    quote = analytics.american_quote(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, "put")
    assert quote["early_exercise"] > 0.5
    assert quote["price"] >= quote["intrinsic"]


# ---- curves and formula terms ---------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
def test_formula_terms_rebuild_the_black_scholes_price(kind):
    terms = analytics.black_scholes_terms(100.0, 105.0, 0.5, R, 0.01, SIGMA, kind)
    assert terms["price"] == pytest.approx(float(bs.price(100.0, 105.0, 0.5, R, SIGMA, 0.01, kind)))
    assert terms["d2"] == pytest.approx(terms["d1"] - SIGMA * np.sqrt(0.5))
    assert 0 < terms["N_d2"] < 1


def test_curves_match_the_greeks_and_end_at_the_payoff():
    S = np.array([90.0, 100.0, 110.0])
    args = (100.0, 0.25, R, Q, SIGMA, "call")
    np.testing.assert_allclose(analytics.greek_curve("Delta", S, *args),
                               bs.delta(S, 100.0, 0.25, R, SIGMA, Q, "call"))
    np.testing.assert_allclose(analytics.greek_curve("Vega", S, *args),
                               bs.vega(S, 100.0, 0.25, R, SIGMA, Q) / 100)
    np.testing.assert_allclose(analytics.greek_curve("Theta", S, *args),
                               bs.theta(S, 100.0, 0.25, R, SIGMA, Q, "call") / 365)
    np.testing.assert_allclose(analytics.greek_curve("Price", S, 100.0, 0.0, R, Q, SIGMA, "call"),
                               [0.0, 0.0, 10.0])
    assert np.isnan(analytics.greek_curve("Gamma", S, 100.0, 0.0, R, Q, SIGMA, "call")).all()
    # Gamma peaks near the strike.
    assert np.argmax(analytics.greek_curve("Gamma", S, *args)) == 1
    with pytest.raises(ValueError):
        analytics.greek_curve("Rho", S, *args)


# ---- realised vol ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def gbm_close():
    """Ten years of daily closes with a known 20% vol."""
    path = simulate_gbm(100.0, 0.05, SIGMA, 10.0, 2520, 1, seed=12)[0]
    return pd.Series(path, index=pd.bdate_range("2016-01-04", periods=2521))


def test_realised_vol_recovers_the_true_vol(gbm_close):
    assert analytics.realised_vol(gbm_close, 2520).iloc[-1] == pytest.approx(SIGMA, rel=0.05)
    assert analytics.realised_vol(gbm_close, 21).dropna().mean() == pytest.approx(SIGMA, rel=0.05)


def test_vol_cone_narrows_for_longer_windows(gbm_close):
    cone = analytics.vol_cone(gbm_close)
    width = (cone["p95"] - cone["p05"]).to_numpy()
    assert np.all(np.diff(width) < 0)
    assert cone["median"].between(0.17, 0.23).all()


# ---- backtests ------------------------------------------------------------------------

def test_single_backtest_is_the_engine_on_a_real_window(gbm_close):
    run = analytics.hedge_backtest(gbm_close, gbm_close.index[500], 63, 1.0, "call",
                                   SIGMA, SIGMA, R, Q, cost_bp=1.0)
    window = gbm_close.iloc[500:564].to_numpy()[None, :]
    direct = delta_hedge(window, window[0, 0], 0.25, R, SIGMA, cost_rate=1e-4)
    s = run["summary"]
    assert s["pnl"] == pytest.approx(float(direct.pnl[0]))
    assert s["premium"] == pytest.approx(direct.premium)
    assert len(run["days"]) == 64
    # The daily P&L path is before costs, so it ends at P&L + costs.
    assert run["days"]["pnl"].iloc[-1] == pytest.approx(s["pnl"] + s["costs"])


def test_single_backtest_needs_enough_history(gbm_close):
    with pytest.raises(ValueError):
        analytics.hedge_backtest(gbm_close, gbm_close.index[-10], 63, 1.0, "call",
                                 SIGMA, SIGMA, R)


def test_rolling_backtest_pnl_follows_the_vol_gap(gbm_close):
    trades = analytics.rolling_backtest(gbm_close, markup=0.03, r=R)
    gap = trades["sale_vol"] - trades["realised_vol"]
    assert len(trades) > 400
    assert trades["pnl"].mean() > 0                       # sold 3 points rich
    assert np.corrcoef(gap, trades["pnl"])[0, 1] > 0.7    # stage 3: P&L ~ vega x gap


def test_rolling_backtest_can_sell_at_a_quoted_vol_history(gbm_close):
    # A "vol index" that always quotes 25% while the stock realises 20%.
    quoted = pd.Series(0.25, index=gbm_close.index)
    trades = analytics.rolling_backtest(gbm_close, n_days=21, r=R, sale_vols=quoted, markup=0.0)
    assert np.allclose(trades["sale_vol"], 0.25)
    assert trades["pnl"].mean() > 0 and (trades["pnl"] > 0).mean() > 0.7
    # Days with no quote are skipped, not sold at a made-up vol.
    partial = quoted.copy()
    partial.iloc[:1000] = np.nan
    assert len(analytics.rolling_backtest(gbm_close, n_days=21, r=R, sale_vols=partial,
                                           markup=0.0)) < len(trades)


def test_rolling_backtest_costs_reduce_pnl(gbm_close):
    free = analytics.rolling_backtest(gbm_close, step=20, r=R)
    costly = analytics.rolling_backtest(gbm_close, step=20, r=R, cost_bp=5.0)
    assert (costly["pnl"] < free["pnl"]).all()
    np.testing.assert_allclose(free["pnl"] - costly["pnl"], costly["costs"])


def test_contract_replay_hedge_removes_most_of_the_risk(gbm_close):
    expiry, strike = date(2018, 6, 15), 110.0
    days = gbm_close.loc["2018-01-02":"2018-06-14"]
    T = np.array([analytics.year_fraction(expiry, d.date()) for d in days.index])
    option = pd.Series(bs.price(days.to_numpy(), strike, T, R, SIGMA, Q, "call"),
                       index=days.index)
    table = analytics.contract_replay(option, gbm_close, strike, "call", expiry, R, Q)
    np.testing.assert_allclose(table["iv"], SIGMA, atol=1e-4)     # recovers the vol used
    assert table["pnl"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    unhedged_moves = table["option"].diff().dropna()
    hedged_moves = table["pnl"].diff().dropna()
    assert hedged_moves.std() < 0.25 * unhedged_moves.std()


def test_contract_replay_charges_costs_from_day_one(gbm_close):
    expiry, strike = date(2018, 6, 15), 110.0
    days = gbm_close.loc["2018-01-02":"2018-03-01"]
    T = np.array([analytics.year_fraction(expiry, d.date()) for d in days.index])
    option = pd.Series(bs.price(days.to_numpy(), strike, T, R, SIGMA, Q, "call"),
                       index=days.index)
    table = analytics.contract_replay(option, gbm_close, strike, "call", expiry, R, Q,
                                      cost_bp=10.0)
    expected = -1e-3 * table["delta"].iloc[0] * table["spot"].iloc[0]
    assert table["pnl"].iloc[0] == pytest.approx(expected)
