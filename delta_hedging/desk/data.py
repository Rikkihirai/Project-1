"""Market data for the desk: stock prices, option chains, option history.

Three sources with the same methods, so the rest of the code does not
care which one it is talking to:

    PublicData     real data from public pages, no account needed:
                   Cboe delayed option quotes and VIX history, Yahoo
                   Finance daily prices
    AlpacaData     real data from Alpaca's market-data REST API (needs keys)
    SyntheticData  made-up data from this project's own Heston model, so
                   the dashboard and the tests run offline

    spot(symbol)                      latest price
    daily_bars(symbol, start, end)    one row per trading day
    option_chain(symbol, ...)         every listed contract, with quotes
    option_bars(contract, start, end) daily prices of one contract
    implied_vol_history(symbol)       daily 30-day implied vol, if known

Public sources
--------------
  - Cboe publishes its delayed (about 15 minutes) quote file for every
    optionable symbol: all strikes and expiries in one download.
  - Cboe's VIX history (daily since 1990) is the market's own 30-day
    implied vol for the S&P 500. It gives backtests a real sale vol.
  - Yahoo Finance's chart service gives daily closes back to 2016 and
    beyond, for the stock and for individual option contracts.
  These are unofficial, unsupported endpoints: fine for study, not for
  trading, and they can change or disappear without notice.
  Use SPX for the index itself (European, cash-settled options) or SPY
  for the ETF that tracks it (American options).

Alpaca notes
------------
Keys are read from the environment (APCA_API_KEY_ID, APCA_API_SECRET_KEY)
or from a `.env` file in the project folder. They are never written
anywhere by this code.

  - Stock history goes back to 2016. Prices are adjusted for splits only,
    not dividends, so they are on the same scale as option strikes.
  - Option quotes use the free "indicative" feed: delayed about 15 minutes.
    The real-time "opra" feed needs a signed agreement and a subscription.
  - Daily stock bars are cached on disk (data_cache/), so years of history
    are downloaded once and only new days are fetched afterwards.

Contract symbols follow the OCC format: root, expiry as YYMMDD, C or P,
strike times 1000 in eight digits. SPY261218C00750000 is the SPY call
expiring 18 December 2026 with strike 750.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data_cache"
DATA_URL = "https://data.alpaca.markets"
OCC_PATTERN = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

CHAIN_COLUMNS = ["contract", "expiry", "kind", "strike", "bid", "ask", "mid",
                 "last", "quote_iv", "volume"]
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """The data source could not provide what was asked for."""


# ---- contract symbols ---------------------------------------------------------

@dataclass(frozen=True)
class Contract:
    root: str
    expiry: date
    kind: str        # "call" or "put"
    strike: float


def parse_contract(symbol: str) -> Contract:
    """Split an OCC option symbol into its parts."""
    match = OCC_PATTERN.match(symbol)
    if not match:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, yymmdd, cp, strike = match.groups()
    expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    return Contract(root, expiry, "call" if cp == "C" else "put", int(strike) / 1000.0)


def contract_symbol(root: str, expiry: date, kind: str, strike: float) -> str:
    """Build an OCC option symbol."""
    cp = "C" if kind == "call" else "P"
    return f"{root}{expiry:%y%m%d}{cp}{round(strike * 1000):08d}"


def market_today() -> date:
    """Today's date in New York, where US options trade and expire."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()


def _empty_chain() -> pd.DataFrame:
    return pd.DataFrame(columns=CHAIN_COLUMNS)


def _bars_frame(rows: list) -> pd.DataFrame:
    """Alpaca bar records -> DataFrame indexed by date."""
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    frame = pd.DataFrame(rows).rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    frame["date"] = pd.to_datetime(frame["t"]).dt.tz_localize(None).dt.normalize()
    return frame.set_index("date")[BAR_COLUMNS].astype(float)


# ---- shared: network requests and the disk cache ---------------------------------

def fetch(session, url: str, attempts: int = 2, **kwargs):
    """GET a URL, retrying once on a network failure.

    Servers drop connections that have sat idle, and the first request
    after a pause then fails with "connection reset". A second try opens a
    fresh connection. Anything still failing becomes a DataError, which the
    dashboard shows as a message instead of a crash.
    """
    import requests
    for attempt in range(attempts):
        try:
            return session.get(url, timeout=30, **kwargs)
        except requests.RequestException as error:
            if attempt == attempts - 1:
                raise DataError(f"network problem reaching {url.split('/')[2]}: "
                                f"{type(error).__name__}") from error


class CachedBars:
    """Adds a disk cache to any source that can fetch daily bars.

    A subclass provides `_fetch_daily_bars(symbol, start, end)` and a
    `cache_tag` that keeps its files apart from other sources' files.
    """

    cache_dir: Path | None = CACHE_DIR
    cache_tag = ""

    def daily_bars(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        """Daily OHLCV, adjusted for splits. Cached on disk.

        The cache file holds everything fetched so far, and a small note
        beside it records the earliest date ever asked for (the first row
        is later than that whenever the request began on a holiday). Only
        days after the last cached row are requested again, plus a few in
        case of late revisions.
        """
        end = end or date.today()
        cached, covered_from = None, None
        if self.cache_dir is not None:
            stem = f"{symbol}_1Day{self.cache_tag}".replace("^", "")
            cache_file = self.cache_dir / f"{stem}.csv"
            start_note = self.cache_dir / f"{stem}.start.txt"
            if cache_file.exists() and start_note.exists():
                cached = pd.read_csv(cache_file, index_col="date", parse_dates=True)
                covered_from = date.fromisoformat(start_note.read_text().strip())

        if cached is None or cached.empty or covered_from > start:
            bars = self._fetch_daily_bars(symbol, start, end)
            covered_from = start
        else:
            refresh_from = cached.index[-1].date() - timedelta(days=5)
            fresh = self._fetch_daily_bars(symbol, refresh_from, end)
            bars = pd.concat([cached[cached.index < pd.Timestamp(refresh_from)], fresh])
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()

        if self.cache_dir is not None and not bars.empty:
            self.cache_dir.mkdir(exist_ok=True)
            bars.to_csv(cache_file)
            start_note.write_text(covered_from.isoformat())
        return bars.loc[pd.Timestamp(start):pd.Timestamp(end)]


# ---- public sources: Cboe and Yahoo Finance --------------------------------------

CBOE_URL = "https://cdn.cboe.com/api/global"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
# How each service names an index (ETFs and stocks use the plain ticker).
INDEX_NAMES = {"SPX": {"cboe": "_SPX", "yahoo": "^GSPC"},
               "NDX": {"cboe": "_NDX", "yahoo": "^NDX"},
               "RUT": {"cboe": "_RUT", "yahoo": "^RUT"}}
# The 30-day implied vol index Cboe publishes for each underlying.
VOL_INDEX = {"SPY": "VIX", "SPX": "VIX", "QQQ": "VXN", "NDX": "VXN",
             "IWM": "RVX", "RUT": "RVX"}


class PublicData(CachedBars):
    """Real market data from public pages: no account, no keys."""

    name = "Public (Cboe + Yahoo)"
    quote_delay = "option quotes delayed ~15 min (Cboe delayed feed)"
    cache_tag = "_yahoo"

    def __init__(self, session=None, cache_dir: Path | None = CACHE_DIR,
                 quote_ttl: float = 30.0):
        import requests
        self.session = session or requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; options-study)"}
        self.cache_dir = cache_dir
        self.quote_ttl = quote_ttl          # seconds a downloaded quote file is reused
        self._quotes: dict = {}             # symbol -> (time fetched, spot, chain)

    def _get(self, url: str, params: dict | None = None):
        response = fetch(self.session, url, params=params, headers=self.headers)
        if response.status_code != 200:
            raise DataError(f"{url} returned {response.status_code}")
        return response

    # -- Cboe: one file holds the spot and every option quote -------------------
    def _quote_file(self, symbol: str):
        """(spot, full chain) for a symbol, downloaded at most every quote_ttl s."""
        import time
        held = self._quotes.get(symbol)
        if held and time.monotonic() - held[0] < self.quote_ttl:
            return held[1], held[2]
        name = INDEX_NAMES.get(symbol, {}).get("cboe", symbol)
        data = self._get(f"{CBOE_URL}/delayed_quotes/options/{name}.json").json()["data"]
        rows = []
        for quote in data.get("options", []):
            try:
                parsed = parse_contract(quote["option"])
            except ValueError:
                continue        # non-standard contracts (e.g. after a corporate action)
            bid, ask = float(quote.get("bid") or 0.0), float(quote.get("ask") or 0.0)
            mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else np.nan
            rows.append([quote["option"], parsed.expiry, parsed.kind, parsed.strike, bid,
                         ask, mid, float(quote.get("last_trade_price") or np.nan),
                         float(quote.get("iv") or np.nan), float(quote.get("volume") or 0.0)])
        # An index can list the same expiry, strike and type under two roots
        # (SPX monthly and SPXW weekly). Keep the one that traded more.
        chain = (pd.DataFrame(rows, columns=CHAIN_COLUMNS)
                 .sort_values("volume", ascending=False)
                 .drop_duplicates(["expiry", "strike", "kind"])
                 .sort_values(["expiry", "strike", "kind"]).reset_index(drop=True))
        spot = float(data["current_price"])
        self._quotes[symbol] = (time.monotonic(), spot, chain)
        return spot, chain

    def spot(self, symbol: str) -> float:
        return self._quote_file(symbol)[0]

    def option_chain(self, symbol: str, expiry_from: date | None = None,
                     expiry_to: date | None = None, strike_from: float | None = None,
                     strike_to: float | None = None) -> pd.DataFrame:
        """Latest quote for every listed contract in the requested range."""
        chain = self._quote_file(symbol)[1]
        keep = pd.Series(True, index=chain.index)
        if expiry_from is not None:
            keep &= chain["expiry"] >= expiry_from
        if expiry_to is not None:
            keep &= chain["expiry"] <= expiry_to
        if strike_from is not None:
            keep &= chain["strike"] >= strike_from
        if strike_to is not None:
            keep &= chain["strike"] <= strike_to
        return chain[keep].reset_index(drop=True)

    def implied_vol_history(self, symbol: str):
        """Daily closes of the Cboe vol index for this underlying, as decimals.

        VIX is the 30-day implied vol of S&P 500 options, computed by Cboe
        from the whole smile. Returns None if Cboe publishes no such index.
        """
        index = VOL_INDEX.get(symbol)
        if index is None:
            return None
        cache_file = None if self.cache_dir is None else self.cache_dir / f"{index}_history.csv"
        fresh_enough = (cache_file is not None and cache_file.exists() and
                        date.fromtimestamp(cache_file.stat().st_mtime) == date.today())
        if fresh_enough:
            text = cache_file.read_text()
        else:
            text = self._get(f"{CBOE_URL}/us_indices/daily_prices/{index}_History.csv").text
            if cache_file is not None:
                cache_file.parent.mkdir(exist_ok=True)
                cache_file.write_text(text)
        import io
        table = pd.read_csv(io.StringIO(text))
        days = pd.to_datetime(table["DATE"], format="%m/%d/%Y").rename("date")
        return pd.Series(table["CLOSE"].to_numpy() / 100.0, index=days, name=index)

    # -- Yahoo Finance: daily history ----------------------------------------------
    def _yahoo_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        as_seconds = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        params = {"period1": as_seconds(start), "period2": as_seconds(end + timedelta(days=1)),
                  "interval": "1d"}
        result = (self._get(f"{YAHOO_URL}/{ticker}", params).json()["chart"]["result"] or [None])[0]
        if not result or not result.get("timestamp"):
            return _bars_frame([])
        quote = result["indicators"]["quote"][0]
        days = pd.to_datetime(result["timestamp"], unit="s").normalize().rename("date")
        frame = pd.DataFrame({name: quote.get(name) for name in BAR_COLUMNS}, index=days)
        return frame.dropna(subset=["close"]).astype(float)

    def _fetch_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self._yahoo_daily(INDEX_NAMES.get(symbol, {}).get("yahoo", symbol), start, end)

    def option_bars(self, contract: str, start: date, end: date | None = None) -> pd.DataFrame:
        """Daily OHLCV of one option contract (days it did not trade are absent)."""
        return self._yahoo_daily(contract, start, end or date.today())


# ---- Alpaca -------------------------------------------------------------------

def load_keys(env_file: Path = ROOT / ".env"):
    """Alpaca keys from the environment, else from a .env file. (id, secret) or None."""
    values = dict(os.environ)
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                values.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    key_id, secret = values.get("APCA_API_KEY_ID"), values.get("APCA_API_SECRET_KEY")
    return (key_id, secret) if key_id and secret else None


class AlpacaData(CachedBars):
    """Real market data from Alpaca."""

    name = "Alpaca"
    quote_delay = "option quotes delayed ~15 min (free indicative feed)"

    def __init__(self, key_id: str, secret: str, session=None,
                 cache_dir: Path | None = CACHE_DIR):
        import requests
        self.session = session or requests.Session()
        self.headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self.cache_dir = cache_dir

    def _get(self, path: str, params: dict) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        response = fetch(self.session, DATA_URL + path, params=params, headers=self.headers)
        if response.status_code == 401:
            raise DataError("Alpaca rejected the API keys in .env. Check that the key ID "
                            "and the secret were pasted in full.")
        if response.status_code != 200:
            raise DataError(f"Alpaca returned {response.status_code} for {path}: "
                            f"{response.text[:200]}")
        return response.json()

    def _pages(self, path: str, params: dict, field: str):
        """Yield the `field` of every page, following next_page_token."""
        token = None
        while True:
            page = self._get(path, {**params, "page_token": token})
            yield page.get(field) or {}
            token = page.get("next_page_token")
            if not token:
                return

    def spot(self, symbol: str) -> float:
        snapshot = self._get(f"/v2/stocks/{symbol}/snapshot", {"feed": "iex"})
        trade = snapshot.get("latestTrade") or {}
        if "p" not in trade:
            raise DataError(f"no recent trade for {symbol}")
        return float(trade["p"])

    def _fetch_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        # The free plan may not query the last 15 minutes of the full feed.
        latest = datetime.now(timezone.utc) - timedelta(minutes=20)
        end_time = min(datetime.combine(end, datetime.max.time(), timezone.utc), latest)
        params = {"symbols": symbol, "timeframe": "1Day", "adjustment": "split",
                  "start": f"{start:%Y-%m-%d}", "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "limit": 10000, "feed": "sip"}
        rows = []
        for page in self._pages("/v2/stocks/bars", params, "bars"):
            rows.extend(page.get(symbol, []))
        return _bars_frame(rows)

    def option_chain(self, symbol: str, expiry_from: date | None = None,
                     expiry_to: date | None = None, strike_from: float | None = None,
                     strike_to: float | None = None) -> pd.DataFrame:
        """Latest quote for every listed contract in the requested range."""
        params = {"feed": "indicative", "limit": 1000,
                  "expiration_date_gte": expiry_from and f"{expiry_from:%Y-%m-%d}",
                  "expiration_date_lte": expiry_to and f"{expiry_to:%Y-%m-%d}",
                  "strike_price_gte": strike_from, "strike_price_lte": strike_to}
        rows = []
        for page in self._pages(f"/v1beta1/options/snapshots/{symbol}", params, "snapshots"):
            for contract, snap in page.items():
                rows.append(_chain_row(contract, snap))
        if not rows:
            return _empty_chain()
        return (pd.DataFrame(rows, columns=CHAIN_COLUMNS)
                .sort_values(["expiry", "strike", "kind"]).reset_index(drop=True))

    def implied_vol_history(self, symbol: str):
        return None                      # Alpaca has no vol-index history

    def option_bars(self, contract: str, start: date, end: date | None = None) -> pd.DataFrame:
        """Daily OHLCV of one option contract."""
        end = end or date.today()
        params = {"symbols": contract, "timeframe": "1Day", "limit": 10000,
                  "start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"}
        rows = []
        for page in self._pages("/v1beta1/options/bars", params, "bars"):
            rows.extend(page.get(contract, []))
        return _bars_frame(rows)


def _chain_row(contract: str, snap: dict) -> list:
    """One Alpaca option snapshot -> one row of the chain table."""
    parsed = parse_contract(contract)
    quote = snap.get("latestQuote") or {}
    bid, ask = float(quote.get("bp") or 0.0), float(quote.get("ap") or 0.0)
    mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else np.nan
    last = (snap.get("latestTrade") or {}).get("p", np.nan)
    volume = (snap.get("dailyBar") or {}).get("v", 0.0)
    return [contract, parsed.expiry, parsed.kind, parsed.strike, bid, ask, mid,
            float(last), float(snap.get("impliedVolatility") or np.nan), float(volume)]


# ---- synthetic ----------------------------------------------------------------

class SyntheticData:
    """Made-up but self-consistent market, generated by the Heston model.

    A single ten-year Heston path provides the daily prices and the daily
    variance. Option quotes on any day are Heston prices given that day's
    price and variance, with a bid-ask spread around them. Nothing here is
    real; it exists so the dashboard and the tests run without an account.
    """

    name = "Demo (synthetic)"
    quote_delay = "synthetic data, not real quotes"

    def __init__(self, today: date | None = None, years: int = 10, seed: int = 7,
                 start_price: float = 250.0, r: float = 0.04, q: float = 0.012):
        from hedging.paths import simulate_heston
        from pricing.heston import HestonParams
        self.params = HestonParams(v0=0.03, kappa=3.0, theta=0.03, xi=0.45, rho=-0.7)
        self.r, self.q = r, q
        self.today = today or date.today()
        days = pd.bdate_range(end=pd.Timestamp(self.today), periods=252 * years)
        p = self.params
        prices, variances = simulate_heston(
            start_price, 0.09, p.v0, p.kappa, p.theta, p.xi, p.rho,
            T=len(days) / 252, n_steps=len(days) - 1, n_paths=1, seed=seed)
        close = pd.Series(prices[0], index=days.rename("date"))
        self.variance = pd.Series(variances[0], index=days)
        wiggle = np.random.default_rng(seed).uniform(0.001, 0.008, (2, len(days)))
        self.bars = pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]), "high": close * (1 + wiggle[0]),
            "low": close * (1 - wiggle[1]), "close": close, "volume": 5e7})

    def spot(self, symbol: str) -> float:
        return float(self.bars["close"].iloc[-1])

    def daily_bars(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        end = end or self.today
        return self.bars.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def implied_vol_history(self, symbol: str):
        """The model's own instantaneous vol each day, standing in for a VIX."""
        return np.sqrt(self.variance).rename("model vol")

    def _params_on(self, day) -> "HestonParams":
        from dataclasses import replace
        return replace(self.params, v0=max(float(self.variance.loc[day]), 1e-4))

    def _expiries(self, count: int = 6) -> list:
        """The next `count` third Fridays."""
        found, month_start = [], pd.Timestamp(self.today).replace(day=1)
        while len(found) < count:
            fridays = pd.date_range(month_start, periods=31, freq="D")
            fridays = [d for d in fridays if d.weekday() == 4 and d.month == month_start.month]
            if fridays[2].date() > self.today + timedelta(days=6):
                found.append(fridays[2].date())
            month_start += pd.offsets.MonthBegin(1)
        return found

    def option_chain(self, symbol: str, expiry_from: date | None = None,
                     expiry_to: date | None = None, strike_from: float | None = None,
                     strike_to: float | None = None) -> pd.DataFrame:
        from pricing.heston import heston_price
        from .analytics import implied_vols
        spot = self.spot(symbol)
        step = 5.0 if spot > 200 else 1.0
        strikes = np.arange(np.floor(0.8 * spot / step) * step, 1.2 * spot + step, step)
        strikes = strikes[(strikes >= (strike_from or 0)) & (strikes <= (strike_to or np.inf))]
        params = self._params_on(self.bars.index[-1])
        rows = []
        for expiry in self._expiries():
            if (expiry_from and expiry < expiry_from) or (expiry_to and expiry > expiry_to):
                continue
            T = (expiry - self.today).days / 365.0
            for kind in ("call", "put"):
                mids = np.atleast_1d(heston_price(spot, strikes, T, self.r, params, self.q, kind))
                vols = implied_vols(mids, spot, strikes, T, self.r, self.q, kind)
                for strike, mid, iv in zip(strikes, mids, vols):
                    if mid < 0.03:
                        continue
                    half_spread = max(0.01, 0.01 * mid)
                    rows.append([contract_symbol(symbol, expiry, kind, strike), expiry,
                                 kind, float(strike), round(mid - half_spread, 2),
                                 round(mid + half_spread, 2), float(mid), float(mid), iv, 100.0])
        return pd.DataFrame(rows, columns=CHAIN_COLUMNS) if rows else _empty_chain()

    def option_bars(self, contract: str, start: date, end: date | None = None) -> pd.DataFrame:
        from pricing.heston import heston_price
        parsed = parse_contract(contract)
        last_day = min(end or self.today, parsed.expiry - timedelta(days=1))
        days = self.bars.loc[pd.Timestamp(start):pd.Timestamp(last_day)].index
        closes = [float(heston_price(self.bars["close"].loc[day], parsed.strike,
                                     (parsed.expiry - day.date()).days / 365.0, self.r,
                                     self._params_on(day), self.q, parsed.kind))
                  for day in days]
        close = pd.Series(closes, index=days, dtype=float)
        return pd.DataFrame({"open": close, "high": close, "low": close,
                             "close": close, "volume": 100.0})


def available_sources() -> list:
    """Names of the sources that can be used right now, best first."""
    names = [PublicData.name]
    if load_keys():
        names.append(AlpacaData.name)
    return names + [SyntheticData.name]


def connect(name: str | None = None):
    """Open a data source by name (default: public data)."""
    name = name or PublicData.name
    if name == PublicData.name:
        return PublicData()
    if name == AlpacaData.name:
        keys = load_keys()
        if not keys:
            raise DataError("no Alpaca keys found in the environment or .env")
        return AlpacaData(*keys)
    if name == SyntheticData.name:
        return SyntheticData()
    raise ValueError(f"unknown data source: {name!r}")
