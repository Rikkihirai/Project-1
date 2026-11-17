# Delta-hedging study: notes for Claude Code

## Purpose
Portfolio project for applying to quant research / e-trading roles at bulge-bracket
banks. The owner must be able to explain every line in an interview, so when
writing code: explain the maths behind each step, keep functions small and readable,
and prefer clarity over cleverness.

## Layout
- `hedging/bs.py`: vectorised Black-Scholes price + Greeks (numpy/scipy, broadcasts over (path, time)), `implied_vol()`
- `hedging/paths.py`: `simulate_gbm`, `simulate_heston` (full-truncation Euler, returns (prices, variances)),
  `simulate_merton` (jump-diffusion; equals GBM for the same seed when lam=0)
- `hedging/hedge.py`: `delta_hedge()` sells an option, delta-hedges at every grid date and
  returns `HedgeResult` (pnl, gamma_pnl attribution, costs, deltas, premium). P&L is valued at T.
  `mark_to_market()` gives the P&L at every grid date; `variance_edge_pnl()` is ½ΣΓᵢS²(σᵢ²−σᵣ²)dt.
  `delta_hedge(cost_rate=, band=)` does costs and move-based hedging (`band_deltas`); `leland_vol()`.
- `pricing/american.py`: `binomial_price()` (CRR tree), `longstaff_schwartz()` (returns (price, std error))
- `pricing/pde.py`: `fd_price()` / `fd_solve()` Crank-Nicolson; `american=`, `lower_barrier=`, `upper_barrier=`
- `pricing/heston.py`: `HestonParams`, `heston_price()` (Fourier, vectorised over strikes), `heston_smile()`
- `pricing/calibration.py`: `calibrate_heston()` least squares on (model − market price) / vega
- `desk/`: Streamlit dashboard (`python3 -m streamlit run desk/app.py`). `data.py` = `PublicData` (default: Cboe
  delayed quote file + VIX history, Yahoo daily bars; no keys), `AlpacaData` (keys from env or `.env`) and
  `SyntheticData` (Heston demo market), daily bars cached in `data_cache/`; `analytics.py` = pure functions;
  `app.py` = layout only, one screen computed at a time; `charts.hover_lines()` = Altair charts with hover.
  After editing `desk/*.py`, restart the Streamlit server: it keeps imported modules in memory. Never fetch market data through chat tools: output is huge.
- `experiments/stageN_*.py`: one script per study stage; writes charts + a summary .md to `results/`
- `tests/`: pytest; run `python3 -m pytest -q` from this folder

## Conventions
- Docstrings carry the formulas (same style as `../task1_black_scholes`).
- Every new feature gets tests that check a property from theory, not just "it runs".
- Experiments reuse the same simulated paths across settings so only one thing changes.
- Charts: matplotlib, palette and rcParams as in `experiments/stage2_rebalancing.py`
  (blue #2a78d6, orange #eb6834, aqua #1baf7a; light surface #fcfcfb; no dual axes).
- Update the README findings + roadmap checkboxes when a stage is done.
- This Mac is slow (one hourly-grid hedge of 20,000 paths takes ~4 s): keep sweeps to ~5,000 paths.

## Status
- [x] Stage 1: pricer + Greeks
- [x] Stage 2: hedging error vs rebalancing frequency (std ~ 1/sqrt(N), matches Derman-Kamal)
- [x] Stage 3: implied != realised vol (sold at 30%, realises 20%). Both hedges average
      (V_i − V_r)e^{rT} = 1.99; realised-vol delta locks it in (std 0.16 hourly), implied-vol delta
      is smooth but path-dependent (std 0.62, peaks when S_T ≈ K, depends on drift). Ahmad & Wilmott (2005).
- [x] Stage 4: 5 bp costs. Rebalancing cost = c·vega·√(2N/(πT)) (slope +0.54), so cost × risk is
      constant for a clock hedger; a delta band saves 12–17% risk at equal cost; Leland's vol
      recovers the rebalancing cost (not the on/off cost). Band trades back to the model delta;
      trading to the band edge (Whalley-Wilmott) is not implemented.
- [x] Stage 5: Heston and Merton paths (same average 20% vol) hedged with BS at 20%. Hourly std:
      GBM 0.16, Heston 0.75, Merton 1.83; more hedging barely helps. Merton worst-1% average −11
      on a 4.36 premium. Fair smile by Monte Carlo (no risk premium) explains mean P&L by strike.
- [x] Stage 5b: pricing toolkit in `pricing/`. Tree, PDE and Longstaff-Schwartz agree on an American put
      (4.487 / 4.486 / 4.472±0.009); Fourier Heston matches Broadie-Kaya 6.8061 and stage 5's Monte Carlo
      smile; calibration recovers a noisy synthetic smile. Not yet calibrated to real quotes.
- [x] Stage 5c: the desk. All four screens verified in the browser on real data from the public source
      (SPY; SPX checked at the data layer only). Alpaca source never worked live: the keys in `.env` are
      rejected (401) and look truncated. Yahoo/Cboe endpoints are unofficial and may break.
      Real-data quirks handled: non-OCC contract symbols, SPX/SPXW duplicate listings, New York date.
- [ ] Stage 6 (next): write-up on real SPY/SPX data (first numbers are in the README's desk section).
      SPY options are American: use the tree or PDE.
      The owner wants an interface traders would use daily, fed by historical and live data.
- [ ] Stage 7: deep hedging
- [ ] Stage 8: options market-making layer (Avellaneda-Stoikov skew)
