# Stage 4 results: transaction costs

Short 3-month ATM call, S0=K=100, r=3%, sigma=20% (priced, hedged and realised), drift 8%. Cost 5 bp of notional per trade, one way. 20,000 paths, the same for every row. Premium = 4.3576.

## Time-based hedging

Fitted log-log slope of rebalancing cost vs N: **+0.538** (theory: +0.5). "Rebalancing cost" excludes putting the hedge on and unwinding it; the prediction is c·vega·√(2N/(πT)).

| Rebalances | Frequency | Mean P&L | Std P&L | Total cost | Rebalancing cost | Predicted | Cost % of premium | 1st pct P&L |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | every ~3 weeks | -0.0846 | 1.6305 | 0.0832 | 0.0255 | 0.0316 | 1.9% | -4.725 |
| 6 | every ~2 weeks | -0.0711 | 1.3526 | 0.0919 | 0.0341 | 0.0387 | 2.1% | -3.833 |
| 12 | ~weekly | -0.0969 | 0.9759 | 0.1102 | 0.0521 | 0.0547 | 2.5% | -2.867 |
| 21 | every 3 days | -0.1196 | 0.7480 | 0.1291 | 0.0709 | 0.0724 | 3.0% | -2.135 |
| 63 | daily | -0.1772 | 0.4512 | 0.1837 | 0.1254 | 0.1253 | 4.2% | -1.446 |
| 126 | twice daily | -0.2314 | 0.3276 | 0.2365 | 0.1782 | 0.1773 | 5.4% | -1.179 |
| 252 | 4x daily | -0.3089 | 0.2460 | 0.3110 | 0.2526 | 0.2507 | 7.1% | -1.043 |
| 504 | ~hourly | -0.4160 | 0.2112 | 0.4159 | 0.3576 | 0.3545 | 9.5% | -1.037 |

## Delta-band hedging (checked 504 times, ~hourly)

"Clock std at same cost" is the risk a time-based hedger gets for the same total spend (interpolated from the table above).

| Band (deltas) | Trades per path | Mean P&L | Std P&L | Total cost | Clock std at same cost | Risk saved |
|---:|---:|---:|---:|---:|---:|---:|
| 0.010 | 251.9 | -0.3785 | 0.2150 | 0.3780 | 0.2221 | 3% |
| 0.020 | 148.2 | -0.3224 | 0.2251 | 0.3213 | 0.2418 | 7% |
| 0.030 | 94.3 | -0.2757 | 0.2449 | 0.2755 | 0.2792 | 12% |
| 0.050 | 47.3 | -0.2181 | 0.3033 | 0.2173 | 0.3646 | 17% |
| 0.075 | 25.4 | -0.1818 | 0.3979 | 0.1770 | 0.4761 | 16% |
| 0.100 | 15.9 | -0.1539 | 0.4977 | 0.1528 | 0.5873 | 15% |
| 0.150 | 7.7 | -0.1398 | 0.7870 | 0.1224 | 0.8183 | 4% |
| 0.200 | 4.7 | -0.1159 | 0.8920 | 0.1109 | 0.9658 | 8% |

## Leland's adjusted vol

Sold and hedged at sigma versus sold and hedged at Leland's vol, both paying costs. "On/off cost" is the part Leland's formula does not cover: putting the hedge on and unwinding it.

| Rebalances | Leland vol | Leland premium | Mean P&L at sigma | Mean P&L at Leland vol | On/off cost | Std at sigma | Std at Leland vol |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 20.27% | 4.4120 | -0.0969 | -0.0418 | 0.0580 | 0.9759 | 0.9761 |
| 63 | 20.62% | 4.4810 | -0.1772 | -0.0508 | 0.0583 | 0.4512 | 0.4499 |
| 252 | 21.23% | 4.6009 | -0.3089 | -0.0561 | 0.0583 | 0.2460 | 0.2296 |
| 504 | 21.72% | 4.6976 | -0.4160 | -0.0583 | 0.0583 | 0.2112 | 0.1642 |
