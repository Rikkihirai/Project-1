# Stage 2 results: hedging error vs rebalancing frequency

Short 3-month ATM call, S0=K=100, r=3%, sigma=20% (priced, hedged and realised), real-world drift 8%. 20,000 paths. Premium = 4.3576.

Fitted log-log slope of std vs N: **-0.482** (theory: -0.5).

| Rebalances | Frequency | Mean P&L | Std P&L | Std % of premium | Derman-Kamal std | 1st pct P&L | corr(P&L, gamma attribution) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | every ~3 weeks | -0.0015 | 1.6242 | 37.3% | 1.7540 | -4.637 | 0.934 |
| 6 | every ~2 weeks | +0.0208 | 1.3461 | 30.9% | 1.4321 | -3.718 | 0.945 |
| 12 | ~weekly | +0.0133 | 0.9692 | 22.2% | 1.0127 | -2.725 | 0.961 |
| 21 | every 3 days | +0.0095 | 0.7417 | 17.0% | 0.7655 | -1.979 | 0.971 |
| 63 | daily | +0.0065 | 0.4431 | 10.2% | 0.4420 | -1.199 | 0.984 |
| 126 | twice daily | +0.0051 | 0.3154 | 7.2% | 0.3125 | -0.849 | 0.987 |
| 252 | 4x daily | +0.0021 | 0.2226 | 5.1% | 0.2210 | -0.597 | 0.991 |
| 504 | ~hourly | -0.0001 | 0.1564 | 3.6% | 0.1563 | -0.427 | 0.993 |
