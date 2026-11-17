# Stage 3 results: implied vol vs realised vol

Short 3-month ATM call, S0=K=100, r=3%. Sold at 30% implied vol (premium 6.3372); the stock realises 20% (fair value 4.3576). Real-world drift 8%, 20,000 paths, the same paths for both hedges.

Locked-in target (V_imp - V_real) e^(rT) = **1.9945**.

## Which vol goes into the delta?

| Rebalancing | Hedge delta | Mean P&L | Std P&L | 1st pct | 99th pct | Paths with P&L > 0 | corr with ½∫ΓS²(σi²−σr²)dt |
|---|---|---:|---:|---:|---:|---:|---:|
| ~hourly | implied vol | 1.9970 | 0.6190 | 0.742 | 3.266 | 100.00% | 0.980 |
| ~hourly | realised vol | 1.9944 | 0.1564 | 1.567 | 2.394 | 100.00% |  |
| daily | implied vol | 2.0025 | 0.7498 | 0.558 | 3.765 | 100.00% | 0.885 |
| daily | realised vol | 2.0010 | 0.4431 | 0.796 | 3.099 | 99.91% |  |

## Realised vol varies (sold at 30%, ~hourly hedging, 5,000 paths per row)

| Realised vol | (V_imp − V_real) e^(rT) | vega × (σi − σr) e^(rT) | Implied hedge: mean | std | Realised hedge: mean | std |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | +3.9840 | +3.9882 | +3.9426 | 0.7637 | +3.9840 | 0.0763 |
| 15% | +2.9909 | +2.9911 | +2.9841 | 0.7632 | +2.9911 | 0.1163 |
| 20% | +1.9945 | +1.9941 | +1.9987 | 0.6228 | +1.9945 | 0.1560 |
| 25% | +0.9972 | +0.9970 | +1.0019 | 0.3930 | +0.9970 | 0.1961 |
| 30% | +0.0000 | +0.0000 | -0.0000 | 0.2344 | -0.0000 | 0.2344 |
| 35% | -0.9967 | -0.9970 | -1.0054 | 0.4962 | -0.9967 | 0.2725 |
| 40% | -1.9926 | -1.9941 | -2.0122 | 0.9311 | -1.9923 | 0.3116 |
| 45% | -2.9875 | -2.9911 | -3.0199 | 1.4269 | -2.9873 | 0.3502 |
| 50% | -3.9810 | -3.9882 | -4.0270 | 1.9621 | -3.9810 | 0.3896 |

## Real-world drift varies (sold at 30%, realised 20%, ~hourly hedging, 5,000 paths per row)

| Drift μ | Implied hedge: mean | std | Realised hedge: mean | std |
|---:|---:|---:|---:|---:|
| -60% | 1.4635 | 0.6161 | 1.9867 | 0.1170 |
| -30% | 1.8140 | 0.6433 | 1.9914 | 0.1461 |
| +0% | 2.0031 | 0.6224 | 1.9945 | 0.1561 |
| +8% | 1.9987 | 0.6228 | 1.9945 | 0.1560 |
| +30% | 1.8641 | 0.6334 | 1.9925 | 0.1471 |
| +60% | 1.5293 | 0.6170 | 1.9869 | 0.1209 |
