# Stage 5 results: hedging with the wrong model

Short 3-month ATM call, S0=K=100, r=3%, sold and hedged with Black-Scholes at 20% (premium 4.3576). Real-world drift 8%, 20,000 paths per model. All three models have the same average variance (20% vol).

- **Heston**: kappa=3, theta=v0=0.04, xi=0.4, rho=-0.7
- **Merton**: 1 jump/year, log jump ~ N(-0.08, 0.08²), diffusive vol 16.5%

## Hedged P&L by model and rebalancing frequency

Excess kurtosis: a normal distribution has 0. ES 1% is the average of the worst 1% of paths.

| Model | Rebalances | Mean | Std | Skew | Excess kurtosis | 1st pct | ES 1% | Worst path | corr with gamma attribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GBM | 12 | +0.0133 | 0.9692 | -0.36 | 1.3 | -2.725 | -3.362 | -5.51 | 0.961 |
| GBM | 21 | +0.0095 | 0.7417 | -0.28 | 1.6 | -1.979 | -2.563 | -5.15 | 0.971 |
| GBM | 63 | +0.0065 | 0.4431 | -0.27 | 2.1 | -1.199 | -1.524 | -3.55 | 0.984 |
| GBM | 126 | +0.0051 | 0.3154 | -0.24 | 2.5 | -0.849 | -1.084 | -3.24 | 0.987 |
| GBM | 252 | +0.0021 | 0.2226 | -0.10 | 1.8 | -0.597 | -0.743 | -1.31 | 0.991 |
| GBM | 504 | -0.0001 | 0.1564 | -0.08 | 1.9 | -0.427 | -0.526 | -1.08 | 0.993 |
| Heston | 12 | +0.0384 | 1.2378 | -1.08 | 3.9 | -3.846 | -5.026 | -13.68 | 0.965 |
| Heston | 21 | +0.0329 | 1.0631 | -1.01 | 3.5 | -3.314 | -4.273 | -9.93 | 0.978 |
| Heston | 63 | +0.0362 | 0.8618 | -0.85 | 2.8 | -2.594 | -3.349 | -6.73 | 0.992 |
| Heston | 126 | +0.0370 | 0.8054 | -0.78 | 2.5 | -2.394 | -3.047 | -6.41 | 0.997 |
| Heston | 252 | +0.0375 | 0.7685 | -0.66 | 1.9 | -2.227 | -2.802 | -5.10 | 0.999 |
| Heston | 504 | +0.0393 | 0.7520 | -0.62 | 1.7 | -2.232 | -2.724 | -5.09 | 1.000 |
| Merton | 12 | +0.0907 | 1.9742 | -3.44 | 18.4 | -8.540 | -11.406 | -24.76 | 0.866 |
| Merton | 21 | +0.0883 | 1.9093 | -3.63 | 18.7 | -8.616 | -11.094 | -23.41 | 0.859 |
| Merton | 63 | +0.0893 | 1.8503 | -4.00 | 20.9 | -8.563 | -11.114 | -22.61 | 0.832 |
| Merton | 126 | +0.0917 | 1.8365 | -4.13 | 22.0 | -8.618 | -11.140 | -23.00 | 0.816 |
| Merton | 252 | +0.0898 | 1.8304 | -4.19 | 22.3 | -8.587 | -11.178 | -22.47 | 0.813 |
| Merton | 504 | +0.0896 | 1.8266 | -4.21 | 22.3 | -8.516 | -11.188 | -23.00 | 0.813 |

## The fair smile under each model

Implied vol of the Monte Carlo price (40,000 paths, risk-neutral drift, no risk premium). The GBM row should be flat at 20%; its deviation shows the Monte Carlo error.

| Model | K=80 | K=85 | K=90 | K=95 | K=100 | K=105 | K=110 | K=115 | K=120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GBM | 19.7% | 20.0% | 20.0% | 20.0% | 20.1% | 20.0% | 20.0% | 20.0% | 19.9% |
| Heston | 25.0% | 23.6% | 22.3% | 20.9% | 19.5% | 18.1% | 16.9% | 15.9% | 15.3% |
| Merton | 24.0% | 22.5% | 21.1% | 20.0% | 19.4% | 18.8% | 18.4% | 18.1% | 17.9% |

## Selling other strikes at a flat 20% (daily hedging)

| Option | Model | Fair implied vol | Premium at 20% | Mean hedged P&L | Std error |
|---|---|---:|---:|---:|---:|
| 90 put | GBM | 20.0% | 0.6122 | -0.0021 | 0.0018 |
| 90 put | Heston | 22.3% | 0.6122 | -0.2483 | 0.0046 |
| 90 put | Merton | 21.1% | 0.6122 | -0.1561 | 0.0108 |
| 100 call | GBM | 20.1% | 4.3576 | +0.0065 | 0.0031 |
| 100 call | Heston | 19.5% | 4.3576 | +0.0362 | 0.0061 |
| 100 call | Merton | 19.4% | 4.3576 | +0.0893 | 0.0131 |
| 110 call | GBM | 20.0% | 1.0913 | -0.0009 | 0.0027 |
| 110 call | Heston | 16.9% | 1.0913 | +0.4060 | 0.0057 |
| 110 call | Merton | 18.4% | 1.0913 | +0.2166 | 0.0085 |
