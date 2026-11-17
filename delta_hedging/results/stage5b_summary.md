# Stage 5b results: pricing toolkit

## Convergence to the Black-Scholes formula

European call, S0=K=100, 3 months, r=3%, vol 20%. Exact price 4.357619. For the PDE solver n is both the number of price steps and of time steps.

| n | Binomial tree error | PDE solver error |
|---:|---:|---:|
| 25 | 0.039512 | 0.161417 |
| 50 | 0.019869 | 0.017012 |
| 100 | 0.009947 | 0.003586 |
| 200 | 0.004977 | 0.003102 |
| 400 | 0.002489 | 0.000234 |
| 800 | 0.001245 | 0.000195 |
| 1600 | 0.000622 | 0.000014 |

## One American put, three methods

1-year put, S0=36, K=40, r=6%, vol 20% (Longstaff & Schwartz 2001, Table 1: 4.478 with 50 exercise dates; about 4.486 with continuous exercise).

| Method | Price | Std error | Time |
|---|---:|---:|---:|
| Black-Scholes formula (European, for reference) | 3.8443 |  |  |
| Binomial tree, 2,000 steps | 4.4867 |  | 0.04 s |
| PDE solver, 400 x 400 grid | 4.4858 |  | 0.05 s |
| Longstaff-Schwartz, 100,000 paths, 50 exercise dates | 4.4723 | 0.0092 | 1.36 s |
| Longstaff-Schwartz, 100,000 paths, 250 exercise dates | 4.4795 | 0.0091 | 5.68 s |

## Down-and-out call: PDE solver against the closed form

The study's call (S0=K=100), knocked out if the stock touches the barrier.

| Barrier | Closed form | PDE solver | Difference |
|---:|---:|---:|---:|
| 80 | 4.3576 | 4.3576 | +0.0000 |
| 90 | 4.2915 | 4.2914 | -0.0000 |
| 95 | 3.5151 | 3.5151 | -0.0000 |
| 99 | 1.0314 | 1.0314 | -0.0000 |

## Heston smile: Fourier transform against Monte Carlo

Stage 5 parameters, 3 months. Fourier: 57 ms for the whole smile including the implied-vol inversion (1.5 ms for the nine prices alone). Monte Carlo with 40,000 paths: 0.5 s.

| Strike | 80 | 85 | 90 | 95 | 100 | 105 | 110 | 115 | 120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fourier | 25.14% | 23.78% | 22.40% | 21.00% | 19.58% | 18.19% | 16.93% | 15.95% | 15.33% |
| Monte Carlo | 25.02% | 23.63% | 22.27% | 20.88% | 19.52% | 18.14% | 16.91% | 15.94% | 15.27% |

## Calibration to noisy quotes

36 quotes (9 strikes x 4 maturities) generated from known parameters plus noise of 0.2% vol. Fit error 0.16% vol, 10 smile evaluations, 0.18 s.

| Parameter | Meaning | True | Fitted |
|---|---|---:|---:|
| v0 | variance today | 0.0500 | 0.0500 |
| kappa | mean-reversion speed | 2.5000 | 2.5607 |
| theta | long-run variance | 0.0350 | 0.0353 |
| xi | vol of vol | 0.4500 | 0.4450 |
| rho | correlation | -0.6500 | -0.6533 |
