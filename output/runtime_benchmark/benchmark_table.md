# Comparison of Computational Efficiency for Bitcoin Option GPD Tail Fitting

(Left: 1pt method; Right: 2pt method)

- Execution Time: 2026/03/17 00:28
- Execution Conditions: Each run generates 100 weekly return RNDs with GPD tail fitting.
- Selected Sample: 100 observation/expiration pairs from `RND_stats_1pt_7d_20260312.csv`, observation dates 2020-01-31 to 2022-08-12, expiration dates 2020-02-07 to 2022-08-19.
- Lookback Days: 7

| Item | 1pt method (sec) | 2pt method (sec) |
|---|---:|---:|
| 1st Execution Time | 427.31 | 447.25 |
| 2nd Execution Time | 421.94 | 436.44 |
| 3rd Execution Time | 430.45 | 453.25 |
| 4th Execution Time | 432.20 | 439.96 |
| 5th Execution Time | 423.83 | 440.25 |
| 6th Execution Time | 423.76 | 439.51 |
| 7th Execution Time | 423.23 | 440.28 |
| 8th Execution Time | 436.85 | 458.60 |
| 9th Execution Time | 445.85 | 458.22 |
| 10th Execution Time | 429.24 | 441.02 |
| Shortest Execution Time | 421.94 | 436.44 |
| Longest Execution Time | 445.85 | 458.60 |
| Average Execution Time | 429.47 | 445.48 |

## One-sided paired t-test

- Null hypothesis H0: mean paired runtime difference (1pt - 2pt) >= 0.
- Alternative hypothesis H1: mean paired runtime difference (1pt - 2pt) < 0.
- t statistic: -10.818352
- p value (one-sided): 0.000001
- Mean difference (1pt - 2pt): -16.012285 sec
- Significant at alpha=0.05: Yes
- Conclusion: Reject H0. The paired results show that the 1pt runtime is significantly lower than the 2pt runtime.
