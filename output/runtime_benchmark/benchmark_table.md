# Comparison of Computational Efficiency for Bitcoin Option GPD Tail Fitting

(Left: 1pt method; Right: 2pt method)

- Execution Time: 2026/03/16 17:35
- Execution Conditions: Each run generates 10 weekly return RNDs with GPD tail fitting.
- Selected Sample: 10 observation/expiration pairs from `RND_stats_1pt_7d_20260312.csv`, observation dates 2020-01-31 to 2020-05-01, expiration dates 2020-02-07 to 2020-05-08.
- Lookback Days: 7

| Item | 1pt method (sec) | 2pt method (sec) |
|---|---:|---:|
| 1st Execution Time | 17.36 | 19.11 |
| 2nd Execution Time | 17.82 | 19.27 |
| 3rd Execution Time | 17.78 | 19.24 |
| 4th Execution Time | 17.81 | 19.26 |
| 5th Execution Time | 17.77 | 19.24 |
| 6th Execution Time | 17.70 | 19.28 |
| 7th Execution Time | 17.80 | 19.33 |
| 8th Execution Time | 17.73 | 19.39 |
| 9th Execution Time | 17.95 | 19.43 |
| 10th Execution Time | 17.93 | 19.24 |
| Shortest Execution Time | 17.36 | 19.11 |
| Longest Execution Time | 17.95 | 19.43 |
| Average Execution Time | 17.77 | 19.28 |

## One-sided paired t-test

- Null hypothesis H0: mean paired runtime difference (1pt - 2pt) >= 0.
- Alternative hypothesis H1: mean paired runtime difference (1pt - 2pt) < 0.
- t statistic: -37.894064
- p value (one-sided): 0.000000
- Mean difference (1pt - 2pt): -1.512655 sec
- Significant at alpha=0.05: Yes
- Conclusion: Reject H0. The paired results show that the 1pt runtime is significantly lower than the 2pt runtime.
