# Default-risk explainability report

Global permutation importance measures how much average precision changes when one input column is shuffled on a fixed, training-only sample. This measures predictive contribution in this dataset and is not causal evidence.

Individual explanations are one-feature-at-a-time sensitivity checks against training-sample median/mode values. They can create unrealistic input combinations and are not causal or lending rationales.

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| `term` | 0.089064 | 0.007143 |
| `annual_inc` | 0.023404 | 0.004865 |
| `revol_bal` | 0.015832 | 0.007212 |
| `home_ownership` | 0.014562 | 0.001364 |
| `credit_history_months` | 0.009381 | 0.003749 |
| `purpose` | 0.008255 | 0.004436 |
| `fico_range_low` | 0.007479 | 0.007777 |
| `inq_last_6mths` | 0.006964 | 0.003887 |
| `addr_state` | 0.001924 | 0.002072 |
| `verification_status` | 0.001677 | 0.002260 |

![Training-sample permutation importance](permutation_importance.png)
