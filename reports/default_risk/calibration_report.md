# Default-probability calibration report

Calibration is estimated historical default probability among LendingClub loans with resolved `Charged Off` or `Fully Paid` labels. It is not a guarantee of future performance or a lending decision.

The sigmoid calibrator was fit only on expanding temporal folds within the training sample. The final temporal holdout was not used for fitting.

| Metric | Uncalibrated | Sigmoid calibrated |
|---|---:|---:|
| Brier score | 0.1637 | 0.1635 |
| Log loss | 0.5022 | 0.5025 |
| ROC-AUC | 0.6836 | 0.6821 |
| PR-AUC | 0.3730 | 0.3717 |

- Brier score improved on the holdout: **True**.
- Log loss improved on the holdout: **False**.
- This is one temporal holdout comparison and does not establish calibration stability across future populations or periods.

![Holdout calibration curve](calibration_curve.png)
