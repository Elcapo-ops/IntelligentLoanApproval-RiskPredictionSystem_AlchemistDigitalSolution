"""Independent default-risk prediction package.

This module is intentionally isolated from the historical Loan_Status approval
classifier. It is a future-ready scaffold for genuine loan default risk work
using a repayment dataset such as LendingClub or Give Me Some Credit.
"""

from default_risk.dataset_status import (
    DefaultRiskDatasetStatus,
    evaluate_default_risk_dataset,
    get_default_risk_target_mapping,
)

__all__ = [
    "DefaultRiskDatasetStatus",
    "evaluate_default_risk_dataset",
    "get_default_risk_target_mapping",
]
