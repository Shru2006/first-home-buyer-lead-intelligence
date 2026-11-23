"""Per-customer reason codes from SHAP values.

LightGBM computes exact TreeSHAP contributions natively via `pred_contrib=True`,
so no extra dependency is needed. Each contribution is in log-odds units: positive
values push a customer towards applying, negative values push away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import describe


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    contrib = model.predict(X, pred_contrib=True)
    # Last column is the model's base value (expected log-odds), not a feature.
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


MIN_WATCH_OUT_IMPACT = -0.05  # ignore negligible negative drivers
BINARY_FEATURES = {"salary_credit_regular", "consistent_renter", "joint_account_opened_12m",
                   "spoke_to_lender_flag"}


def _format_value(feature: str, value) -> str:
    if feature == "digital_intent_score":
        return "high" if value > 6 else "moderate" if value > 3 else "low"
    if feature in BINARY_FEATURES:
        return "yes" if value else "no"
    if feature == "tenure_months":
        return f"{int(value)} months"
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        if feature in {"annual_income", "savings_balance", "monthly_rent_paid"}:
            return f"${value:,.0f}"
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if feature in {"deposit_to_income", "rent_to_income", "credit_card_utilisation"}:
            return f"{value:.0%}"
        if feature == "savings_growth_6m_pct":
            return f"{value:+.1f}%"
        return f"{value:,.2f}"
    return str(value)


def reason_codes(model, X: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Return the top_k positive drivers and the single biggest negative driver per row."""
    sv = shap_values(model, X)
    rows = []
    for idx in X.index:
        contrib = sv.loc[idx].sort_values(ascending=False)
        positives = contrib[contrib > 0].head(top_k)
        negative = contrib[contrib < MIN_WATCH_OUT_IMPACT].tail(1)
        row = {}
        for i, (feat, val) in enumerate(positives.items(), start=1):
            row[f"reason_{i}"] = f"{describe(feat).capitalize()} ({_format_value(feat, X.at[idx, feat])})"
            row[f"reason_{i}_impact"] = round(float(val), 3)
        if len(negative):
            feat, val = next(iter(negative.items()))
            row["watch_out"] = f"{describe(feat).capitalize()} ({_format_value(feat, X.at[idx, feat])})"
            row["watch_out_impact"] = round(float(val), 3)
        rows.append(row)
    return pd.DataFrame(rows, index=X.index)


def global_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature: how much each feature moves scores on average."""
    return shap_values(model, X).abs().mean().sort_values(ascending=False)
