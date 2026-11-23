"""Feature engineering: turn raw banking behaviour into model-ready signals.

Business framing for each feature lives in FEATURE_DESCRIPTIONS so that the same
wording is used in reason codes, the dashboard and the LLM lender brief.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ID_COL, PROTECTED_COLS, TARGET

CATEGORICAL = ["state", "employment_type"]

FEATURE_DESCRIPTIONS = {
    "deposit_to_income": "savings relative to annual income (deposit readiness)",
    "savings_balance": "current savings balance",
    "savings_growth_6m_pct": "savings growth over the last 6 months",
    "months_saved_last_6": "number of months with net savings in the last 6",
    "salary_credit_regular": "regular salary credits",
    "annual_income": "annual income",
    "rent_to_income": "rent as a share of income",
    "consistent_renter": "consistent rent payments (10+ of last 12 months)",
    "monthly_rent_paid": "monthly rent paid",
    "credit_card_utilisation": "credit card utilisation",
    "bnpl_txn_90d": "buy-now-pay-later usage",
    "joint_account_opened_12m": "opened a joint account in the last 12 months",
    "borrowing_calc_uses_90d": "borrowing power calculator use",
    "home_loan_page_views_30d": "home loan page views",
    "property_search_sessions_30d": "property search activity in the app",
    "fhb_guide_views_90d": "first home buyer guide views",
    "digital_intent_score": "overall digital home-buying intent",
    "spoke_to_lender_flag": "recent conversation with a lender",
    "app_logins_30d": "app engagement",
    "tenure_months": "length of relationship with the bank",
    "state": "state",
    "employment_type": "employment type",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature matrix. Never includes the target, the ID or protected columns."""
    out = pd.DataFrame(index=df.index)

    income = df["annual_income"].clip(lower=1)
    out["deposit_to_income"] = (df["savings_balance"] / income).round(4)
    out["rent_to_income"] = (df["monthly_rent_paid"] * 12 / income).round(4)
    out["consistent_renter"] = ((df["is_renting"] == 1) & (df["rent_payments_12m"] >= 10)).astype(int)
    out["digital_intent_score"] = (
        np.log1p(df["borrowing_calc_uses_90d"])
        + np.log1p(df["home_loan_page_views_30d"])
        + np.log1p(df["property_search_sessions_30d"])
        + np.log1p(df["fhb_guide_views_90d"])
    ).round(4)

    passthrough = [
        "savings_balance", "savings_growth_6m_pct", "months_saved_last_6", "salary_credit_regular",
        "annual_income", "monthly_rent_paid", "credit_card_utilisation", "bnpl_txn_90d",
        "joint_account_opened_12m", "borrowing_calc_uses_90d", "home_loan_page_views_30d",
        "property_search_sessions_30d", "fhb_guide_views_90d", "spoke_to_lender_flag",
        "app_logins_30d", "tenure_months",
    ]
    for col in passthrough:
        out[col] = df[col]
    for col in CATEGORICAL:
        out[col] = df[col].astype("category")

    leaked = {TARGET, ID_COL, *PROTECTED_COLS} & set(out.columns)
    assert not leaked, f"Feature matrix must not contain {leaked}"
    return out


def describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))
