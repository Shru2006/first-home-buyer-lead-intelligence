"""Generate a realistic *synthetic* dataset of everyday-banking customers who do not
yet hold a home loan, labelled with whether they applied for a first home loan in
the following 90 days.

No real customer data is used anywhere in this project. Distributions are loosely
based on public Australian statistics (ABS income bands, typical rent levels) so the
patterns feel realistic, but every row is randomly generated.

Run:  python -m src.generate_data --n 50000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import ID_COL, RANDOM_STATE, RAW_DATA_PATH, TARGET, ensure_dirs

STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
STATE_WEIGHTS = [0.32, 0.26, 0.20, 0.10, 0.07, 0.02, 0.02, 0.01]
# Rough median weekly rent by state (AUD), used to make rent amounts plausible.
STATE_WEEKLY_RENT = {"NSW": 650, "VIC": 520, "QLD": 580, "WA": 600,
                     "SA": 500, "TAS": 470, "ACT": 620, "NT": 520}
EMPLOYMENT = ["full_time", "part_time", "casual", "self_employed"]
EMPLOYMENT_WEIGHTS = [0.62, 0.16, 0.12, 0.10]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate(n: int = 50_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(33, 8, n), 19, 65).round().astype(int)
    gender = rng.choice(["F", "M", "X"], n, p=[0.495, 0.495, 0.01])
    state = rng.choice(STATES, n, p=STATE_WEIGHTS)
    employment = rng.choice(EMPLOYMENT, n, p=EMPLOYMENT_WEIGHTS)

    # Income: log-normal, lower for part-time / casual.
    emp_income_mult = pd.Series(employment).map(
        {"full_time": 1.0, "part_time": 0.6, "casual": 0.5, "self_employed": 1.05}).to_numpy()
    annual_income = np.round(rng.lognormal(np.log(85_000), 0.38, n) * emp_income_mult, -2)

    tenure_months = rng.integers(1, 240, n)
    is_renting = rng.random(n) < 0.72
    weekly_rent = np.array([STATE_WEEKLY_RENT[s] for s in state]) * rng.lognormal(0, 0.2, n)
    monthly_rent_paid = np.where(is_renting, np.round(weekly_rent * 52 / 12, -1), 0)
    rent_payments_12m = np.where(is_renting, rng.binomial(12, 0.93, n), 0)

    # Savings behaviour: the strongest real-world signal for deposit readiness.
    saver_propensity = rng.beta(2, 3, n)
    savings_balance = np.round(
        rng.lognormal(np.log(25_000), 1.0, n) * (0.3 + 1.6 * saver_propensity), -2)
    savings_growth_6m_pct = np.round(rng.normal(4 + 25 * saver_propensity - 8, 12, n), 1)
    months_saved_last_6 = rng.binomial(6, np.clip(saver_propensity + 0.15, 0, 1))
    salary_credit_regular = (rng.random(n) < np.where(employment == "casual", 0.55, 0.9)).astype(int)
    credit_card_utilisation = np.round(np.clip(rng.beta(2, 4, n) + 0.2 * (1 - saver_propensity) - 0.1, 0, 1), 2)
    bnpl_txn_90d = rng.poisson(np.where(saver_propensity < 0.3, 4, 1.2))
    joint_account_opened_12m = (rng.random(n) < 0.08).astype(int)

    # Latent "home-buying intent" that drives digital behaviour.
    intent = rng.beta(1.2, 6, n) + 0.25 * joint_account_opened_12m
    borrowing_calc_uses_90d = rng.poisson(0.3 + 9 * intent)
    home_loan_page_views_30d = rng.poisson(0.5 + 14 * intent)
    property_search_sessions_30d = rng.poisson(0.4 + 18 * intent)
    fhb_guide_views_90d = rng.poisson(0.1 + 5 * intent * is_renting)
    app_logins_30d = rng.poisson(12 + 20 * rng.random(n))
    spoke_to_lender_flag = (rng.random(n) < 0.02 + 0.25 * intent).astype(int)

    df = pd.DataFrame({
        ID_COL: [f"C{100000 + i}" for i in range(n)],
        "age": age, "gender": gender, "state": state, "employment_type": employment,
        "annual_income": annual_income, "tenure_months": tenure_months,
        "is_renting": is_renting.astype(int), "monthly_rent_paid": monthly_rent_paid,
        "rent_payments_12m": rent_payments_12m, "savings_balance": savings_balance,
        "savings_growth_6m_pct": savings_growth_6m_pct, "months_saved_last_6": months_saved_last_6,
        "salary_credit_regular": salary_credit_regular,
        "credit_card_utilisation": credit_card_utilisation, "bnpl_txn_90d": bnpl_txn_90d,
        "joint_account_opened_12m": joint_account_opened_12m,
        "borrowing_calc_uses_90d": borrowing_calc_uses_90d,
        "home_loan_page_views_30d": home_loan_page_views_30d,
        "property_search_sessions_30d": property_search_sessions_30d,
        "fhb_guide_views_90d": fhb_guide_views_90d, "app_logins_30d": app_logins_30d,
        "spoke_to_lender_flag": spoke_to_lender_flag,
    })

    # Ground-truth label: a hidden logistic process the model has to rediscover.
    deposit_ratio = savings_balance / np.maximum(annual_income, 1)
    digital_intent = (np.log1p(borrowing_calc_uses_90d) + np.log1p(home_loan_page_views_30d)
                      + np.log1p(property_search_sessions_30d))
    logit = (
        -8.3
        + 1.6 * np.tanh(deposit_ratio * 2)            # has a deposit
        + 0.9 * (deposit_ratio >= 0.20)               # threshold: roughly a 20% deposit for a typical first home
        + 0.8 * ((deposit_ratio >= 0.10) & (digital_intent > 3))  # saving AND actively researching
        - 0.9 * (credit_card_utilisation > 0.7)       # threshold: stretched credit
        + 0.035 * np.clip(savings_growth_6m_pct, -20, 40)
        + 0.18 * months_saved_last_6
        + 0.55 * salary_credit_regular
        + 0.35 * np.log1p(borrowing_calc_uses_90d)
        + 0.30 * np.log1p(home_loan_page_views_30d)
        + 0.25 * np.log1p(property_search_sessions_30d)
        + 0.30 * np.log1p(fhb_guide_views_90d)
        + 1.10 * spoke_to_lender_flag
        + 0.60 * joint_account_opened_12m
        + 0.45 * is_renting * (rent_payments_12m >= 10)
        - 1.20 * credit_card_utilisation
        - 0.08 * bnpl_txn_90d
        + 0.25 * np.log(np.maximum(annual_income, 20_000) / 85_000)
        + 0.6 * np.tanh(deposit_ratio * 2) * np.log1p(borrowing_calc_uses_90d) / 2  # interaction
        + rng.normal(0, 0.6, n)                        # unexplained noise
    )
    df[TARGET] = (rng.random(n) < _sigmoid(logit)).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000, help="number of customers")
    args = parser.parse_args()

    ensure_dirs()
    df = generate(args.n)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {RAW_DATA_PATH}")
    print(f"Application rate: {df[TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
