"""Fast checks that run on a small synthetic sample (no saved files needed).

Run:  pytest -q
"""
import pandas as pd
import pytest

from src.config import ID_COL, PROTECTED_COLS, TARGET
from src.explain import reason_codes
from src.features import build_features
from src.generate_data import generate
from src.lead_brief import lead_payload, template_brief
from src.train import gains_table, make_lgbm


@pytest.fixture(scope="module")
def sample():
    return generate(n=4000, seed=7)


def test_generated_data_is_plausible(sample):
    assert sample[ID_COL].is_unique
    assert 0.01 < sample[TARGET].mean() < 0.25
    assert (sample["annual_income"] > 0).all()


def test_features_exclude_target_id_and_protected_attributes(sample):
    X = build_features(sample)
    forbidden = {TARGET, ID_COL, *PROTECTED_COLS}
    assert forbidden.isdisjoint(X.columns)


def test_gains_table_capture_reaches_100_percent():
    gt = gains_table(pd.Series([1, 0, 1, 0, 0, 1, 0, 0, 0, 0]).to_numpy(),
                     pd.Series(range(10, 0, -1)).to_numpy())
    assert gt["cumulative_capture"].iloc[-1] == pytest.approx(1.0)


def test_reason_codes_and_brief(sample):
    X = build_features(sample)
    model = make_lgbm().set_params(n_estimators=50).fit(X, sample[TARGET])
    rc = reason_codes(model, X.head(5))
    assert "reason_1" in rc.columns

    lead = pd.concat([sample.loc[0, [ID_COL]], rc.iloc[0]])
    lead["propensity"], lead["decile"] = 0.42, 1
    payload = lead_payload(lead)
    assert set(PROTECTED_COLS).isdisjoint(payload)
    assert "Why this customer" in template_brief(payload)
