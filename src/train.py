"""Train, evaluate and explain the First Home Buyer propensity model.

Steps
1. Load the synthetic customer data and build features.
2. Train a logistic regression baseline and a LightGBM model on the same split.
3. Evaluate with metrics a lending team cares about: ROC-AUC, PR-AUC, and the
   lift / capture rate if lenders only call the top 10% of scored customers.
4. Check fairness: the model never sees age or gender, but we still monitor
   whether scores differ across those groups.
5. Score every customer, attach reason codes, and save the lead list.

Run:  python -m src.train
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (FIGURES_DIR, ID_COL, METRICS_PATH, MODEL_PATH, RANDOM_STATE,
                        RAW_DATA_PATH, REPORTS_DIR, SCORED_LEADS_PATH, TARGET, ensure_dirs)
from src.explain import global_importance, reason_codes
from src.features import CATEGORICAL, build_features, describe

TOP_SHARE = 0.10  # lenders' realistic capacity: call the top 10% of customers
MIN_FAIRNESS_GROUP = 500  # smaller groups give noisy selection rates; reported but flagged


def gains_table(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "score": scores}).sort_values("score", ascending=False)
    df["decile"] = np.ceil(np.arange(1, len(df) + 1) / len(df) * bins).astype(int)
    g = df.groupby("decile").agg(customers=("y", "size"), applicants=("y", "sum"),
                                 min_score=("score", "min"), max_score=("score", "max"))
    g["application_rate"] = g["applicants"] / g["customers"]
    g["lift"] = g["application_rate"] / df["y"].mean()
    g["cumulative_capture"] = g["applicants"].cumsum() / df["y"].sum()
    return g.round(4)


def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    gt = gains_table(y, p)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "top_decile_lift": float(gt.loc[1, "lift"]),
        "top_decile_capture": float(gt.loc[1, "cumulative_capture"]),
        "base_rate": round(float(y.mean()), 4),
    }


def make_baseline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder=StandardScaler())
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight=None))])


def make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbose=-1,
    )


def plot_curves(y, preds: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(y.mean(), ls="--", color="grey", lw=1, label=f"Base rate {y.mean():.1%}")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_gains(gt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(0, 11) * 10
    ax.plot(x, [0, *(gt["cumulative_capture"] * 100)], marker="o", label="LightGBM model")
    ax.plot(x, x, "--", color="grey", label="Random targeting")
    ax.set(title="Cumulative gains: share of applicants reached",
           xlabel="% of customers contacted (highest score first)", ylabel="% of applicants reached")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cumulative_gains.png", dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.Series) -> None:
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([describe(f) for f in top.index], top.values, color="#2b6cb0")
    ax.set(title="What drives the score (mean |SHAP|)", xlabel="Average impact on log-odds")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_shap.png", dpi=150)
    plt.close(fig)


def plot_calibration(y, p) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", label="LightGBM")
    lim = max(mean_pred.max(), frac_pos.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", label="Perfect calibration")
    ax.set(title="Calibration", xlabel="Predicted probability", ylabel="Observed application rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def fairness_report(raw: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Selection rate into the top 10% by group, and the ratio to the best-treated group."""
    df = raw[["age", "gender", TARGET]].copy()
    df["score"] = scores
    df["selected"] = df["score"] >= np.quantile(scores, 1 - TOP_SHARE)
    df["age_band"] = pd.cut(df["age"], [0, 24, 34, 44, 54, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"])
    frames = []
    for col in ["age_band", "gender"]:
        g = df.groupby(col, observed=True).agg(customers=("score", "size"), mean_score=("score", "mean"),
                                               selection_rate=("selected", "mean"),
                                               actual_application_rate=(TARGET, "mean"))
        g["small_sample"] = g["customers"] < MIN_FAIRNESS_GROUP
        reliable_max = g.loc[~g["small_sample"], "selection_rate"].max()
        g["selection_ratio_vs_max"] = g["selection_rate"] / reliable_max
        g.insert(0, "attribute", col)
        frames.append(g.reset_index().rename(columns={col: "group"}))
    return pd.concat(frames, ignore_index=True).round(4)


def main() -> None:
    ensure_dirs()
    raw = pd.read_csv(RAW_DATA_PATH)
    X = build_features(raw)
    y = raw[TARGET].to_numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, raw.index, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    baseline = make_baseline().fit(X_train, y_train)
    p_base = baseline.predict_proba(X_test)[:, 1]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
    model = make_lgbm()
    model.fit(X_tr, y_tr, eval_X=(X_val,), eval_y=(y_val,), eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgbm = model.predict_proba(X_test)[:, 1]

    results = [evaluate("Logistic regression (baseline)", y_test, p_base),
               evaluate("LightGBM", y_test, p_lgbm)]
    gt = gains_table(y_test, p_lgbm)
    gt.to_csv(REPORTS_DIR / "gains_table_test.csv")

    plot_curves(y_test, {"Logistic regression": p_base, "LightGBM": p_lgbm})
    plot_gains(gt)
    plot_calibration(y_test, p_lgbm)
    imp = global_importance(model, X_test)
    plot_importance(imp)

    fair = fairness_report(raw.loc[idx_test], p_lgbm)
    fair.to_csv(REPORTS_DIR / "fairness_test.csv", index=False)

    # Score the full book and attach reason codes (in production this would be
    # the current month's customers; here it is the whole synthetic population).
    all_scores = model.predict_proba(X)[:, 1]
    leads = raw[[ID_COL, "state", "employment_type", "annual_income", "savings_balance",
                 "monthly_rent_paid", "borrowing_calc_uses_90d", "home_loan_page_views_30d"]].copy()
    leads["propensity"] = all_scores.round(4)
    leads["decile"] = pd.qcut(-all_scores, 10, labels=False) + 1
    top_idx = leads.nlargest(int(len(leads) * TOP_SHARE), "propensity").index
    leads = leads.loc[top_idx].join(reason_codes(model, X.loc[top_idx]))
    leads.sort_values("propensity", ascending=False).to_csv(SCORED_LEADS_PATH, index=False)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "dataset": {"customers": int(len(raw)), "test_customers": int(len(y_test)),
                    "application_rate": round(float(y.mean()), 4)},
        "models": results,
        "lightgbm_best_iteration": int(model.best_iteration_ or model.n_estimators),
        "top_features": [describe(f) for f in imp.head(5).index],
        "min_selection_ratio": {
            attr: float(fair.loc[(fair["attribute"] == attr) & ~fair["small_sample"],
                                 "selection_ratio_vs_max"].min())
            for attr in fair["attribute"].unique()},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(pd.DataFrame(results).set_index("model").to_string())
    print(f"\nTop-decile gains (test set):\n{gt[['applicants', 'application_rate', 'lift', 'cumulative_capture']].head(3)}")
    print(f"\nSaved model -> {MODEL_PATH}\nSaved {len(top_idx):,} leads -> {SCORED_LEADS_PATH}")


if __name__ == "__main__":
    main()
