"""Central paths and settings so every script reads from one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROMPTS_DIR = ROOT / "prompts"

RAW_DATA_PATH = DATA_DIR / "customers_synthetic.csv"
SCORED_LEADS_PATH = DATA_DIR / "scored_leads.csv"
MODEL_PATH = MODELS_DIR / "fhb_propensity_lgbm.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "applied_home_loan_90d"
ID_COL = "customer_id"
RANDOM_STATE = 42

# Columns kept for fairness monitoring only. They are never used as model inputs.
PROTECTED_COLS = ["age", "gender"]

LLM_MODEL = "claude-opus-5-5"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
