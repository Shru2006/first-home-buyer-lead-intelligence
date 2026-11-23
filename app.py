"""Streamlit dashboard: browse first home buyer leads, see why each was flagged,
and generate a lender brief.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, ID_COL, METRICS_PATH, SCORED_LEADS_PATH
from src.lead_brief import generate_brief, has_llm_credentials

st.set_page_config(page_title="First Home Buyer Lead Intelligence", page_icon="🏠", layout="wide")


@st.cache_data
def load():
    leads = pd.read_csv(SCORED_LEADS_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    return leads, metrics


if not SCORED_LEADS_PATH.exists():
    st.error("No scored leads yet. Run `python -m src.generate_data` and `python -m src.train` first.")
    st.stop()

leads, metrics = load()
lgbm = next(m for m in metrics["models"] if m["model"] == "LightGBM")

st.title("🏠 First Home Buyer Lead Intelligence")
st.caption("Synthetic data only. A portfolio project, not affiliated with any bank.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Leads in top 10%", f"{len(leads):,}")
c2.metric("ROC-AUC (test)", f"{lgbm['roc_auc']:.3f}")
c3.metric("Top-decile lift", f"{lgbm['top_decile_lift']:.1f}x")
c4.metric("Applicants captured in top 10%", f"{lgbm['top_decile_capture']:.0%}")

tab_leads, tab_model = st.tabs(["Lead list", "Model performance"])

with tab_leads:
    left, right = st.columns([3, 2])
    with left:
        states = st.multiselect("State", sorted(leads["state"].unique()), default=None)
        min_score = st.slider("Minimum propensity", 0.0, float(leads["propensity"].max()), 0.0, 0.01)
        view = leads[leads["propensity"] >= min_score]
        if states:
            view = view[view["state"].isin(states)]
        cols = [ID_COL, "propensity", "state", "employment_type", "reason_1", "reason_2", "reason_3"]
        st.dataframe(view[cols], hide_index=True, width="stretch", height=480)

    with right:
        selected = st.selectbox("Customer", view[ID_COL].head(200))
        if selected:
            lead = view[view[ID_COL] == selected].iloc[0]
            st.subheader(f"{selected} · score {lead['propensity']:.2f}")
            impacts = pd.Series({lead[f"reason_{i}"]: lead[f"reason_{i}_impact"]
                                 for i in (1, 2, 3) if isinstance(lead.get(f"reason_{i}"), str)})
            if isinstance(lead.get("watch_out"), str):
                impacts[lead["watch_out"]] = lead["watch_out_impact"]
            st.bar_chart(impacts.sort_values(), horizontal=True)

            label = "Generate brief with Claude" if has_llm_credentials() else "Generate brief (template mode)"
            if st.button(label):
                with st.spinner("Writing brief..."):
                    brief, source = generate_brief(lead)
                st.markdown(brief)
                st.caption(f"Source: {source}")

with tab_model:
    st.dataframe(pd.DataFrame(metrics["models"]).set_index("model"), width="stretch")
    for name, caption in [("roc_pr_curves.png", "ROC and precision-recall"),
                          ("cumulative_gains.png", "Cumulative gains"),
                          ("feature_importance_shap.png", "Global drivers (SHAP)"),
                          ("calibration.png", "Calibration")]:
        path = FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=caption)
