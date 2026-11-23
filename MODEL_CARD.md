# Model card: first home buyer propensity model

| | |
|---|---|
| **Purpose** | Rank existing everyday-banking customers by how likely they are to apply for their first home loan in the next 90 days, so lending specialists can prioritise conversations. |
| **Not for** | Credit decisions, pricing, or anything that changes what a customer is offered. It is a marketing propensity model. Every real conversation still goes through standard responsible lending checks. |
| **Model** | LightGBM classifier (early stopping), compared against a logistic regression baseline. |
| **Inputs** | Behavioural and financial signals: savings and deposit readiness, rent, digital research activity, credit use, tenure. **Age and gender are never inputs** (a unit test enforces this). |
| **Training data** | 50,000 synthetic customers from `src/generate_data.py`. Random 75/25 split. |
| **Label** | Applied for a home loan within 90 days (base rate 5.7%). |

## Performance (12,500 held-out customers)

| Model | ROC-AUC | PR-AUC | Top-10% lift | Applicants captured in top 10% |
|---|---|---|---|---|
| Logistic regression | 0.821 | 0.263 | 4.3x | 43% |
| LightGBM | 0.846 | 0.310 | 4.9x | 49% |

## Explanations and the LLM

- Each lead gets SHAP reason codes: its top 3 drivers and 1 watch-out.
- Claude rewrites only those reason codes into a short lender brief. It never sees protected attributes and cannot change who is selected.
- Reason codes are validated before use: anything that isn't `<known feature label> (<short value>)` is withheld, so tampered text cannot reach the brief or the LLM. The brief writer is tested by [llm-brief-evals](https://github.com/Shru2006/llm-brief-evals).

## Fairness

Selection rate into the top 10% is monitored by age band and gender, which are kept for reporting only. The lowest selection-rate ratio is 0.80 for age bands and 0.90 for gender, among groups with 500+ customers. Differences come through proxies such as savings balances, and would be reviewed with risk and compliance before any real use.

## Known limitations

- Synthetic data: the results show the method, not real-world performance.
- Random rather than time-based split. Monitoring and time-based validation are shown in [model-monitoring-pipeline](https://github.com/Shru2006/model-monitoring-pipeline).
- Propensity is not persuadability: some top-ranked customers would apply anyway. See [outreach-uplift-modelling](https://github.com/Shru2006/outreach-uplift-modelling).
