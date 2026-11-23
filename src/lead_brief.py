"""Turn model output into a lender-ready brief using Claude (with an offline fallback).

The model decides *who* to call; the LLM only explains *why* in plain English.
The LLM never sees protected attributes and is instructed to use only the facts
it is given, so it cannot change who gets selected.

Run:  python -m src.lead_brief --top 5
Needs ANTHROPIC_API_KEY for Claude; without it a rule-based template is used.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.config import ID_COL, LLM_MODEL, PROMPTS_DIR, REPORTS_DIR, SCORED_LEADS_PATH

SYSTEM_PROMPT = (PROMPTS_DIR / "lead_brief_system.md").read_text(encoding="utf-8")


def lead_payload(lead: pd.Series) -> dict:
    """Only the fields the LLM is allowed to see."""
    reasons = [lead[f"reason_{i}"] for i in (1, 2, 3)
               if f"reason_{i}" in lead and isinstance(lead[f"reason_{i}"], str)]
    return {
        "customer_id": lead[ID_COL],
        "propensity_score": round(float(lead["propensity"]), 3),
        "score_decile": int(lead["decile"]),
        "reason_codes": reasons,
        "watch_out": lead["watch_out"] if isinstance(lead.get("watch_out"), str) else None,
    }


def template_brief(payload: dict) -> str:
    lines = ["**Why this customer**"]
    lines += [f"- {r}" for r in payload["reason_codes"]]
    if payload["watch_out"]:
        lines += ["**Watch out**", f"- {payload['watch_out']}"]
    lines += ["**Suggested conversation opener**",
              "- \"We noticed you've been exploring home buying. Would a quick chat about "
              "your borrowing power and first home buyer options be useful?\""]
    return "\n".join(lines)


def claude_brief(payload: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    if response.stop_reason == "refusal":
        return template_brief(payload)
    return "".join(b.text for b in response.content if b.type == "text").strip()


def has_llm_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def generate_brief(lead: pd.Series, use_llm: bool | None = None) -> tuple[str, str]:
    """Return (brief_markdown, source) where source is 'claude' or 'template'."""
    payload = lead_payload(lead)
    if use_llm is None:
        use_llm = has_llm_credentials()
    if use_llm:
        try:
            return claude_brief(payload), "claude"
        except Exception as exc:  # network or auth issues should not break the dashboard
            print(f"Claude unavailable ({type(exc).__name__}); using template.")
    return template_brief(payload), "template"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=5, help="number of top leads to brief")
    args = parser.parse_args()

    leads = pd.read_csv(SCORED_LEADS_PATH).head(args.top)
    out = ["# Lender briefs: top first home buyer leads\n"]
    for _, lead in leads.iterrows():
        brief, source = generate_brief(lead)
        out.append(f"## {lead[ID_COL]} · score {lead['propensity']:.2f} · source: {source}\n\n{brief}\n")
    path = REPORTS_DIR / "sample_lead_briefs.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
