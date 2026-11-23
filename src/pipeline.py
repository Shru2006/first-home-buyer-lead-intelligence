"""Run the whole project end to end with one command.

Run:  python -m src.pipeline
"""
import sys

from src import generate_data, lead_brief, train

STEPS = [
    ("1/3 Generate synthetic customers", generate_data.main),
    ("2/3 Train, evaluate, explain and score", train.main),
    ("3/3 Write lender briefs for the top leads", lead_brief.main),
]


def main() -> None:
    sys.argv = sys.argv[:1]  # each step uses its own defaults
    for title, step in STEPS:
        print(f"\n=== {title} ===")
        step()


if __name__ == "__main__":
    main()
