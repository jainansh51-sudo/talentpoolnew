"""Build data/cohort_tags.csv from talent_pool_cleaned.csv (run after cleaning)."""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
CLEANED = BASE / "data" / "talent_pool_cleaned.csv"
OUT = BASE / "data" / "cohort_tags.csv"

SCOPES = ["National", "International", "Regional"]
CHAMPS = [
    "National championships",
    "International meet",
    "Selection trials",
    "World qualification window",
    "Domestic circuit",
]


def main() -> None:
    df = pd.read_csv(CLEANED, usecols=["Athlete ID"])
    ids = df["Athlete ID"].drop_duplicates().sort_values()
    rows = []
    for aid in ids:
        h = hash(str(aid))
        rows.append(
            {
                "Athlete ID": aid,
                "competition_scope": SCOPES[h % 3],
                "championship": CHAMPS[h % len(CHAMPS)],
                "season_note": "Auto-tagged for dashboard (replace with real labels when available)",
            }
        )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
