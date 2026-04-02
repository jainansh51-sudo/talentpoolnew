"""
Clean talent_pool_results CSV: normalize types, fix labels, flag bad performances.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path(r"c:\Users\rahul\Downloads\talent_pool_results (1).csv")

PERCENT_COLS = [
    c
    for c in [
        "LR_Mean_Qualify",
        "LR_Mean_Medal",
        "LR_Median_Qualify",
        "LR_Median_Medal",
        "LR_Best_Qualify",
        "LR_Best_Medal",
        "Bayes_Mean_Qualify",
        "Bayes_Mean_Medal",
        "Bayes_Median_Qualify",
        "Bayes_Median_Medal",
        "Bayes_Best_Qualify",
        "Bayes_Best_Medal",
        "RF_Mean_Qualify",
        "RF_Mean_Medal",
        "RF_Median_Qualify",
        "RF_Median_Medal",
        "RF_Best_Qualify",
        "RF_Best_Medal",
        "GB_Mean_Qualify",
        "GB_Mean_Medal",
        "GB_Median_Qualify",
        "GB_Median_Medal",
        "GB_Best_Qualify",
        "GB_Best_Medal",
        "MC_Mean_Qualify",
        "MC_Mean_Medal",
        "MC_Median_Qualify",
        "MC_Median_Medal",
        "MC_Best_Qualify",
        "MC_Best_Medal",
    ]
]


def _parse_percent(series: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip().rstrip("%")
        try:
            return float(s)
        except ValueError:
            return pd.NA

    return series.map(one).astype("Float64")


def _fix_category(event_name: str, current: str) -> str:
    if pd.isna(event_name):
        return current
    ev = str(event_name)
    if "Pole Vault" in ev:
        return "Jump"
    if re.search(r"Race Walk|Marathon|Half-Marathon", ev, re.I):
        return "Road"
    return current if pd.notna(current) else ""


def _mean_issues(event_name: str, category: str, gender: str, mean: float) -> list[str]:
    issues: list[str] = []
    if pd.isna(mean):
        issues.append("missing_mean")
        return issues
    if category == "Combined" or "Heptathlon" in str(event_name) or "Decathlon" in str(event_name):
        return issues
    ev = str(event_name)
    m = float(mean)
    if "Long Jump" in ev and "Triple" not in ev:
        if m < 2.0 or m > 9.5:
            issues.append("long_jump_range")
    elif "Triple Jump" in ev:
        if m < 8.0 or m > 19.5:
            issues.append("triple_jump_range")
    elif "High Jump" in ev:
        if m < 1.0 or m > 2.6:
            issues.append("high_jump_range")
    elif "1500m" in ev:
        if m < 190 or m > 420:
            issues.append("1500m_range")
    elif "Men's 100m" == ev or (re.match(r"Women's 100m$", ev)):
        if m < 9.0 or m > 14.5:
            issues.append("100m_range")
    elif "110m Hurdles" in ev:
        if m < 12.0 or m > 16.5:
            issues.append("110h_range")
    elif "100m Hurdles" in ev:
        if m < 11.0 or m > 16.0:
            issues.append("100mh_range")
    return issues


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Athlete ID", "Athlete Name", "Event Name", "Category", "Gender"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
            out[c] = out[c].replace({"nan": pd.NA})
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out.loc[(out["Age"] < 10) | (out["Age"] > 80), "Age"] = pd.NA

    out["Category"] = out.apply(
        lambda r: _fix_category(r.get("Event Name"), r.get("Category")), axis=1
    )

    for c in PERCENT_COLS:
        if c in out.columns:
            out[c] = _parse_percent(out[c])

    issues_col = out.apply(
        lambda r: ";".join(
            _mean_issues(
                r.get("Event Name"),
                r.get("Category"),
                r.get("Gender"),
                pd.to_numeric(r.get("Mean"), errors="coerce"),
            )
        ),
        axis=1,
    )

    out["cleaning_issues"] = issues_col
    out["performance_flag_ok"] = out["cleaning_issues"].eq("")

    return out


def load_and_clean(source: Path | str | None = None) -> pd.DataFrame:
    path = Path(source) if source else DEFAULT_SOURCE
    df = pd.read_csv(path)
    return clean_dataframe(df)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out_path = args.out or Path(__file__).resolve().parent / "data" / "talent_pool_cleaned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = load_and_clean(args.source)
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned)} rows to {out_path}")
    bad = (~cleaned["performance_flag_ok"]).sum()
    print(f"Rows with performance range flags: {bad}")


if __name__ == "__main__":
    main()
