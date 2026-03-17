from pathlib import Path

import pandas as pd


DATA_1PT_PATH = Path("output/regression_raw_data/RND_regression_all_1pt_7d_20260312.csv")
DATA_2PT_PATH = Path("output/regression_raw_data/RND_regression_all_2pt_7d_20260312.csv")
OUTPUT_DIR = Path("output/focused_model_2_raw_7d_candidates")

DATE_COLS = ["Observation Date", "Expiration Date"]

CANDIDATES = {
    "best_oos": [
        {"column": "VIX", "operator": "<=", "threshold": 34.464000},
        {"column": "Kurtosis", "operator": ">=", "threshold": 1.270652},
        {"column": "T-3 Return", "operator": ">=", "threshold": -0.110710},
    ],
    "simplest_rule": [
        {"column": "T-3 Return", "operator": ">=", "threshold": -0.028431},
        {"column": "Median", "operator": "<=", "threshold": 66247.920000},
    ],
}


def apply_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    out = df.copy()
    for rule in rules:
        col = rule["column"]
        if rule["operator"] == "<=":
            out = out[out[col] <= rule["threshold"]]
        else:
            out = out[out[col] >= rule["threshold"]]
    return out


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "T Return" in numeric_cols:
        numeric_cols = ["T Return"] + [c for c in numeric_cols if c != "T Return"]
    desc = df[numeric_cols].describe().T
    desc.insert(0, "Variable", desc.index)
    return desc.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_1pt = pd.read_csv(DATA_1PT_PATH).copy()
    df_2pt = pd.read_csv(DATA_2PT_PATH).copy()

    for df in (df_1pt, df_2pt):
        df["Observation Date"] = pd.to_datetime(df["Observation Date"], errors="coerce")
        df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce")

    common_dates = df_1pt[DATE_COLS].merge(df_2pt[DATE_COLS], on=DATE_COLS, how="inner").drop_duplicates()
    df_1pt = df_1pt.merge(common_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)
    df_2pt = df_2pt.merge(common_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)

    for candidate_name, rules in CANDIDATES.items():
        filtered_1pt = apply_rules(df_1pt, rules).sort_values(DATE_COLS).reset_index(drop=True)
        selected_dates = filtered_1pt[DATE_COLS].drop_duplicates().copy()
        filtered_2pt = df_2pt.merge(selected_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)

        path_1pt = OUTPUT_DIR / f"{candidate_name}_1pt_raw_7d.csv"
        path_2pt = OUTPUT_DIR / f"{candidate_name}_2pt_raw_7d.csv"
        stats_1pt = OUTPUT_DIR / f"{candidate_name}_1pt_raw_7d_descriptive_stats.csv"
        stats_2pt = OUTPUT_DIR / f"{candidate_name}_2pt_raw_7d_descriptive_stats.csv"

        filtered_1pt.to_csv(path_1pt, index=False, encoding="utf-8-sig")
        filtered_2pt.to_csv(path_2pt, index=False, encoding="utf-8-sig")
        descriptive_stats(filtered_1pt).to_csv(stats_1pt, index=False, encoding="utf-8-sig")
        descriptive_stats(filtered_2pt).to_csv(stats_2pt, index=False, encoding="utf-8-sig")

        print(candidate_name)
        print(f"1pt rows: {len(filtered_1pt)} -> {path_1pt}")
        print(f"2pt rows: {len(filtered_2pt)} -> {path_2pt}")
        print(f"1pt stats: {stats_1pt}")
        print(f"2pt stats: {stats_2pt}")


if __name__ == "__main__":
    main()
