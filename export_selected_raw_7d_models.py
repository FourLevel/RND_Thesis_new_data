from pathlib import Path

import pandas as pd


DATA_1PT_PATH = Path("output/regression_raw_data/RND_regression_all_1pt_7d_20260312.csv")
DATA_2PT_PATH = Path("output/regression_raw_data/RND_regression_all_2pt_7d_20260312.csv")
OUTPUT_DIR = Path("output/selected_raw_7d_models")

DATE_COLS = ["Observation Date", "Expiration Date"]

CANDIDATES = {
    "model_1_selected": {
        "rules": [
            {"column": "Std", "operator": ">=", "threshold": 905.962517},
            {"column": "VIX", "operator": "<=", "threshold": 29.206000},
        ],
        "features": ["Skewness", "Median", "Fear and Greed Index"],
    },
    "model_2_best_oos": {
        "rules": [
            {"column": "VIX", "operator": "<=", "threshold": 34.464000},
            {"column": "Kurtosis", "operator": ">=", "threshold": 1.270652},
            {"column": "T-3 Return", "operator": ">=", "threshold": -0.110710},
        ],
        "features": ["Kurtosis", "Median", "Fear and Greed Index"],
    },
}


def apply_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    out = df.copy()
    for rule in rules:
        if rule["operator"] == "<=":
            out = out[out[rule["column"]] <= rule["threshold"]]
        else:
            out = out[out[rule["column"]] >= rule["threshold"]]
    return out


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "T Return" in numeric_cols:
        numeric_cols = ["T Return"] + [c for c in numeric_cols if c != "T Return"]
    desc = df[numeric_cols].describe().T
    desc.insert(0, "Variable", desc.index)
    return desc.reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "T Return" in numeric_cols:
        numeric_cols = ["T Return"] + [c for c in numeric_cols if c != "T Return"]
    return df[numeric_cols].corr()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_1pt = pd.read_csv(DATA_1PT_PATH).copy()
    df_2pt = pd.read_csv(DATA_2PT_PATH).copy()

    for df in (df_1pt, df_2pt):
        df["Observation Date"] = pd.to_datetime(df["Observation Date"], errors="coerce")
        df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce")

    common_dates = (
        df_1pt[DATE_COLS]
        .merge(df_2pt[DATE_COLS], on=DATE_COLS, how="inner")
        .drop_duplicates()
    )
    df_1pt = df_1pt.merge(common_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)
    df_2pt = df_2pt.merge(common_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)

    metadata_rows = []

    for candidate_name, config in CANDIDATES.items():
        candidate_dir = OUTPUT_DIR / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)

        filtered_1pt = apply_rules(df_1pt, config["rules"]).sort_values(DATE_COLS).reset_index(drop=True)
        selected_dates = filtered_1pt[DATE_COLS].drop_duplicates().copy()
        filtered_2pt = df_2pt.merge(selected_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)

        filtered_1pt.to_csv(candidate_dir / f"{candidate_name}_1pt_raw_7d.csv", index=False, encoding="utf-8-sig")
        filtered_2pt.to_csv(candidate_dir / f"{candidate_name}_2pt_raw_7d.csv", index=False, encoding="utf-8-sig")
        descriptive_stats(filtered_1pt).to_csv(candidate_dir / f"{candidate_name}_1pt_descriptive_stats.csv", index=False, encoding="utf-8-sig")
        descriptive_stats(filtered_2pt).to_csv(candidate_dir / f"{candidate_name}_2pt_descriptive_stats.csv", index=False, encoding="utf-8-sig")
        correlation_matrix(filtered_1pt).to_csv(candidate_dir / f"{candidate_name}_1pt_correlation_matrix.csv", encoding="utf-8-sig")
        correlation_matrix(filtered_2pt).to_csv(candidate_dir / f"{candidate_name}_2pt_correlation_matrix.csv", encoding="utf-8-sig")

        metadata_rows.append(
            {
                "candidate_name": candidate_name,
                "features": " + ".join(config["features"]),
                "filter_rule": " AND ".join(f"{r['column']} {r['operator']} {r['threshold']:.6f}" for r in config["rules"]),
                "n_obs": len(filtered_1pt),
            }
        )

    pd.DataFrame(metadata_rows).to_csv(OUTPUT_DIR / "selected_models_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
