from pathlib import Path

import pandas as pd


DATA_1PT_PATH = Path("output/regression_cleaned_data/RND_regression_all_1pt_7d_20260312.csv")
DATA_2PT_PATH = Path("output/regression_cleaned_data/RND_regression_all_2pt_7d_20260312.csv")
OUTPUT_DIR = Path("output/oos_sample_selection_7d")

DATE_COLS = ["Observation Date", "Expiration Date"]
MODEL_FILTER_COL = "Skewness"
MODEL_FILTER_THRESHOLD = -0.233984


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

    df_1pt = (
        df_1pt.merge(common_dates, on=DATE_COLS, how="inner")
        .sort_values(DATE_COLS)
        .reset_index(drop=True)
    )
    df_2pt = (
        df_2pt.merge(common_dates, on=DATE_COLS, how="inner")
        .sort_values(DATE_COLS)
        .reset_index(drop=True)
    )

    filter_dates = (
        df_1pt.loc[df_1pt[MODEL_FILTER_COL] >= MODEL_FILTER_THRESHOLD, DATE_COLS]
        .drop_duplicates()
        .copy()
    )

    model_0293_1pt = (
        df_1pt.merge(filter_dates, on=DATE_COLS, how="inner")
        .sort_values(DATE_COLS)
        .reset_index(drop=True)
    )
    model_0293_2pt = (
        df_2pt.merge(filter_dates, on=DATE_COLS, how="inner")
        .sort_values(DATE_COLS)
        .reset_index(drop=True)
    )

    output_1pt = OUTPUT_DIR / "model_0293_1pt_7d_filtered.csv"
    output_2pt = OUTPUT_DIR / "model_0293_2pt_7d_filtered.csv"

    model_0293_1pt.to_csv(output_1pt, index=False, encoding="utf-8-sig")
    model_0293_2pt.to_csv(output_2pt, index=False, encoding="utf-8-sig")

    print(f"1pt rows: {len(model_0293_1pt)}")
    print(f"2pt rows: {len(model_0293_2pt)}")
    print(f"1pt output: {output_1pt}")
    print(f"2pt output: {output_2pt}")


if __name__ == "__main__":
    main()
