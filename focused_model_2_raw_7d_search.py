import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_1PT_PATH = Path("output/regression_raw_data/RND_regression_all_1pt_7d_20260312.csv")
DATA_2PT_PATH = Path("output/regression_raw_data/RND_regression_all_2pt_7d_20260312.csv")
OUTPUT_DIR = Path("output/focused_model_2_raw_7d_search")

DATE_COLS = ["Observation Date", "Expiration Date"]
TARGET_COL = "T Return"
FEATURE_COLS = ["Kurtosis", "Median", "Fear and Greed Index"]

WINDOW_TYPE = "expanding"
INITIAL_WINDOW = 120

FILTER_COLUMNS = [
    "Kurtosis",
    "Median",
    "Fear and Greed Index",
    "VIX",
    "Std",
    "Mean",
    "T-1 Return",
    "T-2 Return",
    "T-3 Return",
    "T-4 Return",
    "Skewness",
]
FILTER_QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MIN_RETAIN_RATIO = 0.45
MIN_ROWS_AFTER_FILTER = 150
MAX_RULE_DEPTH = 3
BEAM_WIDTH = 40

STRICT_P_THRESHOLD = 0.10


def load_and_align() -> pd.DataFrame:
    df_1pt = pd.read_csv(DATA_1PT_PATH).copy()
    df_2pt = pd.read_csv(DATA_2PT_PATH).copy()

    for df in (df_1pt, df_2pt):
        df["Observation Date"] = pd.to_datetime(df["Observation Date"], errors="coerce")
        df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce")

    common_dates = df_1pt[DATE_COLS].merge(df_2pt[DATE_COLS], on=DATE_COLS, how="inner").drop_duplicates()
    df_1pt = df_1pt.merge(common_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)
    df_2pt = df_2pt.merge(common_dates, on=DATE_COLS, how="inner").sort_values(DATE_COLS).reset_index(drop=True)

    merged = df_1pt.merge(df_2pt, on=DATE_COLS, suffixes=("_1pt", "_2pt"), how="inner")
    merged = merged.sort_values(DATE_COLS).reset_index(drop=True)
    merged["target_next_1pt"] = merged[f"{TARGET_COL}_1pt"].shift(-1)
    merged["target_next_2pt"] = merged[f"{TARGET_COL}_2pt"].shift(-1)
    return merged


def fit_regression(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    model_df = df[[y_col] + x_cols].dropna().copy()
    X = sm.add_constant(model_df[x_cols], has_constant="add")
    y = model_df[y_col]
    result = sm.OLS(y, X).fit()
    preds = result.predict(X)
    mse = float(np.mean((y - preds) ** 2))
    return result, mse


def prepare_oos_arrays(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    out = df[feature_cols + [target_col]].dropna().copy()
    return out[feature_cols].to_numpy(dtype=float), out[target_col].to_numpy(dtype=float)


def rolling_oos_r2(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X, y = prepare_oos_arrays(df, feature_cols, target_col)
    total_rows = len(y)
    if total_rows <= INITIAL_WINDOW + 1:
        return np.nan, 0

    actual_arr = np.empty(total_rows - INITIAL_WINDOW, dtype=float)
    pred_arr = np.empty(total_rows - INITIAL_WINDOW, dtype=float)
    mean_arr = np.empty(total_rows - INITIAL_WINDOW, dtype=float)

    for out_idx, t in enumerate(range(INITIAL_WINDOW, total_rows)):
        start = 0 if WINDOW_TYPE == "expanding" else t - INITIAL_WINDOW
        X_train = X[start:t]
        y_train = y[start:t]
        x_test = X[t]

        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0, ddof=0)
        x_std[x_std == 0] = 1.0

        X_train_scaled = (X_train - x_mean) / x_std
        x_test_scaled = (x_test - x_mean) / x_std

        X_design = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        beta, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
        pred = np.dot(np.concatenate(([1.0], x_test_scaled)), beta)

        actual_arr[out_idx] = y[t]
        pred_arr[out_idx] = pred
        mean_arr[out_idx] = y_train.mean()

    numerator = np.sum((actual_arr - pred_arr) ** 2)
    denominator = np.sum((actual_arr - mean_arr) ** 2)
    if denominator == 0:
        return np.nan, len(actual_arr)
    return 1 - numerator / denominator, len(actual_arr)


def make_rules(df: pd.DataFrame) -> list[dict]:
    rules = []
    for col in FILTER_COLUMNS:
        series = df[f"{col}_1pt"].dropna()
        if series.empty or series.nunique() < 5:
            continue
        for q in FILTER_QUANTILES:
            threshold = float(series.quantile(q))
            rules.append({"filter_column": col, "operator": "<=", "threshold": threshold, "rule_text": f"{col} <= {threshold:.6f}"})
            rules.append({"filter_column": col, "operator": ">=", "threshold": threshold, "rule_text": f"{col} >= {threshold:.6f}"})
    return rules


def apply_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    out = df
    for rule in rules:
        col = f"{rule['filter_column']}_1pt"
        if rule["operator"] == "<=":
            out = out[out[col] <= rule["threshold"]]
        else:
            out = out[out[col] >= rule["threshold"]]
    return out.copy()


def evaluate_rules(df: pd.DataFrame, rules: list[dict], base_n: int) -> dict | None:
    filtered = apply_rules(df, rules)
    retain_ratio = len(filtered) / base_n if base_n else 0.0
    if len(filtered) < MIN_ROWS_AFTER_FILTER or retain_ratio < MIN_RETAIN_RATIO:
        return None

    df_1pt = filtered[[f"{TARGET_COL}_1pt"] + [f"{c}_1pt" for c in FEATURE_COLS] + ["target_next_1pt"]].copy()
    df_1pt.columns = [TARGET_COL] + FEATURE_COLS + ["target_next"]
    df_2pt = filtered[[f"{TARGET_COL}_2pt"] + [f"{c}_2pt" for c in FEATURE_COLS] + ["target_next_2pt"]].copy()
    df_2pt.columns = [TARGET_COL] + FEATURE_COLS + ["target_next"]

    reg_1pt, mse_1pt = fit_regression(df_1pt, TARGET_COL, FEATURE_COLS)
    reg_2pt, mse_2pt = fit_regression(df_2pt, TARGET_COL, FEATURE_COLS)
    r2_os_1pt, n_oos_1 = rolling_oos_r2(df_1pt, FEATURE_COLS, "target_next")
    r2_os_2pt, n_oos_2 = rolling_oos_r2(df_2pt, FEATURE_COLS, "target_next")

    pvals_1pt = {var: reg_1pt.pvalues.get(var, np.nan) for var in FEATURE_COLS}
    pvals_2pt = {var: reg_2pt.pvalues.get(var, np.nan) for var in FEATURE_COLS}
    p_count_1pt = sum(pd.notna(p) and p < STRICT_P_THRESHOLD for p in pvals_1pt.values())
    p_count_2pt = sum(pd.notna(p) and p < STRICT_P_THRESHOLD for p in pvals_2pt.values())

    delta = np.nan if np.isnan(r2_os_1pt) or np.isnan(r2_os_2pt) else r2_os_1pt - r2_os_2pt
    return {
        "filter_rule": "No filter" if not rules else " AND ".join(rule["rule_text"] for rule in rules),
        "num_rules": len(rules),
        "n_obs_after_filter": len(filtered),
        "retention_ratio": retain_ratio,
        "n_oos": min(n_oos_1, n_oos_2),
        "r2_1pt": reg_1pt.rsquared,
        "r2_2pt": reg_2pt.rsquared,
        "mse_1pt": mse_1pt,
        "mse_2pt": mse_2pt,
        "r2_os_1pt": r2_os_1pt,
        "r2_os_2pt": r2_os_2pt,
        "delta_1pt_minus_2pt": delta,
        "p_count_lt_0_1_1pt": p_count_1pt,
        "p_count_lt_0_1_2pt": p_count_2pt,
        "all_p_le_0_1_1pt": p_count_1pt == len(FEATURE_COLS),
        "all_p_le_0_1_2pt": p_count_2pt == len(FEATURE_COLS),
        "coef_1pt_Kurtosis": reg_1pt.params.get("Kurtosis", np.nan),
        "p_1pt_Kurtosis": pvals_1pt["Kurtosis"],
        "coef_1pt_Median": reg_1pt.params.get("Median", np.nan),
        "p_1pt_Median": pvals_1pt["Median"],
        "coef_1pt_Fear and Greed Index": reg_1pt.params.get("Fear and Greed Index", np.nan),
        "p_1pt_Fear and Greed Index": pvals_1pt["Fear and Greed Index"],
        "coef_2pt_Kurtosis": reg_2pt.params.get("Kurtosis", np.nan),
        "p_2pt_Kurtosis": pvals_2pt["Kurtosis"],
        "coef_2pt_Median": reg_2pt.params.get("Median", np.nan),
        "p_2pt_Median": pvals_2pt["Median"],
        "coef_2pt_Fear and Greed Index": reg_2pt.params.get("Fear and Greed Index", np.nan),
        "p_2pt_Fear and Greed Index": pvals_2pt["Fear and Greed Index"],
    }


def candidate_score(row: dict) -> tuple:
    delta = row["delta_1pt_minus_2pt"]
    r2_os_1pt = row["r2_os_1pt"]
    return (
        row["p_count_lt_0_1_1pt"],
        1 if pd.notna(r2_os_1pt) and r2_os_1pt > 0 else 0,
        1 if pd.notna(delta) and delta > 0 else 0,
        -abs((delta or 0.0)) if pd.isna(delta) else delta,
        -math.inf if pd.isna(r2_os_1pt) else r2_os_1pt,
        -row["num_rules"],
    )


def unique_key(rules: list[dict]) -> tuple:
    if not rules:
        return tuple()
    return tuple(sorted((r["filter_column"], r["operator"], round(r["threshold"], 8)) for r in rules))


def beam_search(df: pd.DataFrame) -> pd.DataFrame:
    base_n = len(df)
    all_rules = make_rules(df)
    current_level = [[]]
    seen = {tuple()}
    results = []

    for depth in range(0, MAX_RULE_DEPTH + 1):
        next_level = []
        evaluated = []

        for rule_set in current_level:
            result = evaluate_rules(df, rule_set, base_n)
            if result is not None:
                results.append(result)
                evaluated.append((rule_set, result))

        if depth == MAX_RULE_DEPTH:
            break

        evaluated.sort(key=lambda x: candidate_score(x[1]), reverse=True)
        beam = evaluated[:BEAM_WIDTH] if evaluated else [( [], None )]

        for rule_set, _ in beam:
            used_cols = {r["filter_column"] for r in rule_set}
            for rule in all_rules:
                if rule["filter_column"] in used_cols:
                    continue
                new_rule_set = rule_set + [rule]
                key = unique_key(new_rule_set)
                if key in seen:
                    continue
                seen.add(key)
                next_level.append(new_rule_set)

        current_level = next_level

    return pd.DataFrame(results)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_and_align()
    summary_df = beam_search(merged)
    summary_df = summary_df.sort_values(
        ["p_count_lt_0_1_1pt", "r2_os_1pt", "delta_1pt_minus_2pt"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_path = OUTPUT_DIR / "focused_model_2_raw_7d_search_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    hits = summary_df[
        (summary_df["r2_os_1pt"] > 0)
        & (summary_df["r2_os_1pt"] > summary_df["r2_os_2pt"])
        & (summary_df["p_count_lt_0_1_1pt"] >= 2)
    ].copy()
    hits_path = OUTPUT_DIR / "focused_model_2_raw_7d_hits.csv"
    hits.to_csv(hits_path, index=False, encoding="utf-8-sig")

    print(f"Candidates evaluated: {len(summary_df)}")
    print(f"Summary output: {summary_path}")
    print(f"Hit output: {hits_path}")


if __name__ == "__main__":
    main()
