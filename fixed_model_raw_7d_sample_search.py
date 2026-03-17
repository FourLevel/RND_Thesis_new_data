import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_1PT_PATH = Path("output/regression_raw_data/RND_regression_all_1pt_7d_20260312.csv")
DATA_2PT_PATH = Path("output/regression_raw_data/RND_regression_all_2pt_7d_20260312.csv")
OUTPUT_DIR = Path("output/fixed_model_raw_7d_search")

DATE_COLS = ["Observation Date", "Expiration Date"]
TARGET_COL = "T Return"

MODELS = {
    "model_1": ["Skewness", "Median", "Fear and Greed Index"],
    "model_2": ["Kurtosis", "Median", "Fear and Greed Index"],
}

WINDOW_TYPE = "expanding"  # or "fixed"
INITIAL_WINDOW = 120

FILTER_COLUMNS = [
    "Skewness",
    "Kurtosis",
    "Mean",
    "Std",
    "Median",
    "Fear and Greed Index",
    "VIX",
    "T-1 Return",
    "T-2 Return",
    "T-3 Return",
    "T-4 Return",
]
FILTER_QUANTILES = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]
MAX_FILTER_RULES = 2
MIN_RETAIN_RATIO = 0.60
MIN_ROWS_AFTER_FILTER = 180

STRICT_P_THRESHOLD = 0.10
NEAR_P_THRESHOLD = 0.15
TOP_N_PER_MODEL = 30


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


def rolling_oos_r2(df: pd.DataFrame, feature_cols: list[str], target_col: str, initial_window: int, window_type: str):
    X, y = prepare_oos_arrays(df, feature_cols, target_col)
    total_rows = len(y)
    if total_rows <= initial_window + 1:
        return np.nan, 0

    actual_arr = np.empty(total_rows - initial_window, dtype=float)
    pred_arr = np.empty(total_rows - initial_window, dtype=float)
    mean_arr = np.empty(total_rows - initial_window, dtype=float)

    for out_idx, t in enumerate(range(initial_window, total_rows)):
        if window_type == "expanding":
            start = 0
        elif window_type == "fixed":
            start = t - initial_window
        else:
            raise ValueError("window_type must be 'expanding' or 'fixed'")

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


def make_single_rules(df: pd.DataFrame) -> list[dict]:
    rules = []
    for col in FILTER_COLUMNS:
        if f"{col}_1pt" not in df.columns:
            continue
        series = df[f"{col}_1pt"].dropna()
        if series.empty or series.nunique() < 5:
            continue

        for q in FILTER_QUANTILES:
            threshold = float(series.quantile(q))
            rules.append(
                {
                    "filter_column": col,
                    "operator": "<=",
                    "threshold": threshold,
                    "rule_text": f"{col} <= {threshold:.6f}",
                }
            )
            rules.append(
                {
                    "filter_column": col,
                    "operator": ">=",
                    "threshold": threshold,
                    "rule_text": f"{col} >= {threshold:.6f}",
                }
            )
    return rules


def apply_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    col = f"{rule['filter_column']}_1pt"
    if rule["operator"] == "<=":
        return df[df[col] <= rule["threshold"]].copy()
    return df[df[col] >= rule["threshold"]].copy()


def generate_rule_sets(single_rules: list[dict]):
    yield []
    for rule in single_rules:
        yield [rule]
    if MAX_FILTER_RULES >= 2:
        for rule_a, rule_b in itertools.combinations(single_rules, 2):
            if rule_a["filter_column"] == rule_b["filter_column"]:
                continue
            yield [rule_a, rule_b]


def evaluate_model_on_sample(model_name: str, feature_cols: list[str], df: pd.DataFrame, rule_set: list[dict], base_n: int) -> tuple[dict, list[dict]]:
    filtered = df.copy()
    for rule in rule_set:
        filtered = apply_rule(filtered, rule)

    retain_ratio = len(filtered) / base_n if base_n else 0.0
    if len(filtered) < MIN_ROWS_AFTER_FILTER or retain_ratio < MIN_RETAIN_RATIO:
        return None, []

    df_1pt = filtered[[f"{TARGET_COL}_1pt"] + [f"{c}_1pt" for c in feature_cols] + ["target_next_1pt"]].copy()
    df_1pt.columns = [TARGET_COL] + feature_cols + ["target_next"]
    df_2pt = filtered[[f"{TARGET_COL}_2pt"] + [f"{c}_2pt" for c in feature_cols] + ["target_next_2pt"]].copy()
    df_2pt.columns = [TARGET_COL] + feature_cols + ["target_next"]

    reg_1pt, mse_1pt = fit_regression(df_1pt, TARGET_COL, feature_cols)
    reg_2pt, mse_2pt = fit_regression(df_2pt, TARGET_COL, feature_cols)
    r2_os_1pt, n_oos_1 = rolling_oos_r2(df_1pt, feature_cols, "target_next", INITIAL_WINDOW, WINDOW_TYPE)
    r2_os_2pt, n_oos_2 = rolling_oos_r2(df_2pt, feature_cols, "target_next", INITIAL_WINDOW, WINDOW_TYPE)
    n_oos = min(n_oos_1, n_oos_2)

    pvals_1pt = {var: reg_1pt.pvalues.get(var, np.nan) for var in feature_cols}
    pvals_2pt = {var: reg_2pt.pvalues.get(var, np.nan) for var in feature_cols}

    summary_row = {
        "model_name": model_name,
        "variables": " + ".join(feature_cols),
        "filter_rule": "No filter" if not rule_set else " AND ".join(rule["rule_text"] for rule in rule_set),
        "num_rules": len(rule_set),
        "n_obs_after_filter": len(filtered),
        "retention_ratio": retain_ratio,
        "n_oos": n_oos,
        "r2_1pt": reg_1pt.rsquared,
        "r2_2pt": reg_2pt.rsquared,
        "mse_1pt": mse_1pt,
        "mse_2pt": mse_2pt,
        "r2_os_1pt": r2_os_1pt,
        "r2_os_2pt": r2_os_2pt,
        "delta_1pt_minus_2pt": np.nan if np.isnan(r2_os_1pt) or np.isnan(r2_os_2pt) else r2_os_1pt - r2_os_2pt,
        "all_p_le_0_1_1pt": all(pd.notna(p) and p <= STRICT_P_THRESHOLD for p in pvals_1pt.values()),
        "all_p_le_0_1_2pt": all(pd.notna(p) and p <= STRICT_P_THRESHOLD for p in pvals_2pt.values()),
        "all_p_le_0_15_1pt": all(pd.notna(p) and p <= NEAR_P_THRESHOLD for p in pvals_1pt.values()),
        "all_p_le_0_15_2pt": all(pd.notna(p) and p <= NEAR_P_THRESHOLD for p in pvals_2pt.values()),
        "max_p_1pt": max(pvals_1pt.values()),
        "max_p_2pt": max(pvals_2pt.values()),
    }

    detail_rows = []
    for method, reg_result, pvals in [("1pt", reg_1pt, pvals_1pt), ("2pt", reg_2pt, pvals_2pt)]:
        for var in ["const"] + feature_cols:
            detail_rows.append(
                {
                    "model_name": model_name,
                    "variables": " + ".join(feature_cols),
                    "filter_rule": summary_row["filter_rule"],
                    "method": method,
                    "variable": var,
                    "coefficient": reg_result.params.get(var, np.nan),
                    "p_value": reg_result.pvalues.get(var, np.nan),
                }
            )

    for var in feature_cols:
        summary_row[f"coef_1pt_{var}"] = reg_1pt.params.get(var, np.nan)
        summary_row[f"p_1pt_{var}"] = reg_1pt.pvalues.get(var, np.nan)
        summary_row[f"coef_2pt_{var}"] = reg_2pt.params.get(var, np.nan)
        summary_row[f"p_2pt_{var}"] = reg_2pt.pvalues.get(var, np.nan)

    return summary_row, detail_rows


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        [
            "all_p_le_0_1_1pt",
            "all_p_le_0_15_1pt",
            "delta_1pt_minus_2pt",
            "r2_os_1pt",
            "max_p_1pt",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_and_align()
    base_n = len(merged)
    single_rules = make_single_rules(merged)

    summary_rows = []
    detail_rows = []

    for model_name, feature_cols in MODELS.items():
        for rule_set in generate_rule_sets(single_rules):
            summary_row, detail = evaluate_model_on_sample(model_name, feature_cols, merged, rule_set, base_n)
            if summary_row is None:
                continue
            summary_rows.append(summary_row)
            detail_rows.extend(detail)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = OUTPUT_DIR / "fixed_models_raw_7d_sample_search_summary.csv"
    detail_path = OUTPUT_DIR / "fixed_models_raw_7d_sample_search_coefficients.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    top_frames = []
    for model_name in MODELS:
        sub = summary_df[summary_df["model_name"] == model_name].copy()
        if sub.empty:
            continue
        top_frames.append(sort_candidates(sub).head(TOP_N_PER_MODEL))

    if top_frames:
        top_df = pd.concat(top_frames, ignore_index=True)
    else:
        top_df = pd.DataFrame()

    top_path = OUTPUT_DIR / "fixed_models_raw_7d_top_candidates.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    print(f"Total candidates evaluated: {len(summary_df)}")
    print(f"Summary output: {summary_path}")
    print(f"Coefficient output: {detail_path}")
    print(f"Top candidates output: {top_path}")


if __name__ == "__main__":
    main()
