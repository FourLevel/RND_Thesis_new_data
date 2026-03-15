import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


warnings.filterwarnings("ignore")


# ============================================================================
# Config
# ============================================================================
DATA_1PT_PATH = Path("output/regression_cleaned_data/RND_regression_all_1pt_1d_20260312.csv")
DATA_2PT_PATH = Path("output/regression_cleaned_data/RND_regression_all_2pt_1d_20260312.csv")
OUTPUT_DIR = Path("output/oos_sample_selection_1d")

DATE_COLS = ["Observation Date", "Expiration Date"]
TARGET_COL = "T Return"
REQUIRED_FACTOR = "Skewness"

# 用 1pt 資料做模型候選篩選；若想改成 2pt，可改成 "2pt"
MODEL_SCREENING_SOURCE = "1pt"

# 這裡控制候選模型的變數池與最大組合大小，避免無窮枚舉
CONTROL_VARIABLES = [
    "Mean",
    "Std",
    "Kurtosis",
    "Median",
    "Fear and Greed Index",
    "VIX",
    "T-1 Return",
    "T-2 Return",
    "T-3 Return",
    "T-4 Return",
]
MAX_CONTROLS_IN_MODEL = 4

P_THRESHOLD_REQUIRED = 0.10
P_THRESHOLD_CONTROLS = 0.50

WINDOW_TYPE = "expanding"  # "expanding" or "fixed"
INITIAL_WINDOW = 120

# 樣本篩選規則搜尋設定
FILTER_COLUMNS = [
    "Skewness",
    "Mean",
    "Std",
    "Kurtosis",
    "Median",
    "Fear and Greed Index",
    "VIX",
    "T-1 Return",
    "T-2 Return",
    "T-3 Return",
    "T-4 Return",
]
FILTER_QUANTILES = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]
MIN_RETAIN_RATIO = 0.70
MAX_RETAIN_RATIO = 0.80
MAX_FILTER_ROUNDS = 2
MIN_ROWS_AFTER_FILTER = 250
MAX_MODELS_FOR_FILTER_SEARCH = 100
N_JOBS = max(1, (os.cpu_count() or 2) - 1)


@dataclass
class OOSResult:
    r2_os_1pt: float
    r2_os_2pt: float
    delta_1pt_minus_2pt: float
    n_obs: int
    n_oos: int


def load_and_align_datasets(path_1pt: Path, path_2pt: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_1pt = pd.read_csv(path_1pt).copy()
    df_2pt = pd.read_csv(path_2pt).copy()

    for df in (df_1pt, df_2pt):
        df["Observation Date"] = pd.to_datetime(df["Observation Date"], errors="coerce")
        df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce")

    common_dates = df_1pt[DATE_COLS].merge(df_2pt[DATE_COLS], on=DATE_COLS, how="inner").drop_duplicates()

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

    merged = df_1pt.merge(df_2pt, on=DATE_COLS, suffixes=("_1pt", "_2pt"), how="inner")
    merged = merged.sort_values(DATE_COLS).reset_index(drop=True)

    merged["target_next_1pt"] = merged[f"{TARGET_COL}_1pt"].shift(-1)
    merged["target_next_2pt"] = merged[f"{TARGET_COL}_2pt"].shift(-1)

    return df_1pt, df_2pt, merged


def fit_ols_with_pvalues(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    model_df = df[[y_col] + x_cols].dropna().copy()
    X = sm.add_constant(model_df[x_cols], has_constant="add")
    y = model_df[y_col]
    return sm.OLS(y, X).fit()


def screen_candidate_models(
    df: pd.DataFrame,
    y_col: str,
    required_factor: str,
    control_vars: list[str],
    max_controls: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []

    control_cap = min(max_controls, len(control_vars))
    for k in range(0, control_cap + 1):
        for combo in itertools.combinations(control_vars, k):
            features = [required_factor] + list(combo)
            model_df = df[[y_col] + features].dropna().copy()
            if len(model_df) <= len(features) + 5:
                continue

            result = fit_ols_with_pvalues(model_df, y_col, features)
            pvalues = result.pvalues.to_dict()

            required_p = pvalues.get(required_factor, np.nan)
            control_ps = [pvalues.get(var, np.nan) for var in combo]

            pass_required = pd.notna(required_p) and required_p < P_THRESHOLD_REQUIRED
            pass_controls = all(pd.notna(p) and p < P_THRESHOLD_CONTROLS for p in control_ps)
            passed = pass_required and pass_controls

            model_id = f"model_{len(rows) + 1:04d}"
            combo_name = " + ".join(features)

            rows.append(
                {
                    "model_id": model_id,
                    "variables": combo_name,
                    "num_variables": len(features),
                    "n_obs": int(result.nobs),
                    "rsquared": result.rsquared,
                    "adj_rsquared": result.rsquared_adj,
                    "aic": result.aic,
                    "bic": result.bic,
                    "required_factor": required_factor,
                    "required_factor_pvalue": required_p,
                    "pass_pvalue_screen": passed,
                }
            )

            for var in ["const"] + features:
                coef_rows.append(
                    {
                        "model_id": model_id,
                        "variables": combo_name,
                        "variable": var,
                        "coefficient": result.params.get(var, np.nan),
                        "p_value": pvalues.get(var, np.nan),
                    }
                )

    models_df = pd.DataFrame(rows)
    coef_df = pd.DataFrame(coef_rows)
    return models_df, coef_df


def prepare_oos_arrays(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[np.ndarray, np.ndarray]:
    needed = feature_cols + [target_col]
    out = df[needed].dropna().copy()
    X = out[feature_cols].to_numpy(dtype=float)
    y = out[target_col].to_numpy(dtype=float)
    return X, y


def rolling_oos_r2(df: pd.DataFrame, feature_cols: list[str], target_col: str, initial_window: int, window_type: str) -> tuple[float, int]:
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

        # 僅用訓練窗做標準化，避免 look-ahead bias
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


def evaluate_oos(merged_df: pd.DataFrame, feature_cols: list[str], initial_window: int, window_type: str) -> OOSResult:
    df_1pt = merged_df[[f"{col}_1pt" for col in feature_cols] + ["target_next_1pt"]].copy()
    df_1pt.columns = feature_cols + ["target_next"]

    df_2pt = merged_df[[f"{col}_2pt" for col in feature_cols] + ["target_next_2pt"]].copy()
    df_2pt.columns = feature_cols + ["target_next"]

    r2_1pt, n_oos_1 = rolling_oos_r2(df_1pt, feature_cols, "target_next", initial_window, window_type)
    r2_2pt, n_oos_2 = rolling_oos_r2(df_2pt, feature_cols, "target_next", initial_window, window_type)

    n_oos = min(n_oos_1, n_oos_2)
    delta = np.nan if np.isnan(r2_1pt) or np.isnan(r2_2pt) else (r2_1pt - r2_2pt)

    return OOSResult(
        r2_os_1pt=r2_1pt,
        r2_os_2pt=r2_2pt,
        delta_1pt_minus_2pt=delta,
        n_obs=len(merged_df),
        n_oos=n_oos,
    )


def make_filter_candidates(df: pd.DataFrame, filter_columns: list[str], quantiles: list[float]) -> list[dict]:
    candidates = []
    for col in filter_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty or series.nunique() < 5:
            continue

        for q in quantiles:
            threshold = series.quantile(q)
            candidates.append(
                {
                    "filter_column": col,
                    "operator": "<=",
                    "threshold": float(threshold),
                    "rule_text": f"{col} <= {threshold:.6f}",
                }
            )
            candidates.append(
                {
                    "filter_column": col,
                    "operator": ">=",
                    "threshold": float(threshold),
                    "rule_text": f"{col} >= {threshold:.6f}",
                }
            )
    return candidates


def apply_filter_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    col = rule["filter_column"]
    threshold = rule["threshold"]
    if rule["operator"] == "<=":
        return df[df[col] <= threshold].copy()
    return df[df[col] >= threshold].copy()


def fit_final_coefficients(merged_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for label in ["1pt", "2pt"]:
        y_col = f"{TARGET_COL}_{label}"
        x_cols = [f"{col}_{label}" for col in feature_cols]

        model_df = merged_df[[y_col] + x_cols].dropna().copy()
        if len(model_df) <= len(x_cols) + 5:
            continue

        X = sm.add_constant(model_df[x_cols], has_constant="add")
        X.columns = ["const"] + feature_cols
        y = model_df[y_col]
        result = sm.OLS(y, X).fit()

        for var in ["const"] + feature_cols:
            rows.append(
                {
                    "method": label,
                    "variable": var,
                    "coefficient": result.params.get(var, np.nan),
                    "p_value": result.pvalues.get(var, np.nan),
                }
            )
    return pd.DataFrame(rows)


def search_best_filters_for_model(
    merged_df: pd.DataFrame,
    feature_cols: list[str],
    screening_df: pd.DataFrame,
) -> tuple[list[dict], OOSResult, pd.DataFrame]:
    base_result = evaluate_oos(merged_df, feature_cols, INITIAL_WINDOW, WINDOW_TYPE)
    accepted_results = []

    base_retention = 1.0
    accepted_results.append(
        {
            "rules": [],
            "rule_text": "No filter",
            "filtered_df": merged_df.copy(),
            "screening_df": screening_df.copy(),
            "oos": base_result,
            "retention_ratio": base_retention,
        }
    )

    current_best = accepted_results[0]

    used_rules = set()
    for _ in range(MAX_FILTER_ROUNDS):
        candidate_pool = make_filter_candidates(current_best["screening_df"], FILTER_COLUMNS, FILTER_QUANTILES)
        trial_results = []

        for rule in candidate_pool:
            rule_key = (rule["filter_column"], rule["operator"], round(rule["threshold"], 10))
            if rule_key in used_rules:
                continue

            filtered_screening = apply_filter_rule(current_best["screening_df"], rule)
            filtered_merged = apply_filter_rule(current_best["filtered_df"], rule)

            retain_ratio = len(filtered_merged) / len(merged_df) if len(merged_df) else 0.0

            if len(filtered_merged) < MIN_ROWS_AFTER_FILTER:
                continue
            if retain_ratio < MIN_RETAIN_RATIO:
                continue

            oos = evaluate_oos(filtered_merged, feature_cols, INITIAL_WINDOW, WINDOW_TYPE)
            if np.isnan(oos.r2_os_1pt) or np.isnan(oos.r2_os_2pt):
                continue

            trial_results.append(
                {
                    "rule": rule,
                    "filtered_df": filtered_merged,
                    "screening_df": filtered_screening,
                    "oos": oos,
                    "retention_ratio": retain_ratio,
                }
            )

        if not trial_results:
            break

        trial_results = sorted(
            trial_results,
            key=lambda x: (
                x["oos"].delta_1pt_minus_2pt,
                x["oos"].r2_os_1pt,
                -abs(x["retention_ratio"] - ((MIN_RETAIN_RATIO + MAX_RETAIN_RATIO) / 2)),
            ),
            reverse=True,
        )

        best_trial = trial_results[0]
        if np.isnan(best_trial["oos"].delta_1pt_minus_2pt):
            break

        if best_trial["oos"].delta_1pt_minus_2pt <= current_best["oos"].delta_1pt_minus_2pt:
            break

        used_rules.add((best_trial["rule"]["filter_column"], best_trial["rule"]["operator"], round(best_trial["rule"]["threshold"], 10)))
        rules = current_best["rules"] + [best_trial["rule"]]
        current_best = {
            "rules": rules,
            "rule_text": " AND ".join(rule["rule_text"] for rule in rules),
            "filtered_df": best_trial["filtered_df"],
            "screening_df": best_trial["screening_df"],
            "oos": best_trial["oos"],
            "retention_ratio": best_trial["retention_ratio"],
        }
        accepted_results.append(current_best)

        if current_best["retention_ratio"] <= MAX_RETAIN_RATIO:
            break

    return accepted_results, current_best["oos"], current_best["filtered_df"]


def build_merged_screening_frame(merged_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = DATE_COLS + [f"{col}_1pt" for col in FILTER_COLUMNS]
    out = merged_df[keep_cols].copy()
    rename_map = {f"{col}_1pt": col for col in FILTER_COLUMNS}
    return out.rename(columns=rename_map)


def evaluate_single_model_task(model_row: dict, merged: pd.DataFrame, merged_screening: pd.DataFrame) -> tuple[dict | None, pd.DataFrame | None]:
    features = model_row["variables"].split(" + ")
    model_id = model_row["model_id"]

    base_needed_cols = (
        [f"{col}_1pt" for col in features]
        + [f"{col}_2pt" for col in features]
        + [f"{TARGET_COL}_1pt", f"{TARGET_COL}_2pt"]
        + ["target_next_1pt", "target_next_2pt"]
    )
    model_merged = pd.concat([merged[DATE_COLS], merged[base_needed_cols], merged_screening[FILTER_COLUMNS]], axis=1)
    model_merged = model_merged.dropna(subset=base_needed_cols).reset_index(drop=True)
    model_screening = model_merged[DATE_COLS + FILTER_COLUMNS].copy()

    if len(model_merged) < MIN_ROWS_AFTER_FILTER:
        return None, None

    filter_history, best_oos, best_filtered_df = search_best_filters_for_model(
        merged_df=model_merged,
        feature_cols=features,
        screening_df=model_screening,
    )

    best_step = filter_history[-1]
    passed_oos = (
        pd.notna(best_oos.r2_os_1pt)
        and pd.notna(best_oos.r2_os_2pt)
        and best_oos.r2_os_1pt > 0
        and best_oos.r2_os_1pt > best_oos.r2_os_2pt
        and best_step["retention_ratio"] >= MIN_RETAIN_RATIO
        and best_step["retention_ratio"] <= 1.0
    )

    summary_row = {
        "model_id": model_id,
        "variables": model_row["variables"],
        "num_variables": model_row["num_variables"],
        "screening_required_factor_pvalue": model_row["required_factor_pvalue"],
        "final_filter_rule": best_step["rule_text"],
        "n_obs_after_filter": best_oos.n_obs,
        "retention_ratio": best_step["retention_ratio"],
        "n_oos": best_oos.n_oos,
        "r2_os_1pt": best_oos.r2_os_1pt,
        "r2_os_2pt": best_oos.r2_os_2pt,
        "delta_1pt_minus_2pt": best_oos.delta_1pt_minus_2pt,
        "pass_oos_rule": passed_oos,
    }

    final_coef_df = fit_final_coefficients(best_filtered_df, features)
    if not final_coef_df.empty:
        final_coef_df.insert(0, "model_id", model_id)
        final_coef_df.insert(1, "variables", model_row["variables"])
        final_coef_df["final_filter_rule"] = best_step["rule_text"]
        final_coef_df["retention_ratio"] = best_step["retention_ratio"]
        return summary_row, final_coef_df

    return summary_row, pd.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_1pt, df_2pt, merged = load_and_align_datasets(DATA_1PT_PATH, DATA_2PT_PATH)
    screening_source_df = df_1pt if MODEL_SCREENING_SOURCE == "1pt" else df_2pt

    models_df, model_coef_df = screen_candidate_models(
        df=screening_source_df,
        y_col=TARGET_COL,
        required_factor=REQUIRED_FACTOR,
        control_vars=CONTROL_VARIABLES,
        max_controls=MAX_CONTROLS_IN_MODEL,
    )

    models_df = models_df.sort_values(
        ["pass_pvalue_screen", "required_factor_pvalue", "adj_rsquared"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    models_path = OUTPUT_DIR / "candidate_models_1d.csv"
    coef_path = OUTPUT_DIR / "candidate_model_coefficients_1d.csv"
    models_df.to_csv(models_path, index=False, encoding="utf-8-sig")
    model_coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

    passed_models = models_df[models_df["pass_pvalue_screen"]].head(MAX_MODELS_FOR_FILTER_SEARCH).copy()

    summary_rows = []
    final_coef_rows = []

    merged_screening = build_merged_screening_frame(merged)
    task_rows = passed_models.to_dict("records")

    if N_JOBS == 1 or len(task_rows) <= 1:
        task_results = [evaluate_single_model_task(model_row, merged, merged_screening) for model_row in task_rows]
    else:
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            task_results = list(
                executor.map(
                    evaluate_single_model_task,
                    task_rows,
                    itertools.repeat(merged),
                    itertools.repeat(merged_screening),
                )
            )

    for summary_row, final_coef_df in task_results:
        if summary_row is None:
            continue
        summary_rows.append(summary_row)
        if final_coef_df is not None and not final_coef_df.empty:
            final_coef_rows.append(final_coef_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["pass_oos_rule", "delta_1pt_minus_2pt", "r2_os_1pt"],
        ascending=[False, False, False],
    )
    summary_path = OUTPUT_DIR / "final_model_oos_summary_1d.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if final_coef_rows:
        final_coef_out = pd.concat(final_coef_rows, ignore_index=True)
    else:
        final_coef_out = pd.DataFrame(
            columns=[
                "model_id",
                "variables",
                "method",
                "variable",
                "coefficient",
                "p_value",
                "final_filter_rule",
                "retention_ratio",
            ]
        )
    final_coef_path = OUTPUT_DIR / "final_model_coefficients_1d.csv"
    final_coef_out.to_csv(final_coef_path, index=False, encoding="utf-8-sig")

    passed_summary = summary_df[summary_df["pass_oos_rule"]].copy()
    passed_summary_path = OUTPUT_DIR / "passed_oos_models_1d.csv"
    passed_summary.to_csv(passed_summary_path, index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("完成 1d 模型搜尋、樣本篩選與 OOS 比較")
    print(f"平行工作數: {N_JOBS}")
    print(f"候選模型總數: {len(models_df)}")
    print(f"通過 p-value 篩選的模型數: {len(passed_models)}")
    print(f"通過最終 OOS 條件的模型數: {len(passed_summary)}")
    print(f"候選模型輸出: {models_path}")
    print(f"候選模型係數輸出: {coef_path}")
    print(f"最終 OOS 摘要輸出: {summary_path}")
    print(f"最終模型係數輸出: {final_coef_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
