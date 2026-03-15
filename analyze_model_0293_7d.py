from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_1PT_PATH = Path("output/oos_sample_selection_7d/model_0293_1pt_7d_filtered.csv")
DATA_2PT_PATH = Path("output/oos_sample_selection_7d/model_0293_2pt_7d_filtered.csv")
OUTPUT_DIR = Path("output/oos_sample_selection_7d/model_0293_analysis")

DATE_COL = "Observation Date"
TARGET_COL = "T Return"
SELECTED_VARS = ["Kurtosis", "VIX", "T-1 Return"]

WINDOW_TYPE = "expanding"  # "expanding" or "fixed"
INITIAL_WINDOW = 120


def get_significance(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def ordered_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    remaining_cols = [col for col in numeric_cols if col != TARGET_COL]
    return [TARGET_COL] + remaining_cols if TARGET_COL in numeric_cols else numeric_cols


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ordered_numeric_columns(df)
    desc = df[cols].describe().T
    desc.insert(0, "Variable", desc.index)
    desc = desc.reset_index(drop=True)
    return desc


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ordered_numeric_columns(df)
    corr = df[cols].corr()
    corr = corr.loc[cols, cols]
    corr.index.name = "Variable"
    return corr


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    model_df = df[[y_col] + x_cols].dropna().copy()
    X = sm.add_constant(model_df[x_cols], has_constant="add")
    y = model_df[y_col]
    result = sm.OLS(y, X).fit()
    preds = result.predict(X)
    mse = float(np.mean((y - preds) ** 2))
    return result, mse


def rolling_oos_r2(df: pd.DataFrame, target_col: str, feature_cols: list[str], initial_window: int, window_type: str) -> float:
    data = df[[target_col] + feature_cols].copy()
    data["target_next"] = data[target_col].shift(-1)
    data = data.dropna().reset_index(drop=True)

    X = data[feature_cols].to_numpy(dtype=float)
    y = data["target_next"].to_numpy(dtype=float)
    total_rows = len(y)

    if total_rows <= initial_window + 1:
        return np.nan

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
        return np.nan
    return 1 - numerator / denominator


def build_regression_oos_table(df_1pt: pd.DataFrame, df_2pt: pd.DataFrame) -> pd.DataFrame:
    result_1pt, mse_1pt = fit_ols(df_1pt, TARGET_COL, SELECTED_VARS)
    result_2pt, mse_2pt = fit_ols(df_2pt, TARGET_COL, SELECTED_VARS)

    r2_os_1pt = rolling_oos_r2(df_1pt, TARGET_COL, SELECTED_VARS, INITIAL_WINDOW, WINDOW_TYPE)
    r2_os_2pt = rolling_oos_r2(df_2pt, TARGET_COL, SELECTED_VARS, INITIAL_WINDOW, WINDOW_TYPE)

    rows = []
    for idx, var in enumerate(SELECTED_VARS):
        row = {
            "Variable": var,
            "Proposed_Coef": result_1pt.params.get(var, np.nan),
            "Proposed_p_value": result_1pt.pvalues.get(var, np.nan),
            "Proposed_Sig": get_significance(result_1pt.pvalues.get(var, np.nan)),
            "Proposed_R_squared": result_1pt.rsquared if idx == 0 else np.nan,
            "Proposed_MSE": mse_1pt if idx == 0 else np.nan,
            "Proposed_R2_OS": r2_os_1pt if idx == 0 else np.nan,
            "BF_Coef": result_2pt.params.get(var, np.nan),
            "BF_p_value": result_2pt.pvalues.get(var, np.nan),
            "BF_Sig": get_significance(result_2pt.pvalues.get(var, np.nan)),
            "BF_R_squared": result_2pt.rsquared if idx == 0 else np.nan,
            "BF_MSE": mse_2pt if idx == 0 else np.nan,
            "BF_R2_OS": r2_os_2pt if idx == 0 else np.nan,
        }
        rows.append(row)

    meta_row = {
        "Variable": "OOS Setting",
        "Proposed_Coef": f"{WINDOW_TYPE}, initial_window={INITIAL_WINDOW}",
        "Proposed_p_value": np.nan,
        "Proposed_Sig": "",
        "Proposed_R_squared": np.nan,
        "Proposed_MSE": np.nan,
        "Proposed_R2_OS": np.nan,
        "BF_Coef": f"{WINDOW_TYPE}, initial_window={INITIAL_WINDOW}",
        "BF_p_value": np.nan,
        "BF_Sig": "",
        "BF_R_squared": np.nan,
        "BF_MSE": np.nan,
        "BF_R2_OS": np.nan,
    }
    rows.append(meta_row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_1pt = load_dataset(DATA_1PT_PATH)
    df_2pt = load_dataset(DATA_2PT_PATH)

    descriptive_stats(df_1pt).to_csv(
        OUTPUT_DIR / "model_0293_1pt_descriptive_stats_7d.csv",
        index=False,
        encoding="utf-8-sig",
    )
    descriptive_stats(df_2pt).to_csv(
        OUTPUT_DIR / "model_0293_2pt_descriptive_stats_7d.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlation_matrix(df_1pt).to_csv(
        OUTPUT_DIR / "model_0293_1pt_correlation_matrix_7d.csv",
        encoding="utf-8-sig",
    )
    correlation_matrix(df_2pt).to_csv(
        OUTPUT_DIR / "model_0293_2pt_correlation_matrix_7d.csv",
        encoding="utf-8-sig",
    )

    reg_table = build_regression_oos_table(df_1pt, df_2pt)
    reg_table.to_csv(
        OUTPUT_DIR / "model_0293_regression_oos_comparison_7d.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Analysis completed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
