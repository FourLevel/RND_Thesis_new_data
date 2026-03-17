from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_DIR = Path("output/selected_raw_7d_models")
OUTPUT_DIR = BASE_DIR
TARGET_COL = "T Return"
DATE_COL = "Observation Date"

WINDOW_TYPE = "expanding"
INITIAL_WINDOW = 120

CANDIDATES = {
    "model_1_selected": ["Skewness", "Median", "Fear and Greed Index"],
    "model_2_best_oos": ["Kurtosis", "Median", "Fear and Greed Index"],
}


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


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    model_df = df[[y_col] + x_cols].dropna().copy()
    X_raw = model_df[x_cols].copy()
    X_std = X_raw.std(ddof=0).replace(0, 1.0)
    X_scaled = (X_raw - X_raw.mean()) / X_std
    X = sm.add_constant(X_scaled, has_constant="add")
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
        start = 0 if window_type == "expanding" else t - initial_window
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


def build_regression_oos_table(df_1pt: pd.DataFrame, df_2pt: pd.DataFrame, selected_vars: list[str]) -> pd.DataFrame:
    result_1pt, mse_1pt = fit_ols(df_1pt, TARGET_COL, selected_vars)
    result_2pt, mse_2pt = fit_ols(df_2pt, TARGET_COL, selected_vars)

    r2_os_1pt = rolling_oos_r2(df_1pt, TARGET_COL, selected_vars, INITIAL_WINDOW, WINDOW_TYPE)
    r2_os_2pt = rolling_oos_r2(df_2pt, TARGET_COL, selected_vars, INITIAL_WINDOW, WINDOW_TYPE)

    rows = []
    for idx, var in enumerate(selected_vars):
        rows.append(
            {
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
        )

    rows.append(
        {
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
    )
    return pd.DataFrame(rows)


def main() -> None:
    for candidate_name, features in CANDIDATES.items():
        candidate_dir = BASE_DIR / candidate_name
        df_1pt = load_dataset(candidate_dir / f"{candidate_name}_1pt_raw_7d.csv")
        df_2pt = load_dataset(candidate_dir / f"{candidate_name}_2pt_raw_7d.csv")
        out = build_regression_oos_table(df_1pt, df_2pt, features)
        out.to_csv(candidate_dir / f"{candidate_name}_regression_oos_comparison.csv", index=False, encoding="utf-8-sig")

    print(f"Analysis output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
