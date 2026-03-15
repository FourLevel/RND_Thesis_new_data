import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


warnings.filterwarnings("ignore")


def get_significance(p_value):
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    return ""


def standardize_columns(df, cols):
    df = df.copy()
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            # 避免除以 0
            df[col] = 0.0
        else:
            df[col] = (df[col] - mean) / std
    return df


def run_fixed_factor_regression(df, y_col, fixed_factor, candidate_vars):
    result_col_prefix = fixed_factor
    results = []

    y = df[y_col]

    # 先跑「僅 fixed_factor」
    X_single = sm.add_constant(df[[fixed_factor]])
    model_single = sm.OLS(y, X_single).fit()
    y_pred_single = model_single.predict(X_single)
    mse_single = np.mean((y - y_pred_single) ** 2)

    p_fixed_single = model_single.pvalues[fixed_factor]
    results.append({
        "變數組合": f"僅 {fixed_factor}",
        f"{result_col_prefix}_係數": model_single.params[fixed_factor],
        f"{result_col_prefix}_p值": p_fixed_single,
        f"{result_col_prefix}_顯著性": get_significance(p_fixed_single),
        "變數數量": 1,
        "R平方": model_single.rsquared,
        "MSE": mse_single
    })

    min_p_value = p_fixed_single
    best_combination = f"僅 {fixed_factor}"

    # 再跑 fixed_factor + 任意組合
    for k in range(1, len(candidate_vars) + 1):
        for combo in itertools.combinations(candidate_vars, k):
            all_vars = [fixed_factor] + list(combo)
            X = sm.add_constant(df[all_vars])

            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)

            p_fixed = model.pvalues[fixed_factor]
            combo_name = f"{fixed_factor} + " + " + ".join(combo)

            if p_fixed < min_p_value:
                min_p_value = p_fixed
                best_combination = combo_name

            results.append({
                "變數組合": combo_name,
                f"{result_col_prefix}_係數": model.params[fixed_factor],
                f"{result_col_prefix}_p值": p_fixed,
                f"{result_col_prefix}_顯著性": get_significance(p_fixed),
                "變數數量": len(all_vars),
                "R平方": model.rsquared,
                "MSE": mse
            })

    result_df = pd.DataFrame(results).sort_values(by=f"{result_col_prefix}_p值")
    return result_df, best_combination, min_p_value


def analyze_one_dataset(dataset_name, csv_path, output_dir):
    print("\n" + "=" * 90)
    print(f"開始分析: {dataset_name}")
    print(f"資料來源: {csv_path}")
    print("=" * 90)

    df = pd.read_csv(csv_path)

    # 敘述統計欄位
    numeric_columns = [
        "T Return", "Mean", "Std", "Skewness", "Kurtosis", "Median",
        "Fear and Greed Index", "VIX",
        "T-1 Return", "T-2 Return", "T-3 Return", "T-4 Return"
    ]

    # 敘述統計
    stats_summary = df[numeric_columns].describe().T
    print("\n變數的敘述統計：")
    print(stats_summary.round(4))

    stats_path = output_dir / f"descriptive_stats_{dataset_name}.csv"
    stats_summary.to_csv(stats_path, encoding="utf-8-sig")
    print(f"\n敘述統計結果已儲存至: {stats_path}")

    # 標準化
    df_std = standardize_columns(df, numeric_columns)

    y_col = "T Return"

    # 固定 Skewness，排除 Skewness 本身
    skew_candidates = [
        "Mean", "Std", "Kurtosis", "Median", "Fear and Greed Index",
        "VIX", "T-1 Return", "T-2 Return", "T-3 Return", "T-4 Return"
    ]
    skew_results, skew_best, skew_min_p = run_fixed_factor_regression(
        df=df_std,
        y_col=y_col,
        fixed_factor="Skewness",
        candidate_vars=skew_candidates
    )

    print("\nSkewness 各變數組合的迴歸分析結果：")
    print(skew_results.round(4))
    if skew_min_p is None:
        print("\nSkewness: 沒有任何模型符合『所有變數 p < 0.1』條件")
    else:
        print(f"\nSkewness 的 p 值最小組合是「{skew_best}」，p 值為 {skew_min_p:.4f}")

    skew_path = output_dir / f"skewness_regression_results_{dataset_name}.csv"
    skew_results.to_csv(skew_path, index=False, encoding="utf-8-sig")
    print(f"Skewness 迴歸分析結果已儲存至: {skew_path}")

    # 固定 Kurtosis，排除 Kurtosis 本身
    kurt_candidates = [
        "Mean", "Std", "Skewness", "Median", "Fear and Greed Index",
        "VIX", "T-1 Return", "T-2 Return", "T-3 Return", "T-4 Return"
    ]
    kurt_results, kurt_best, kurt_min_p = run_fixed_factor_regression(
        df=df_std,
        y_col=y_col,
        fixed_factor="Kurtosis",
        candidate_vars=kurt_candidates
    )

    print("\nKurtosis 各變數組合的迴歸分析結果：")
    print(kurt_results.round(4))
    if kurt_min_p is None:
        print("\nKurtosis: 沒有任何模型符合『所有變數 p < 0.1』條件")
    else:
        print(f"\nKurtosis 的 p 值最小組合是「{kurt_best}」，p 值為 {kurt_min_p:.4f}")

    kurt_path = output_dir / f"kurtosis_regression_results_{dataset_name}.csv"
    kurt_results.to_csv(kurt_path, index=False, encoding="utf-8-sig")
    print(f"Kurtosis 迴歸分析結果已儲存至: {kurt_path}")


def main():
    base_dir = Path("output/regression_cleaned_data")
    output_dir = Path("output/model_selections")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "1pt_1d": base_dir / "RND_regression_all_1pt_1d_20260312.csv",
        "2pt_1d": base_dir / "RND_regression_all_2pt_1d_20260312.csv",
        "1pt_7d": base_dir / "RND_regression_all_1pt_7d_20260312.csv",
        "2pt_7d": base_dir / "RND_regression_all_2pt_7d_20260312.csv",
    }

    for name, path in datasets.items():
        if not path.exists():
            print(f"\n警告: 找不到檔案，略過 {name} -> {path}")
            continue
        analyze_one_dataset(name, path, output_dir)

    print("\n全部分析完成。")


if __name__ == "__main__":
    main()