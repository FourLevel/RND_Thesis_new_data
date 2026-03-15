import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def prepare_dataset(csv_path, feature_cols, target_col="T Return", date_col="Observation Date"):
    df = pd.read_csv(csv_path).copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # 建立 R_{t+1}
    df["target_next"] = df[target_col].shift(-1)

    required_cols = feature_cols + ["target_next"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def out_of_sample_analysis(data, initial_window, target_col, feature_cols, window_type="expanding"):
    results = []
    T = len(data)
    initial_window = int(initial_window)

    if T <= initial_window + 1:
        return pd.DataFrame(), np.nan

    for t in range(initial_window, T):
        if window_type == "expanding":
            train_data = data.iloc[:t]
        elif window_type == "fixed":
            train_data = data.iloc[t - initial_window:t]
        else:
            raise ValueError("window_type must be 'expanding' or 'fixed'")

        # t 時點特徵，預測 t 時點對應的 target_next (即原始的 t+1 報酬)
        test_row = data.iloc[t:t + 1]

        y_train = train_data[target_col].values
        X_train_raw = train_data[feature_cols].values
        X_test_raw = test_row[feature_cols].values

        # 僅用訓練窗做標準化，避免 look-ahead bias
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predicted_value = model.predict(X_test)[0]
        actual_value = test_row[target_col].values[0]
        historical_mean = y_train.mean()

        results.append(
            {
                "time": t,
                "actual": actual_value,
                "predicted": predicted_value,
                "historical_mean": historical_mean,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df, np.nan

    numerator = np.sum((results_df["actual"] - results_df["predicted"]) ** 2)
    denominator = np.sum((results_df["actual"] - results_df["historical_mean"]) ** 2)

    if denominator == 0:
        return results_df, np.nan

    r2_os = 1 - numerator / denominator
    return results_df, r2_os


def compare_1pt_2pt(
    df_1pt,
    df_2pt,
    horizon_name,
    feature_pool,
    initial_window,
    window_type="expanding",
    min_vars=1,
    max_vars=4,
):
    if max_vars is None:
        max_vars = len(feature_pool)

    rows = []
    for k in range(min_vars, max_vars + 1):
        for combo in itertools.combinations(feature_pool, k):
            features = list(combo)

            _, r2_1 = out_of_sample_analysis(
                data=df_1pt,
                initial_window=initial_window,
                target_col="target_next",
                feature_cols=features,
                window_type=window_type,
            )

            _, r2_2 = out_of_sample_analysis(
                data=df_2pt,
                initial_window=initial_window,
                target_col="target_next",
                feature_cols=features,
                window_type=window_type,
            )

            if np.isnan(r2_1) or np.isnan(r2_2):
                continue

            delta = r2_1 - r2_2
            rows.append(
                {
                    "horizon": horizon_name,
                    "variables": " + ".join(features),
                    "num_vars": len(features),
                    "r2_os_1pt": r2_1,
                    "r2_os_2pt": r2_2,
                    "delta_1pt_minus_2pt": delta,
                    "winner": "1pt" if delta > 0 else ("2pt" if delta < 0 else "tie"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "horizon",
                "variables",
                "num_vars",
                "r2_os_1pt",
                "r2_os_2pt",
                "delta_1pt_minus_2pt",
                "winner",
            ]
        )

    out = (
        pd.DataFrame(rows)
        .sort_values("delta_1pt_minus_2pt", ascending=False)
        .reset_index(drop=True)
    )
    return out


def main():
    base_dir = Path("output/regression_cleaned_data")
    output_dir = Path("output/model_selections")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_pool = [
        "Mean",
        "Std",
        "Skewness",
        "Kurtosis",
        "Median",
        "Fear and Greed Index",
        "VIX",
        "T-1 Return",
        "T-2 Return",
        "T-3 Return",
        "T-4 Return",
    ]

    initial_window = 120
    window_type = "expanding"  # 或 "fixed"

    # 1d
    df_1pt_1d = prepare_dataset(base_dir / "RND_regression_all_1pt_1d_20260312.csv", feature_pool)
    df_2pt_1d = prepare_dataset(base_dir / "RND_regression_all_2pt_1d_20260312.csv", feature_pool)
    cmp_1d = compare_1pt_2pt(
        df_1pt_1d,
        df_2pt_1d,
        "1d",
        feature_pool,
        initial_window=initial_window,
        window_type=window_type,
    )
    cmp_1d.to_csv(output_dir / "oos_compare_1d.csv", index=False, encoding="utf-8-sig")

    # 7d
    df_1pt_7d = prepare_dataset(base_dir / "RND_regression_all_1pt_7d_20260312.csv", feature_pool)
    df_2pt_7d = prepare_dataset(base_dir / "RND_regression_all_2pt_7d_20260312.csv", feature_pool)
    cmp_7d = compare_1pt_2pt(
        df_1pt_7d,
        df_2pt_7d,
        "7d",
        feature_pool,
        initial_window=initial_window,
        window_type=window_type,
    )
    cmp_7d.to_csv(output_dir / "oos_compare_7d.csv", index=False, encoding="utf-8-sig")

    # 只看 1pt 優於 2pt
    best_1d = cmp_1d[cmp_1d["delta_1pt_minus_2pt"] > 0]
    best_7d = cmp_7d[cmp_7d["delta_1pt_minus_2pt"] > 0]

    best_1d.to_csv(output_dir / "oos_better_1pt_than_2pt_1d.csv", index=False, encoding="utf-8-sig")
    best_7d.to_csv(output_dir / "oos_better_1pt_than_2pt_7d.csv", index=False, encoding="utf-8-sig")

    print("1d: 1pt 優於 2pt 的組合數 =", len(best_1d))
    print("7d: 1pt 優於 2pt 的組合數 =", len(best_7d))


if __name__ == "__main__":
    main()