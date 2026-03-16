from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats

import main_20260312 as main_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 1pt vs 2pt GPD runtime on a fixed set of valid observation/expiration pairs."
    )
    parser.add_argument(
        "--input-csv",
        default="output/regression_cleaned_data/RND_regression_all_1pt_7d_20260312.csv",
        help="CSV used to select valid observation/expiration pairs.",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of valid pairs to benchmark.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated runs per method.")
    parser.add_argument("--lookback-days", type=int, default=7, help="Observation-to-expiration gap in days.")
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="How to choose the benchmark pairs from the input CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when selection=random.")
    parser.add_argument(
        "--output-dir",
        default="output/runtime_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed warm-up case per method before the benchmark.",
    )
    return parser.parse_args()


def load_pairs(input_csv: Path, sample_size: int, lookback_days: int, selection: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required_columns = {"Observation Date", "Expiration Date"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    pairs = df[["Observation Date", "Expiration Date"]].drop_duplicates().copy()
    pairs["Observation Date"] = pd.to_datetime(pairs["Observation Date"])
    pairs["Expiration Date"] = pd.to_datetime(pairs["Expiration Date"])
    pairs["gap_days"] = (pairs["Expiration Date"] - pairs["Observation Date"]).dt.days
    pairs = pairs.loc[pairs["gap_days"] == lookback_days].sort_values(
        ["Observation Date", "Expiration Date"], kind="stable"
    )

    if len(pairs) < sample_size:
        raise ValueError(
            f"Only {len(pairs)} valid {lookback_days}-day pairs found in {input_csv}, fewer than sample-size={sample_size}."
        )

    if selection == "random":
        pairs = pairs.sample(n=sample_size, random_state=seed).sort_values(
            ["Observation Date", "Expiration Date"], kind="stable"
        )
    else:
        pairs = pairs.head(sample_size)

    pairs["Observation Date"] = pairs["Observation Date"].dt.strftime("%Y-%m-%d")
    pairs["Expiration Date"] = pairs["Expiration Date"].dt.strftime("%Y-%m-%d")
    return pairs.reset_index(drop=True)


def fit_gpd_tails_use_slope_and_cdf_with_one_point_optimized(
    fit: pd.DataFrame,
    initial_i: int,
    delta_x: float,
    alpha_1L: float = 0.05,
    alpha_1R: float = 0.95,
) -> tuple[pd.DataFrame, float, float]:
    if fit["left_cumulative"].iloc[0] > alpha_1L:
        raise ValueError(
            f"left_cumulative[0] ({fit['left_cumulative'].iloc[0]:.4f}) 大於 alpha_1L ({alpha_1L})，"
            "RND 資料不足以進行左尾 GPD 擬合，跳過此日期。"
        )

    left_cumulative = fit["left_cumulative"]
    right_cumulative = fit["right_cumulative"]
    strike_price = fit["strike_price"]
    rnd_density = fit["RND_density"]

    right_idx = (left_cumulative - alpha_1R).abs().idxmin()
    loc_right = fit.loc[right_idx]
    right_end = float(loc_right["strike_price"])
    right_missing_tail = float(loc_right["right_cumulative"])
    right_density = float(loc_right["RND_density"])
    right_sigma_guess = right_missing_tail / right_density
    loc_right_pos = fit.index.get_indexer([right_idx])[0]

    right_candidates = rnd_density.iloc[:loc_right_pos].to_numpy()
    if right_candidates.size >= initial_i:
        steps = main_module.np.arange(initial_i, right_candidates.size + 1)
        compared = right_candidates[::-1][initial_i - 1 :]
        right_slopes = (compared - right_density) / (steps * delta_x)
        negative_mask = right_slopes < 0
        if negative_mask.any():
            right_slope = float(right_slopes[negative_mask.argmax()])
        else:
            right_slope = float(right_slopes[-1])
    else:
        right_slope = 0.0

    def right_objective(x: main_module.np.ndarray) -> float:
        xi, scale = x
        slope_error = (
            (right_missing_tail * main_module.gpd.pdf(right_end + delta_x, xi, loc=right_end, scale=scale) - right_density)
            / delta_x
            - right_slope
        )
        cdf_error = main_module.gpd.cdf(right_end, xi, loc=right_end, scale=scale) - right_missing_tail
        return (1e12 * slope_error**2) + (1e12 * cdf_error**2)

    right_fit = main_module.minimize(
        right_objective,
        [0, right_sigma_guess],
        bounds=[(-1, 1), (0, main_module.np.inf)],
        method="SLSQP",
    )
    right_xi, right_sigma = right_fit.x

    right_extension = pd.DataFrame(
        {"strike_price": main_module.np.arange(float(strike_price.max()) + delta_x, 160010, delta_x)}
    )
    fit = pd.concat([fit, right_extension], ignore_index=True, copy=False)
    fit["right_extra_density"] = right_missing_tail * main_module.gpd.pdf(
        fit["strike_price"], right_xi, loc=right_end, scale=right_sigma
    )
    fit["full_density"] = fit["RND_density"].fillna(0.0)
    right_mask = fit["strike_price"] >= right_end
    fit.loc[right_mask, "full_density"] = fit.loc[right_mask, "right_extra_density"].fillna(0.0)

    fit["reverse_strike"] = fit["strike_price"].max() - fit["strike_price"]
    left_idx = (fit["left_cumulative"] - alpha_1L).abs().idxmin()
    loc_left = fit.loc[left_idx]
    left_end = float(loc_left["strike_price"])
    left_missing_tail = float(loc_left["left_cumulative"])
    left_density = float(loc_left["RND_density"])
    left_sigma_guess = left_missing_tail / left_density
    left_reverse = float(loc_left["reverse_strike"])
    loc_left_pos = fit.index.get_indexer([left_idx])[0]

    left_candidates = fit["RND_density"].iloc[loc_left_pos + initial_i :].to_numpy()
    if left_candidates.size > 0:
        steps = main_module.np.arange(initial_i, initial_i + left_candidates.size)
        left_slopes = (left_candidates - left_density) / (steps * delta_x)
        positive_mask = left_slopes > 0
        if positive_mask.any():
            left_slope = float(left_slopes[positive_mask.argmax()])
        else:
            left_slope = float(left_slopes[-1])
    else:
        left_slope = 0.0

    def left_objective(x: main_module.np.ndarray) -> float:
        xi, scale = x
        slope_error = (
            (left_missing_tail * main_module.gpd.pdf(left_reverse + delta_x, xi, loc=left_reverse, scale=scale) - left_density)
            / delta_x
            - left_slope
        )
        cdf_error = main_module.gpd.cdf(left_reverse, xi, loc=left_reverse, scale=scale) - left_missing_tail
        return (1e12 * slope_error**2) + (1e12 * cdf_error**2)

    left_fit = main_module.minimize(
        left_objective,
        [0, left_sigma_guess],
        bounds=[(-1, 1), (0, main_module.np.inf)],
        method="SLSQP",
    )
    left_xi, left_sigma = left_fit.x

    left_extension = pd.DataFrame(
        {"strike_price": main_module.np.arange(0, float(fit["strike_price"].min()) - delta_x, delta_x)}
    )
    fit = pd.concat([fit, left_extension], ignore_index=True, copy=False)
    fit["reverse_strike"] = fit["strike_price"].max() - fit["strike_price"]
    fit["left_extra_density"] = left_missing_tail * main_module.gpd.pdf(
        fit["reverse_strike"], left_xi, loc=left_reverse, scale=left_sigma
    )
    left_mask = fit["strike_price"] <= left_end
    fit.loc[left_mask, "full_density"] = fit.loc[left_mask, "left_extra_density"].fillna(0.0)

    fit = fit.sort_values("strike_price")
    fit["full_density"] = fit["full_density"].interpolate(method="cubic")
    fit["full_density_cumulative"] = fit["full_density"].cumsum() * delta_x
    return fit, left_end, right_end


def process_single_date_benchmark(exp_date: str, lookback_days: int, gpd_method: str, delta_x: float) -> tuple[dict | None, str]:
    obs_date = (pd.to_datetime(exp_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        call_iv, put_iv, call_price, put_price, df_idx = main_module.read_data(exp_date)
        if obs_date not in call_price.index or obs_date not in put_price.index or obs_date not in call_iv.index or obs_date not in put_iv.index:
            return None, f"跳過：觀察日 {obs_date} 的資料不完整（{exp_date}）"

        df_options_mix, basicinfo = main_module.mix_cp_function(obs_date, exp_date, call_iv, put_iv, call_price, put_price, df_idx)
        smooth_iv = main_module.UnivariateSpline_function(df_options_mix, basicinfo, power=4)
        fit = main_module.RND_function(smooth_iv, df_options_mix)

        if gpd_method == "1pt":
            fit, _, _ = fit_gpd_tails_use_slope_and_cdf_with_one_point_optimized(
                fit, main_module.initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
            )
        else:
            fit, _, _ = main_module.fit_gpd_tails_use_pdf_with_two_points(
                fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
            )

        stats_result = main_module.calculate_rnd_statistics(fit, delta_x)
        result = {
            "Observation Date": obs_date,
            "Expiration Date": exp_date,
            "Mean": stats_result["mean"],
            "Std": stats_result["std"],
            "Skewness": stats_result["skewness"],
            "Kurtosis": stats_result["kurtosis"],
            "5% Quantile": stats_result["quantiles"][0.05],
            "25% Quantile": stats_result["quantiles"][0.25],
            "Median": stats_result["quantiles"][0.5],
            "75% Quantile": stats_result["quantiles"][0.75],
            "95% Quantile": stats_result["quantiles"][0.95],
        }
        return result, f"成功處理：觀察日 {obs_date}，到期日 {exp_date}"
    except Exception as exc:  # noqa: BLE001
        return None, f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}，錯誤：{exc}"


def run_single_method(expiration_dates: list[str], lookback_days: int, gpd_method: str) -> tuple[float, int]:
    start = time.perf_counter()
    success_count = 0
    for exp_date in expiration_dates:
        result, message = process_single_date_benchmark(exp_date, lookback_days, gpd_method, main_module.delta_x)
        if result is None:
            raise RuntimeError(f"{gpd_method} failed for {exp_date}: {message}")
        success_count += 1
    elapsed = time.perf_counter() - start
    return elapsed, success_count


def ordinal_label(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_markdown_table(
    execution_timestamp: str,
    sample_size: int,
    lookback_days: int,
    pairs: pd.DataFrame,
    runs_1pt: list[float],
    runs_2pt: list[float],
    ttest_result: dict[str, float | str | bool],
) -> str:
    min_obs = pairs["Observation Date"].min()
    max_obs = pairs["Observation Date"].max()
    min_exp = pairs["Expiration Date"].min()
    max_exp = pairs["Expiration Date"].max()

    lines = [
        "# Comparison of Computational Efficiency for Bitcoin Option GPD Tail Fitting",
        "",
        "(Left: 1pt method; Right: 2pt method)",
        "",
        f"- Execution Time: {execution_timestamp}",
        f"- Execution Conditions: Each run generates {sample_size} weekly return RNDs with GPD tail fitting.",
        (
            f"- Selected Sample: {sample_size} observation/expiration pairs from "
            f"`RND_stats_1pt_7d_20260312.csv`, observation dates {min_obs} to {max_obs}, "
            f"expiration dates {min_exp} to {max_exp}."
        ),
        f"- Lookback Days: {lookback_days}",
        "",
        "| Item | 1pt method (sec) | 2pt method (sec) |",
        "|---|---:|---:|",
    ]

    for idx, (left, right) in enumerate(zip(runs_1pt, runs_2pt), start=1):
        lines.append(f"| {ordinal_label(idx)} Execution Time | {left:.2f} | {right:.2f} |")

    lines.extend(
        [
            f"| Shortest Execution Time | {min(runs_1pt):.2f} | {min(runs_2pt):.2f} |",
            f"| Longest Execution Time | {max(runs_1pt):.2f} | {max(runs_2pt):.2f} |",
            f"| Average Execution Time | {sum(runs_1pt) / len(runs_1pt):.2f} | {sum(runs_2pt) / len(runs_2pt):.2f} |",
        ]
    )
    lines.extend(
        [
            "",
            "## One-sided paired t-test",
            "",
            "- Null hypothesis H0: mean paired runtime difference (1pt - 2pt) >= 0.",
            "- Alternative hypothesis H1: mean paired runtime difference (1pt - 2pt) < 0.",
            f"- t statistic: {ttest_result['t_statistic']:.6f}",
            f"- p value (one-sided): {ttest_result['p_value_one_sided']:.6f}",
            f"- Mean difference (1pt - 2pt): {ttest_result['mean_difference_seconds']:.6f} sec",
            f"- Significant at alpha=0.05: {'Yes' if ttest_result['significant_at_0_05'] else 'No'}",
            f"- Conclusion: {ttest_result['conclusion']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_runtime_ttest(runs_df: pd.DataFrame) -> dict[str, float | str | bool]:
    paired = runs_df.pivot(index="repeat", columns="method", values="elapsed_seconds").sort_index()
    runs_1pt = paired["1pt"].to_numpy()
    runs_2pt = paired["2pt"].to_numpy()

    t_statistic, p_value_two_sided = stats.ttest_rel(runs_1pt, runs_2pt)

    mean_difference = float(runs_1pt.mean() - runs_2pt.mean())
    if t_statistic < 0:
        p_value_one_sided = float(p_value_two_sided / 2)
    else:
        p_value_one_sided = float(1 - (p_value_two_sided / 2))

    significant = p_value_one_sided < 0.05
    if significant:
        conclusion = "Reject H0. The paired results show that the 1pt runtime is significantly lower than the 2pt runtime."
    else:
        conclusion = "Fail to reject H0. The paired results do not provide enough evidence that the 1pt runtime is lower than the 2pt runtime."

    return {
        "t_statistic": float(t_statistic),
        "p_value_two_sided": float(p_value_two_sided),
        "p_value_one_sided": p_value_one_sided,
        "mean_difference_seconds": mean_difference,
        "significant_at_0_05": significant,
        "test_type": "paired_t_test",
        "conclusion": conclusion,
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(
        input_csv=input_csv,
        sample_size=args.sample_size,
        lookback_days=args.lookback_days,
        selection=args.selection,
        seed=args.seed,
    )
    pairs_path = output_dir / "selected_pairs.csv"
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    expiration_dates = pairs["Expiration Date"].tolist()

    if args.warmup:
        for method in ("1pt", "2pt"):
            _, _ = run_single_method(expiration_dates[:1], args.lookback_days, method)

    run_records: list[dict] = []
    for repeat in range(1, args.repeats + 1):
        for method in ("1pt", "2pt"):
            elapsed, success_count = run_single_method(expiration_dates, args.lookback_days, method)
            run_records.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "sample_size": args.sample_size,
                    "success_count": success_count,
                    "elapsed_seconds": elapsed,
                }
            )
            print(f"repeat={repeat}, method={method}, elapsed={elapsed:.2f}s, success_count={success_count}")

    runs_df = pd.DataFrame(run_records)
    runs_path = output_dir / "benchmark_runs.csv"
    runs_df.to_csv(runs_path, index=False, encoding="utf-8-sig")

    summary = (
        runs_df.groupby("method")["elapsed_seconds"]
        .agg(["min", "max", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "min": "shortest_seconds",
                "max": "longest_seconds",
                "mean": "average_seconds",
                "median": "median_seconds",
                "std": "std_seconds",
            }
        )
    )
    ttest_result = run_runtime_ttest(runs_df)
    ttest_df = pd.DataFrame([ttest_result])
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ttest_path = output_dir / "benchmark_ttest.csv"
    ttest_df.to_csv(ttest_path, index=False, encoding="utf-8-sig")

    pivot = runs_df.pivot(index="repeat", columns="method", values="elapsed_seconds").reset_index()
    wide_path = output_dir / "benchmark_wide.csv"
    pivot.to_csv(wide_path, index=False, encoding="utf-8-sig")

    execution_timestamp = datetime.now().strftime("%Y/%m/%d %H:%M")
    markdown = build_markdown_table(
        execution_timestamp=execution_timestamp,
        sample_size=args.sample_size,
        lookback_days=args.lookback_days,
        pairs=pairs,
        runs_1pt=runs_df.loc[runs_df["method"] == "1pt", "elapsed_seconds"].tolist(),
        runs_2pt=runs_df.loc[runs_df["method"] == "2pt", "elapsed_seconds"].tolist(),
        ttest_result=ttest_result,
    )
    markdown_path = output_dir / "benchmark_table.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"selected pairs -> {pairs_path}")
    print(f"run details -> {runs_path}")
    print(f"summary -> {summary_path}")
    print(f"t-test -> {ttest_path}")
    print(f"wide table -> {wide_path}")
    print(f"markdown table -> {markdown_path}")


if __name__ == "__main__":
    main()
