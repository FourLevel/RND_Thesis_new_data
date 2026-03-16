from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import requests
import seaborn as sns


HISTORY_URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
UTC = timezone.utc
INSTRUMENT_PATTERN = re.compile(
    r"^(?P<asset>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<option_code>[CP])$"
)


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        return self.start.strftime("%Y-%m")


CSV_READ_KWARGS = {
    "low_memory": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Deribit BTC option trades, build analysis tables, and export volume charts."
    )
    parser.add_argument("--start-date", default="2020-01-01", help="Inclusive start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-09-01", help="Inclusive end date in YYYY-MM-DD.")
    parser.add_argument(
        "--data-dir",
        default="data/deribit_data/BTC-option",
        help="Directory for raw monthly files and merged outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/deribit_option_volume",
        help="Directory for charts and summary CSV files.",
    )
    parser.add_argument("--currency", default="BTC", help="Currency passed to Deribit. Default: BTC.")
    parser.add_argument("--force", action="store_true", help="Redownload monthly files even if they already exist.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.20,
        help="Delay between API requests to avoid hammering Deribit.",
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Skip downloading and rebuild merged tables/charts from existing monthly raw files.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rebuild summary CSV files and charts without writing merged raw/enriched trade CSV files.",
    )
    return parser.parse_args()


def ensure_utc(date_text: str, inclusive_end: bool = False) -> datetime:
    dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=UTC)
    if inclusive_end:
        dt = dt + timedelta(days=1)
    return dt


def month_windows(start: datetime, end_exclusive: datetime) -> Iterable[Window]:
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end_exclusive:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield Window(start=max(cursor, start), end=min(next_month, end_exclusive))
        cursor = next_month


def fetch_page(
    session: requests.Session,
    currency: str,
    start_ms: int,
    end_ms: int,
    sleep_seconds: float,
    max_retries: int = 5,
) -> dict:
    params = {
        "currency": currency,
        "kind": "option",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": 1000,
        "sorting": "asc",
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(HISTORY_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if "result" not in payload:
                raise RuntimeError(f"Unexpected response: {payload}")
            time.sleep(sleep_seconds)
            return payload["result"]
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise RuntimeError(
                    f"Deribit request failed for {datetime.fromtimestamp(start_ms / 1000, UTC)} "
                    f"to {datetime.fromtimestamp(end_ms / 1000, UTC)}"
                ) from exc
            wait_seconds = attempt * 2
            print(f"  retry {attempt}/{max_retries} after error: {exc}")
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable retry loop")


def download_window(
    session: requests.Session,
    currency: str,
    window: Window,
    sleep_seconds: float,
) -> pd.DataFrame:
    start_ms = int(window.start.timestamp() * 1000)
    end_ms = int(window.end.timestamp() * 1000)
    cursor_ms = start_ms
    rows: list[dict] = []
    page = 0

    while cursor_ms < end_ms:
        page += 1
        result = fetch_page(session, currency, cursor_ms, end_ms, sleep_seconds=sleep_seconds)
        trades = result.get("trades", [])
        if not trades:
            break

        rows.extend(trades)
        last_ts = int(trades[-1]["timestamp"])
        next_cursor = last_ts + 1
        if next_cursor <= cursor_ms:
            raise RuntimeError(
                f"Pagination stalled for {window.label}: cursor={cursor_ms}, last_ts={last_ts}, page={page}"
            )
        cursor_ms = next_cursor

        if page % 50 == 0:
            cursor_text = datetime.fromtimestamp(cursor_ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {window.label} page {page}: {len(rows):,} trades downloaded, cursor={cursor_text} UTC")

        if not result.get("has_more", False):
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["trade_seq", "trade_id", "timestamp"])
    df = df.sort_values(["timestamp", "trade_seq"], kind="stable").reset_index(drop=True)
    return df


def inspect_month_file(file_path: Path, window: Window) -> tuple[bool, str]:
    if not file_path.exists():
        return False, "missing"

    try:
        df = pd.read_csv(file_path, usecols=["timestamp"])
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {exc}"

    if df.empty:
        return False, "empty"

    timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    min_day = timestamps.min().date()
    max_day = timestamps.max().date()
    expected_start_day = window.start.date()
    expected_end_day = (window.end - timedelta(milliseconds=1)).date()

    if min_day > expected_start_day or max_day < expected_end_day:
        return False, f"partial coverage: {min_day} to {max_day}"

    return True, f"complete: {min_day} to {max_day}"


def load_existing_monthly_files(start: datetime, end_exclusive: datetime, currency: str, data_dir: Path) -> pd.DataFrame:
    raw_dir = data_dir / "raw_monthly"
    monthly_frames: list[pd.DataFrame] = []

    for window in month_windows(start, end_exclusive):
        file_path = raw_dir / f"{currency}_option_trades_{window.label}.csv"
        is_complete, message = inspect_month_file(file_path, window)
        if not is_complete:
            raise FileNotFoundError(
                f"Monthly file {file_path} is not ready for rebuild-only mode ({message})."
            )

        print(f"Rebuilding from {file_path.name} ({message})")
        month_df = pd.read_csv(file_path, **CSV_READ_KWARGS)
        if not month_df.empty:
            monthly_frames.append(month_df)

    if not monthly_frames:
        raise RuntimeError("No monthly raw files were available for rebuild-only mode.")

    all_trades = pd.concat(monthly_frames, ignore_index=True)
    all_trades = all_trades.drop_duplicates(subset=["trade_seq", "trade_id", "timestamp"])
    all_trades = all_trades.sort_values(["timestamp", "trade_seq"], kind="stable").reset_index(drop=True)
    return all_trades


def save_monthly_files(
    start: datetime,
    end_exclusive: datetime,
    currency: str,
    data_dir: Path,
    force: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    raw_dir = data_dir / "raw_monthly"
    raw_dir.mkdir(parents=True, exist_ok=True)

    monthly_frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for window in month_windows(start, end_exclusive):
            file_path = raw_dir / f"{currency}_option_trades_{window.label}.csv"
            is_complete, message = inspect_month_file(file_path, window)
            if is_complete and not force:
                print(f"Reusing {file_path.name} ({message})")
                month_df = pd.read_csv(file_path, **CSV_READ_KWARGS)
            else:
                if file_path.exists() and not force:
                    print(f"Refreshing {file_path.name} ({message})")
                print(f"Downloading {window.label} ({window.start.date()} to {(window.end - timedelta(days=1)).date()})")
                month_df = download_window(session, currency, window, sleep_seconds=sleep_seconds)
                month_df.to_csv(file_path, index=False)
                print(f"  saved {file_path.name} with {len(month_df):,} rows")

            if not month_df.empty:
                monthly_frames.append(month_df)

    if not monthly_frames:
        raise RuntimeError("No trades were downloaded. Please check the date range or API connectivity.")

    all_trades = pd.concat(monthly_frames, ignore_index=True)
    all_trades = all_trades.drop_duplicates(subset=["trade_seq", "trade_id", "timestamp"])
    all_trades = all_trades.sort_values(["timestamp", "trade_seq"], kind="stable").reset_index(drop=True)
    return all_trades


def enrich_trades(raw_trades: pd.DataFrame) -> pd.DataFrame:
    df = raw_trades.copy()
    parsed = df["instrument_name"].str.extract(INSTRUMENT_PATTERN)
    if parsed.isna().any(axis=None):
        missing = df.loc[parsed.isna().any(axis=1), "instrument_name"].drop_duplicates().tolist()[:10]
        raise ValueError(f"Could not parse instrument_name for sample values: {missing}")

    df["asset"] = parsed["asset"]
    df["expiration_date"] = parsed["expiry"]
    df["expiration_dt"] = pd.to_datetime(parsed["expiry"], format="%d%b%y", utc=True)
    df["strike"] = parsed["strike"].astype(float)
    df["type"] = parsed["option_code"].map({"C": "Call", "P": "Put"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["observe_month"] = df["timestamp"].dt.strftime("%Y-%m")
    df["days"] = (df["expiration_dt"].dt.tz_localize(None) - df["timestamp"].dt.tz_localize(None)).dt.days
    df["T"] = df["days"] / 365
    df["days_category"] = pd.cut(
        df["days"],
        bins=[-float("inf"), 14, 30, 90, 180, float("inf")],
        labels=["<=14 days", "15-30", "31-90", "90-180", ">180"],
        right=False,
    )
    df["moneyness"] = df["strike"] / df["index_price"]
    df["volume"] = df["amount"] * df["index_price"]

    call_mask = df["type"] == "Call"
    put_mask = df["type"] == "Put"

    call_categories = pd.cut(
        df.loc[call_mask, "moneyness"],
        bins=[-float("inf"), 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, float("inf")],
        labels=[
            "<=0.9",
            "0.9-1.0",
            "1.0-1.1",
            "1.1-1.2",
            "1.2-1.3",
            "1.3-1.4",
            "1.4-1.5",
            "1.5-1.6",
            "1.6-1.7",
            "1.7-1.8",
            "1.8-1.9",
            "1.9-2.0",
            ">2.0",
        ],
        right=False,
    )
    put_categories = pd.cut(
        df.loc[put_mask, "moneyness"],
        bins=[-float("inf"), 0.7, 0.8, 0.9, 1, 1.1, float("inf")],
        labels=["<=0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "1.0-1.1", ">1.1"],
        right=False,
    )
    df["moneyness_category"] = None
    df.loc[call_mask, "moneyness_category"] = call_categories.astype("string")
    df.loc[put_mask, "moneyness_category"] = put_categories.astype("string")

    return df


def export_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    trades_by_month = (
        df.groupby(["type", "observe_month"], observed=True)
        .size()
        .rename("trades")
        .reset_index()
    )
    volume_by_month = (
        df.groupby(["type", "observe_month"], observed=True)["volume"]
        .sum()
        .reset_index()
    )
    trades_by_month.to_csv(output_dir / "trades_by_month.csv", index=False)
    volume_by_month.to_csv(output_dir / "volume_by_month.csv", index=False)


def plot_grouped_bar(
    summary: pd.DataFrame,
    value_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    pivot = (
        summary.pivot(index="observe_month", columns="type", values=value_column)
        .fillna(0)
        .sort_index()
    )
    colors = {"Call": "lightcoral", "Put": "skyblue"}
    ax = pivot.plot(kind="bar", figsize=(14, 6), color=[colors.get(col, "grey") for col in pivot.columns])
    ax.set_xlabel("Observe Month")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if value_column == "volume":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    call_volume = (
        df.loc[df["type"] == "Call"]
        .groupby(["moneyness_category", "days_category"], observed=True)["volume"]
        .sum()
        .unstack(fill_value=0)
    )
    put_volume = (
        df.loc[df["type"] == "Put"]
        .groupby(["moneyness_category", "days_category"], observed=True)["volume"]
        .sum()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(call_volume / 1_000_000, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "Volume (USD mn)"}, ax=ax)
    ax.set_title("Call Volume (USD millions)")
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Moneyness (K/S)")
    plt.tight_layout()
    plt.savefig(output_dir / "call_volume_heatmap.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(put_volume / 1_000_000, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Volume (USD mn)"}, ax=ax)
    ax.set_title("Put Volume (USD millions)")
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Moneyness (K/S)")
    plt.tight_layout()
    plt.savefig(output_dir / "put_volume_heatmap.png", dpi=180)
    plt.close()


def write_csv_safely(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path.resolve()), "w", encoding="utf-8", newline="") as handle:
        df.to_csv(handle, index=False, chunksize=200_000)


def main() -> None:
    args = parse_args()
    if args.rebuild_only and args.plots_only:
        raise ValueError("--rebuild-only and --plots-only cannot be used together")

    start = ensure_utc(args.start_date)
    end_exclusive = ensure_utc(args.end_date, inclusive_end=True)

    if start >= end_exclusive:
        raise ValueError("start-date must be earlier than or equal to end-date")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_only or args.plots_only:
        raw_trades = load_existing_monthly_files(
            start=start,
            end_exclusive=end_exclusive,
            currency=args.currency,
            data_dir=data_dir,
        )
    else:
        raw_trades = save_monthly_files(
            start=start,
            end_exclusive=end_exclusive,
            currency=args.currency,
            data_dir=data_dir,
            force=args.force,
            sleep_seconds=args.sleep_seconds,
        )
    option_trades = enrich_trades(raw_trades)
    if not args.plots_only:
        raw_path = data_dir / f"{args.currency}_option_trades_raw_{args.start_date}_{args.end_date}.csv"
        write_csv_safely(raw_trades, raw_path)
        print(f"Saved merged raw trades to {raw_path}")

        enriched_path = data_dir / f"{args.currency}_option_all_{args.start_date}_{args.end_date}.csv"
        write_csv_safely(option_trades, enriched_path)
        print(f"Saved enriched trades to {enriched_path}")
    else:
        print("Skipped merged CSV outputs because --plots-only was set")

    export_summary_tables(option_trades, output_dir)
    trades_by_month = pd.read_csv(output_dir / "trades_by_month.csv")
    volume_by_month = pd.read_csv(output_dir / "volume_by_month.csv")
    plot_grouped_bar(
        trades_by_month,
        value_column="trades",
        y_label="Number of Trades",
        title=f"{args.currency} Option Trades by Type and Month",
        output_path=output_dir / "trades_by_month.png",
    )
    plot_grouped_bar(
        volume_by_month,
        value_column="volume",
        y_label="Volume (USD)",
        title=f"{args.currency} Option Volume by Type and Month",
        output_path=output_dir / "volume_by_month.png",
    )
    plot_heatmaps(option_trades, output_dir)

    print(f"Charts and summaries saved to {output_dir}")


if __name__ == "__main__":
    main()
