"""
Deribit BTC 每日收盤價下載器

使用 Deribit 公開 API (public/get_tradingview_chart_data)
下載 BTC-PERPETUAL 永續合約的每日 OHLCV 資料。
此為公開端點，無需認證。
"""

import requests
import pandas as pd
from datetime import datetime, timezone


def date_to_ms_timestamp(date_str: str) -> int:
    """將日期字串 (YYYY-MM-DD) 轉換為毫秒級 UNIX 時間戳 (UTC)。"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_deribit_ohlcv(
    instrument_name: str,
    start_date: str,
    end_date: str,
    resolution: str = "1D",
) -> pd.DataFrame:
    """
    從 Deribit API 下載 OHLCV 資料。

    Parameters
    ----------
    instrument_name : str
        合約名稱，例如 "BTC-PERPETUAL"。
    start_date : str
        起始日期，格式 "YYYY-MM-DD"。
    end_date : str
        結束日期，格式 "YYYY-MM-DD"。
    resolution : str
        K 線週期，預設 "1D"（日線）。
        支援：1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D

    Returns
    -------
    pd.DataFrame
        包含 date, open, high, low, close, volume 欄位的 DataFrame。
    """
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

    params = {
        "instrument_name": instrument_name,
        "start_timestamp": date_to_ms_timestamp(start_date),
        "end_timestamp": date_to_ms_timestamp(end_date),
        "resolution": resolution,
    }

    print(f"正在從 Deribit 下載資料...")
    print(f"  合約: {instrument_name}")
    print(f"  期間: {start_date} ~ {end_date}")
    print(f"  週期: {resolution}")
    print()

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    # 檢查回傳狀態
    result = data.get("result", {})
    status = result.get("status", "")

    if status != "ok":
        raise ValueError(f"API 回傳狀態異常: {status}，完整回應: {data}")

    # 解析 OHLCV 資料
    df = pd.DataFrame(
        {
            "timestamp_ms": result["ticks"],
            "open": result["open"],
            "high": result["high"],
            "low": result["low"],
            "close": result["close"],
            "volume": result["volume"],
        }
    )

    # 將毫秒時間戳轉為日期
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]]

    return df


def main():
    # 設定參數
    instrument = "BTC-PERPETUAL"
    start_date = "2020-01-01"
    end_date = "2026-01-01"

    # 下載資料
    df = fetch_deribit_ohlcv(instrument, start_date, end_date)

    # 顯示結果
    print("=" * 60)
    print(f"下載完成，共 {len(df)} 筆資料")
    print("=" * 60)
    print()
    print(df.to_string(index=False))
    print()

    # 儲存為 CSV
    output_path = "data/my_data/BTC_price/deribit_BTC_daily_price.csv"
    df.to_csv(output_path, index=False)
    print(f"資料已儲存至: {output_path}")


if __name__ == "__main__":
    main()
