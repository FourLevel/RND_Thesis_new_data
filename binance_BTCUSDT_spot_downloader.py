import requests
import pandas as pd


def get_binance_daily_close_by_date(start_date, end_date, symbol="BTCUSDT"):
    """
    從 Binance API 下載日線收盤價資料。
    自動分頁處理，突破單次 1000 筆限制。
    """
    url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # 將起始日期轉換為毫秒時間戳記
    current_start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)

    while current_start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current_start_ts,
            "endTime": end_ts,
            "limit": 1000
        }

        response = requests.get(url, params=params)
        data = response.json()

        # 錯誤處理
        if not data or (isinstance(data, dict) and "code" in data):
            print(f"獲取資料失敗：{data}")
            break

        all_data.extend(data)
        print(f"已取得 {len(all_data)} 筆資料...")

        # 如果回傳不足 1000 筆，代表已經到底
        if len(data) < 1000:
            break

        # 將下一次的起始時間設為最後一筆的下一天
        current_start_ts = data[-1][0] + 86400000  # +1 天（毫秒）

    if not all_data:
        print("未取得任何資料。")
        return None

    # 整理成 DataFrame
    df = pd.DataFrame(all_data)
    df = df.iloc[:, [0, 4]]
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.date
    df["Close"] = df["Close"].astype(float)

    print(f"共取得 {len(df)} 筆日線資料")
    return df


# 下載範圍：2020/1/1 - 2026/1/1
start_date = "2020-01-01"
end_date = "2026-01-01"
btc_data = get_binance_daily_close_by_date(start_date, end_date)

print(btc_data)
btc_data.to_csv("data/my_data/BTCUSDT_spot/BTCUSDT_spot.csv", index=False)