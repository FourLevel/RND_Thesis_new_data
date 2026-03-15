import requests
import pandas as pd
from datetime import datetime

# 獲取所有可用數據（limit=0）
url = "https://api.alternative.me/fng/?limit=0"
response = requests.get(url)
data = response.json()

# 將數據轉換為 DataFrame
df = pd.DataFrame(data['data'])

# 將 timestamp 轉換為日期時間
df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')

# 將 value 轉換為數值型態
df['value'] = pd.to_numeric(df['value'])

# 篩選 2020-2026 的數據
mask = (df['date'] >= '2020-01-01') & (df['date'] <= '2026-01-01')
df_filtered = df.loc[mask]

# 按日期排序
df_filtered = df_filtered.sort_values('date')

# 重設索引
df_filtered = df_filtered.reset_index(drop=True)

# 顯示前五筆資料
print("前五筆資料：")
print(df_filtered.head())

# 顯示基本統計資訊
print("\n基本統計資訊：")
print(df_filtered['value'].describe())

# 儲存為 CSV 檔案
output_file = 'data/my_data/Crypto_Fear_and_Greed_Index.csv'
df_filtered.to_csv(output_file, index=False)
print(f"\n數據已儲存至：{output_file}")