# 基本數據處理與分析套件
import pandas as pd
import numpy as np
import statsmodels.api as sm
# 繪圖套件
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
from matplotlib.ticker import ScalarFormatter
# 日期時間處理
from datetime import datetime, timedelta
import calendar
# 數學與統計相關套件
from scipy.optimize import bisect, minimize
from scipy.stats import norm, genextreme, genpareto as gpd
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline, InterpolatedUnivariateSpline, CubicSpline, interp1d
from scipy.integrate import quad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
# 系統與工具套件
import os
import re
import asyncio
import nest_asyncio
import warnings
# 自定義套件
from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

nest_asyncio.apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)
today = datetime.now().strftime('%Y-%m-%d')


# RND main
initial_i = 1
delta_x = 0.1 
observation_date = "2024-03-15"
expiration_date = "2024-03-22"
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
plot_implied_volatility(df_options_mix)
# smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4, s=None, w=None)
smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4, s=None, w=None)
fit = RND_function(smooth_IV)
plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date)


''' 擬合 GPD 的函數，選 1 個點，比較斜率與 CDF '''
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)
# 計算 RND 曲線統計量並繪製具有分位數的 RND 曲線
stats = calculate_rnd_statistics(fit, delta_x)
quants = list(stats['quantiles'].values())
plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date)
print(f"  平均值     Mean: {stats['mean']:.4f}")
print(f"  標準差      Std: {stats['std']:.4f}")
print(f"    偏度 Skewness: {stats['skewness']:.4f}")
print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
print()


''' 擬合 GPD 的函數，選 2 個點，比較 PDF '''
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)
# 計算 RND 曲線統計量並繪製具有分位數的 RND 曲線
stats = calculate_rnd_statistics(fit, delta_x)
quants = list(stats['quantiles'].values())
plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date)
print(f"  平均值     Mean: {stats['mean']:.4f}")
print(f"  標準差      Std: {stats['std']:.4f}")
print(f"    偏度 Skewness: {stats['skewness']:.4f}")
print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
print()


''' 於同一張圖繪製多條 RND 曲線，自訂日期 '''
observation_dates = ['2022-09-06', '2022-10-10','2022-11-09', '2022-12-09', '2023-01-09', '2023-02-09', '2023-03-09']
expiration_date = '2023-03-31'
all_stats, all_rnd_data = process_multiple_dates_one_point(observation_dates, expiration_date)
plot_multiple_rnd(all_rnd_data, observation_dates, expiration_date)

# 印出每個日期的 mean, std, skewness, kurtosis
print("每個日期的統計數據：")
for date in observation_dates:
    stats = all_stats[date]
    print(f"{date}:")
    print(f"  平均值     Mean: {stats['mean']:.4f}")
    print(f"  標準差      Std: {stats['std']:.4f}")
    print(f"    偏度 Skewness: {stats['skewness']:.4f}")
    print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
    print()


''' 於同一張圖繪製多條 RND 曲線，僅需輸入起始日和最終日 '''
# 輸入起始日和最終日
start_date = '2021-04-14'
end_date = '2021-06-09'
expiration_date = '2021-06-25'

# 生成日期列表
observation_dates = generate_dates(start_date, end_date, interval_days=7) # interval_days 可設定間隔天數

# 處理數據並繪圖
all_stats, all_rnd_data = process_multiple_dates_two_points(observation_dates, expiration_date) # 使用不同方法可調整函數
plot_multiple_rnd(all_rnd_data, observation_dates, expiration_date)

# 印出每個日期的統計數據
print("每個日期的統計數據：")
for date in observation_dates:
    if date in all_stats:
        stats = all_stats[date]
        print(f"{date}:")
        print(f"  平均值     Mean: {stats['mean']:.4f}")
        print(f"  標準差      Std: {stats['std']:.4f}")
        print(f"    偏度 Skewness: {stats['skewness']:.4f}")
        print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
        print()
    else:
        print(f"{date}: 無可用數據")
        print()


''' 整理統計數據 '''
# 整理統計數據
stats_data = []
for date in observation_dates:
    if date in all_stats:
        stats = all_stats[date]
        stats_data.append({
            '日期': date,
            '平均值 Mean': f"{stats['mean']:.4f}",
            '標準差 Std': f"{stats['std']:.4f}",
            '偏度 Skewness': f"{stats['skewness']:.4f}",
            '峰度 Kurtosis': f"{stats['kurtosis']:.4f}"
        })
    else:
        stats_data.append({
            '日期': date,
            '平均值 Mean': 'N/A',
            '標準差 Std': 'N/A',
            '偏度 Skewness': 'N/A',
            '峰度 Kurtosis': 'N/A'
        })

# 創建 DataFrame
df_stats = pd.DataFrame(stats_data)

# 匯出成 CSV 檔
csv_filename = f'RND_stats_{start_date}_to_{end_date}_exp_{expiration_date}.csv'
df_stats.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"統計數據已匯出至 {csv_filename}")

# 印出每個日期的統計數據
print("每個日期的統計數據：")
print(df_stats.to_string(index=False))


''' 回推買權價格 '''   
# 計算單個買權價格
strike_price = 40000  # 假設行權價為 100
call_option_price = calculate_call_option_price_discrete(fit, strike_price)
print(f"買權價格 Call Option Price: {call_option_price:.4f}")

# 計算所有大於 future_price 的行權價的買權價格，每隔 1000 個計算一次
future_price = F  # 設定 future_price
call_option_prices, x_values, strike_prices, selected_strike_prices = calculate_call_option_prices_above_future_price(fit, future_price, step=1000)

for strike_price, call_price in call_option_prices.items():
    print(f"Strike Price: {strike_price:.2f} 的買權價格: {call_price:.4f}")

# 繪製圖表
plt.figure(figsize=(10, 6), dpi=100)
# 擬合的買權價格
plt.plot(fit['strike_price'], fit['fit_call'], color='orange', label='Fitted Call Price')
# 計算的買權價格
strike_prices = list(call_option_prices.keys())
call_prices = list(call_option_prices.values())
plt.plot(strike_prices, call_prices, linestyle='-', color='blue', label='Calculated Call Price')
plt.title(f'Call Option Prices for Strike Prices Above Future Price (Future Price: {F:.2f})')
plt.xlabel('Strike Price')
plt.ylabel('Call Option Price')
# plt.xlim(30000, 100000)
# plt.ylim(0, 5000)
plt.legend()
plt.grid(True)
plt.show()


''' 回推買權價格 '''
# 輸入起始日和最終日
try:
    start_date = '2021-04-14'
    end_date = '2021-06-09'
    expiration_date = '2021-06-25'

    # 生成日期列表
    observation_dates = generate_dates(start_date, end_date, interval_days=7)

    # 處理數據
    # all_stats, all_rnd_data, all_call_option_prices = find_call_option_prices_above_future_price_multiple_dates_two_points(observation_dates, expiration_date)
    all_stats, all_rnd_data, all_call_option_prices = find_call_option_prices_above_future_price_multiple_dates_one_point(observation_dates, expiration_date)

    # 創建空的 DataFrame 來存儲結果
    df_call_option_prices = pd.DataFrame()

    # 遍歷每個日期的數據
    for date in observation_dates:
        if date in all_call_option_prices:
            call_prices = all_call_option_prices[date]
            if isinstance(call_prices, tuple) and len(call_prices) > 0:
                prices_dict = call_prices[0]  # 取得第一個元素(字典)
                if prices_dict:  # 確認字典不為空
                    df = pd.DataFrame([prices_dict], index=[date])
                    df_call_option_prices = pd.concat([df_call_option_prices, df])
                    print(f"處理日期 {date} 的數據成功")
                else:
                    print(f"日期 {date} 的價格字典為空")
            else:
                print(f"日期 {date} 的數據格式不正確")
        else:
            print(f"找不到日期 {date} 的數據")

    # 打印最終結果
    if not df_call_option_prices.empty:
        print("\n最終的期權價格DataFrame:")
        print(df_call_option_prices)
    else:
        print("\n沒有成功處理任何數據")

except Exception as e:
    print(f"執行過程中發生錯誤: {str(e)}")

# 找尋 call_price 變數中，index 與 df_call_option_prices 的 index 相同的值，並比對 column name，向下增加 row
for date in df_call_option_prices.index:
    if date in call_price.index:
        # 將 call_price 中的值添加到 df_call_option_prices 中的新行
        new_row = call_price.loc[date].reindex(df_call_option_prices.columns)
        df_call_option_prices = pd.concat([df_call_option_prices, pd.DataFrame(new_row).T], ignore_index=False)

# 將 df_call_option_prices 中有 missing value 的 column 刪除
df_call_option_prices = df_call_option_prices.dropna(axis=1)

# 打印更新後的 DataFrame
print("更新後的 df_call_option_prices：")
print(df_call_option_prices)

# 如果需要，可以將 DataFrame 匯出為 CSV 檔案
df_call_option_prices.to_csv('call_option_prices.csv', index=True, encoding='utf-8')


''' 迴歸分析資料整理_每天_一個點方法 '''
# 讀取 instruments.csv
instruments = pd.read_csv('deribit_data/instruments.csv')

# 建立 DataFrame 存放迴歸資料
df_regression_day = pd.DataFrame()

# 選擇日期，將 type 為 day, week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = instruments[instruments['type'].isin(['day', 'week', 'quarter', 'year'])]['date'].unique()

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates)

# 計算 observation_dates，將 expiration_dates 前一日設定為 observation_dates，存入 observation_dates
observation_dates = expiration_dates - pd.Timedelta(days=1)

# 將結果轉回字串格式
observation_dates = observation_dates.strftime('%Y-%m-%d')
expiration_dates = expiration_dates.strftime('%Y-%m-%d')

# 將 expiration_dates 和 observation_dates 設定為 DataFrame 的欄位
df_regression_day['observation_dates'] = observation_dates
df_regression_day['expiration_dates'] = expiration_dates

# 建立儲存統計資料的 list
stats_data = []

# 對每一組日期進行計算
for obs_date, exp_date in zip(df_regression_day['observation_dates'], df_regression_day['expiration_dates']):
    try:
        # 設定全域變數
        global observation_date, expiration_date
        observation_date = obs_date
        expiration_date = exp_date
        
        # 讀取資料並進行 RND 計算
        call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
        F = find_F2()
        get_FTS()
        df_options_mix = mix_cp_function_v2()
        smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
        fit = RND_function(smooth_IV)
        
        # GPD 尾端擬合
        fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(
            fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
        )
        
        # 計算統計量
        stats = calculate_rnd_statistics(fit, delta_x)
        
        # 整理統計資料
        stats_data.append({
            'Observation Date': obs_date,
            'Expiration Date': exp_date,
            'Mean': stats['mean'],
            'Std': stats['std'],
            'Skewness': stats['skewness'],
            'Kurtosis': stats['kurtosis'],
            '5% Quantile': stats['quantiles'][0.05],
            '25% Quantile': stats['quantiles'][0.25],
            'Median': stats['quantiles'][0.5],
            '75% Quantile': stats['quantiles'][0.75],
            '95% Quantile': stats['quantiles'][0.95]
        })
        
        print(f"成功處理：觀察日 {obs_date}，到期日 {exp_date}")
        
    except Exception as e:
        print(f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}")
        print(f"錯誤訊息：{str(e)}")
        continue

# 將統計資料轉換為 DataFrame
df_regression_day_stats = pd.DataFrame(stats_data)

# 將結果儲存為 CSV
output_filename = f'RND_regression_day_stats_一個點_{today}.csv'
df_regression_day_stats.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n統計資料已儲存至 {output_filename}")

# 顯示統計摘要
print("\n統計資料摘要：")
print(df_regression_day_stats.describe())

# 讀取 BTC 價格資料
df_btcusdt = pd.read_csv('binance_data/BTCUSDT_spot.csv')

# 讀取當中 date 與 close 欄位
df_btcusdt_close = df_btcusdt[['date', 'close']]

# 將 df_btcusdt_close 的日期欄位轉換為 datetime 格式
df_btcusdt_close['date'] = pd.to_datetime(df_btcusdt_close['date'])

# 將 df_regression_day_stats 的日期欄位轉換為 datetime 格式
df_regression_day_stats['Observation Date'] = pd.to_datetime(df_regression_day_stats['Observation Date'])
df_regression_day_stats['Expiration Date'] = pd.to_datetime(df_regression_day_stats['Expiration Date'])

# 將 df_btcusdt_close 設定 date 為索引
df_btcusdt_close.set_index('date', inplace=True)

# 顯示可用的日期範圍
print("價格數據的日期範圍：")
print(f"起始日期：{df_btcusdt_close.index.min()}")
print(f"結束日期：{df_btcusdt_close.index.max()}")

# 篩選出在價格數據日期範圍內的觀察資料
mask = (df_regression_day_stats['Observation Date'] >= df_btcusdt_close.index.min()) & \
       (df_regression_day_stats['Expiration Date'] <= df_btcusdt_close.index.max())
df_regression_day_stats_filtered = df_regression_day_stats[mask].copy()

# 計算當期對數報酬率
df_regression_day_stats_filtered['T Return'] = np.log(
    df_btcusdt_close.loc[df_regression_day_stats_filtered['Expiration Date']]['close'].values / 
    df_btcusdt_close.loc[df_regression_day_stats_filtered['Observation Date']]['close'].values
)

# 計算前期對數報酬率
# 先將資料按觀察日排序
df_regression_day_stats_filtered = df_regression_day_stats_filtered.sort_values('Observation Date')
# 使用 shift 函數來獲取前期的報酬率
df_regression_day_stats_filtered['T-1 Return'] = df_regression_day_stats_filtered['T Return'].shift(1)
df_regression_day_stats_filtered['T-2 Return'] = df_regression_day_stats_filtered['T Return'].shift(2)
df_regression_day_stats_filtered['T-3 Return'] = df_regression_day_stats_filtered['T Return'].shift(3)
df_regression_day_stats_filtered['T-4 Return'] = df_regression_day_stats_filtered['T Return'].shift(4)

# 去除 NaN 值
df_regression_day_stats_filtered = df_regression_day_stats_filtered.dropna()

# 顯示結果
print(f"\n符合日期範圍的資料筆數：{len(df_regression_day_stats_filtered)}")
print("\n加入對數報酬率後的資料：")
print(df_regression_day_stats_filtered)

# 將結果儲存為 CSV
output_filename = f'RND_regression_day_stats_with_returns_一個點_{today}.csv'
df_regression_day_stats_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")

# 讀取資料
# 檔名日期需自行更改
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_with_returns_一個點.csv')
df_fear_greed_index = pd.read_csv('Crypto Fear and Greed Index_2020-2024.csv')
df_vix = pd.read_csv('CBOE VIX_2020-2024.csv')

# 將所有 DataFrame 的日期欄位都轉換為 datetime 格式
df_regression_day_stats_with_returns['Observation Date'] = pd.to_datetime(df_regression_day_stats_with_returns['Observation Date'])
df_fear_greed_index['date'] = pd.to_datetime(df_fear_greed_index['date'])
df_vix['日期'] = pd.to_datetime(df_vix['日期'])

# 將 df_fear_greed_index 和 df_vix 的日期欄位設為索引
df_fear_greed_index.set_index('date', inplace=True)
df_vix.set_index('日期', inplace=True)

# 使用 merge 來匹配日期，先合併貪婪指數
df_regression_day_stats_with_returns = pd.merge(
    df_regression_day_stats_with_returns,
    df_fear_greed_index[['value']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 再合併 VIX 指數
df_regression_day_stats_with_returns = pd.merge(
    df_regression_day_stats_with_returns,
    df_vix[['收市']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 重命名欄位
df_regression_day_stats_with_returns.rename(columns={
    'value': 'Fear and Greed Index',
    '收市': 'VIX'
}, inplace=True)

# 檢查合併前的缺失值數量
print("填補前的缺失值數量：")
missing_values_fear_greed = df_regression_day_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_day_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 使用前後最近的值填補 VIX 的空值
df_regression_day_stats_with_returns['VIX'] = df_regression_day_stats_with_returns['VIX'].fillna(method='ffill').fillna(method='bfill')

# 檢查填補後的缺失值數量
print("\n填補後的缺失值數量：")
missing_values_fear_greed = df_regression_day_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_day_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 將結果儲存為 CSV
output_filename = f'RND_regression_day_stats_all_data_一個點_{today}.csv'
df_regression_day_stats_with_returns.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")


''' 執行迴歸分析_每天_一個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('刪6個極端值\RND_regression_day_stats_all_data_一個點_2025-04-28_刪6個極端值.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_day_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv(f'descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std

# 單因子迴歸分析
# 建立一個 DataFrame 來儲存迴歸結果
univariate_regression_results = pd.DataFrame(columns=['Variable', 'Coefficient', 'p value', 'Significance', 'R-squared', 'MSE'])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行簡單迴歸
for var in variables_to_standardize:
    if var != 'T Return':  # 排除 Y 變數
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[var]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷顯著水準
        p_value = model.pvalues[1]
        if p_value < 0.01:
            significance = '***'
        elif p_value < 0.05:
            significance = '**'
        elif p_value < 0.1:
            significance = '*'
        else:
            significance = ''

        # 儲存結果
        univariate_regression_results = pd.concat([univariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Coefficient': [model.params[1]],  # 係數
            'p value': [model.pvalues[1]],    # p value
            'Significance': [significance],   # 顯著水準
            'R-squared': [model.rsquared],     # R-squared
            'MSE': [mse]                      # MSE
        })], ignore_index=True)

# 顯示結果
print("\n簡單迴歸分析結果：")
print(univariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
univariate_regression_results.to_csv('univariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n簡單迴歸分析結果已儲存至 univariate regression results.csv")

# 二因子迴歸分析（固定 Skewness）
# 建立一個 DataFrame 來儲存迴歸結果
bivariate_regression_results = pd.DataFrame(columns=[
    'Variable', 
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行二因子迴歸（與 Skewness 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness']:  # 排除 Y 變數和 Skewness
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Skewness', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷另一個變數的顯著水準
        p_value_var = model.pvalues[2]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        bivariate_regression_results = pd.concat([bivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Variable_Coef': [model.params[2]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n二因子迴歸分析結果（固定 Skewness）：")
print(bivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
bivariate_regression_results.to_csv('bivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n二因子迴歸分析結果已儲存至 bivariate regression results.csv")

# 三因子迴歸分析（固定 Skewness 和 Kurtosis）
# 建立一個 DataFrame 來儲存迴歸結果
trivariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行三因子迴歸（與 Skewness, Kurtosis 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis']:  # 排除 Y 變數、Skewness 和 Kurtosis
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Skewness', 'Kurtosis', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷第三個變數的顯著水準
        p_value_var = model.pvalues[3]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        trivariate_regression_results = pd.concat([trivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Variable_Coef': [model.params[3]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n三因子迴歸分析結果（固定 Skewness 和 Kurtosis）：")
print(trivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
trivariate_regression_results.to_csv('trivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n三因子迴歸分析結果已儲存至 trivariate regression results.csv")

# 四因子迴歸分析（固定 Skewness、Kurtosis 和 Std）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Std_Coef', 'Std_p', 'Std_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Skewness, Kurtosis, Std 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis', 'Std']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Skewness', 'Kurtosis', 'Std', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷 Std 的顯著水準
        p_value_std = model.pvalues[3]
        if p_value_std < 0.01:
            sig_std = '***'
        elif p_value_std < 0.05:
            sig_std = '**'
        elif p_value_std < 0.1:
            sig_std = '*'
        else:
            sig_std = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Std_Coef': [model.params[3]],          # Std 係數
            'Std_p': [p_value_std],                 # Std p-value
            'Std_Sig': [sig_std],                   # Std 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Skewness、Kurtosis 和 Std）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

'''
# 針對所有變數進行 ADF 檢定
variables = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Fear and Greed Index',
            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

print("ADF 檢定結果：")
print("-" * 50)
for var in variables:
    result = adfuller(df_regression_week_stats_with_returns[var].dropna())
    print(f"\n變數：{var}")
    print(f"ADF 統計量：{result[0]:.4f}")
    print(f"p-value：{result[1]:.4f}")
    print("臨界值：")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    # 判斷是否為定態序列
    if result[1] < 0.05:
        print("結論：該序列為定態序列 (拒絕單根假設)")
    else:
        print("結論：該序列為非定態序列 (未能拒絕單根假設)")

# 儲存結果
with open(f'adf_results_day_一個點_{today}.txt', 'w', encoding='utf-8') as f:
    for var in variables:
        result = adfuller(df_regression_day_stats_with_returns[var].dropna())
        f.write(f"變數：{var}\n")
        f.write(f"p-value：{result[1]:.4f}\n")
        f.write("結論：" + ("該序列為定態序列" if result[1] < 0.05 else "該序列為非定態序列") + "\n\n")
'''

###########################################################

# 準備迴歸變數，用這個模型
X_1 = df_regression_day_stats_with_returns[[
    'Kurtosis', 'Median',
    'T-4 Return'
]]
y = df_regression_day_stats_with_returns['T Return']

# 加入常數項
X_1 = sm.add_constant(X_1)

# 執行OLS迴歸
model = sm.OLS(y, X_1).fit()

# 計算 MSE
y_pred = model.predict(X_1)
mse = np.mean((y - y_pred) ** 2)

# 印出迴歸結果
print("迴歸分析結果：")
print(model.summary())
print(f"\nMSE: {mse:.4f}")

# 建立一個 DataFrame 來儲存迴歸結果
regression_results = pd.DataFrame(columns=[
    'Variable',
    'Coefficient',
    'Std Error',
    'T-Stat',
    'P-Value',
    'Significance'
])

# 取得模型結果
variables = ['const', 'Kurtosis', 'Median', 'T-4 Return']
coefficients = model.params
std_errors = model.bse
t_stats = model.tvalues
p_values = model.pvalues

# 判斷顯著性
def get_significance(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''

# 整理結果
for var in variables:
    idx = variables.index(var)
    regression_results = pd.concat([regression_results, pd.DataFrame({
        'Variable': [var],
        'Coefficient': [coefficients[idx]],
        'Std Error': [std_errors[idx]],
        'T-Stat': [t_stats[idx]],
        'P-Value': [p_values[idx]],
        'Significance': [get_significance(p_values[idx])]
    })], ignore_index=True)

# 加入模型整體統計量
model_stats = pd.DataFrame({
    'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
    'Value': [
        model.rsquared,
        model.rsquared_adj,
        model.fvalue,
        model.f_pvalue,
        model.nobs,
        mse
    ]
})

# 顯示結果
print("\n迴歸係數及顯著性：")
print(regression_results.round(4))
print("\n模型統計量：")
print(model_stats.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
regression_results.to_csv('regression_results.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸結果已儲存至 regression_results.csv")
model_stats.to_csv('model_stats.csv', index=False, encoding='utf-8-sig')
print(f"\n模型統計量已儲存至 model_stats.csv")

# 基於這個模型的四因子迴歸分析（固定 Kurtosis、Median 和 T-4 Return）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Median_Coef', 'Median_p', 'Median_Sig',
    'T-4 Return_Coef', 'T-4 Return_p', 'T-4 Return_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Kurtosis, Median, T-4 Return 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Kurtosis', 'Median', 'T-4 Return']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Kurtosis', 'Median', 'T-4 Return', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        p_value_kurtosis = model.pvalues[1]
        if p_value_kurtosis < 0.01:
            sig_kurtosis = '***'
        elif p_value_kurtosis < 0.05:
            sig_kurtosis = '**'
        elif p_value_kurtosis < 0.1:
            sig_kurtosis = '*'
        else:
            sig_kurtosis = ''
            
        # 判斷 Median 的顯著水準
        p_value_median = model.pvalues[2]
        if p_value_median < 0.01:
            sig_median = '***'
        elif p_value_median < 0.05:
            sig_median = '**'
        elif p_value_median < 0.1:
            sig_median = '*'
        else:
            sig_median = ''
            
        # 判斷 T-4 Return 的顯著水準
        p_value_t4 = model.pvalues[3]
        if p_value_t4 < 0.01:
            sig_t4 = '***'
        elif p_value_t4 < 0.05:
            sig_t4 = '**'
        elif p_value_t4 < 0.1:
            sig_t4 = '*'
        else:
            sig_t4 = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Kurtosis_Coef': [model.params[1]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurtosis],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurtosis],             # Kurtosis 顯著性
            'Median_Coef': [model.params[2]],     # Median 係數
            'Median_p': [p_value_median],           # Median p-value
            'Median_Sig': [sig_median],             # Median 顯著性
            'T-4 Return_Coef': [model.params[3]],          # T-4 Return 係數
            'T-4 Return_p': [p_value_t4],                 # T-4 Return p-value
            'T-4 Return_Sig': [sig_t4],                   # T-4 Return 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Kurtosis、Median 和 T-4 Return）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

###########################################################

# 計算每個變數的 correlation matrix，並繪製 heatmap
X = df_regression_day_stats_with_returns[[
    'Mean', 'Std', 'Skewness', 'Kurtosis', 'Fear and Greed Index',
    'Median', '5% Quantile', '25% Quantile', '75% Quantile', '95% Quantile',
    'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return'
]]
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8), dpi=200)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 計算每個變數的VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_1.values, i) for i in range(X_1.shape[1])]

print("\n變異數膨脹因子(VIF)：")
print(vif_data)

# 儲存VIF結果
vif_data.to_csv(f'vif_results_day_一個點_{today}.csv', index=False, encoding='utf-8-sig')


''' 迴歸分析資料整理_每天_兩個點方法 '''
# 讀取 instruments.csv
instruments = pd.read_csv('deribit_data/instruments.csv')

# 建立 DataFrame 存放迴歸資料
df_regression_day = pd.DataFrame()

# 選擇日期，將 type 為 day, week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = instruments[instruments['type'].isin(['day', 'week', 'quarter', 'year'])]['date'].unique()

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates)

# 計算 observation_dates，將 expiration_dates 前一日設定為 observation_dates，存入 observation_dates
observation_dates = expiration_dates - pd.Timedelta(days=1)

# 將結果轉回字串格式
observation_dates = observation_dates.strftime('%Y-%m-%d')
expiration_dates = expiration_dates.strftime('%Y-%m-%d')

# 將 expiration_dates 和 observation_dates 設定為 DataFrame 的欄位
df_regression_day['observation_dates'] = observation_dates
df_regression_day['expiration_dates'] = expiration_dates

# 建立儲存統計資料的 list
stats_data = []

# 對每一組日期進行計算
for obs_date, exp_date in zip(df_regression_day['observation_dates'], df_regression_day['expiration_dates']):
    try:
        # 設定全域變數
        global observation_date, expiration_date
        observation_date = obs_date
        expiration_date = exp_date
        
        # 讀取資料並進行 RND 計算
        call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
        F = find_F2()
        get_FTS()
        df_options_mix = mix_cp_function_v2()
        smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
        fit = RND_function(smooth_IV)
        
        # GPD 尾端擬合
        fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(
            fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
        )
        
        # 計算統計量
        stats = calculate_rnd_statistics(fit, delta_x)
        
        # 整理統計資料
        stats_data.append({
            'Observation Date': obs_date,
            'Expiration Date': exp_date,
            'Mean': stats['mean'],
            'Std': stats['std'],
            'Skewness': stats['skewness'],
            'Kurtosis': stats['kurtosis'],
            '5% Quantile': stats['quantiles'][0.05],
            '25% Quantile': stats['quantiles'][0.25],
            'Median': stats['quantiles'][0.5],
            '75% Quantile': stats['quantiles'][0.75],
            '95% Quantile': stats['quantiles'][0.95]
        })
        
        print(f"成功處理：觀察日 {obs_date}，到期日 {exp_date}")
        
    except Exception as e:
        print(f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}")
        print(f"錯誤訊息：{str(e)}")
        continue

# 將統計資料轉換為 DataFrame
df_regression_day_stats = pd.DataFrame(stats_data)

# 將結果儲存為 CSV
output_filename = f'RND_regression_day_stats_兩個點_{today}.csv'
df_regression_day_stats.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n統計資料已儲存至 {output_filename}")

# 顯示統計摘要
print("\n統計資料摘要：")
print(df_regression_day_stats.describe())

# 讀取 BTC 價格資料
df_btcusdt = pd.read_csv('binance_data/BTCUSDT_spot.csv')

# 讀取當中 date 與 close 欄位
df_btcusdt_close = df_btcusdt[['date', 'close']]

# 將 df_btcusdt_close 的日期欄位轉換為 datetime 格式
df_btcusdt_close['date'] = pd.to_datetime(df_btcusdt_close['date'])

# 將 df_regression_day_stats 的日期欄位轉換為 datetime 格式
df_regression_day_stats['Observation Date'] = pd.to_datetime(df_regression_day_stats['Observation Date'])
df_regression_day_stats['Expiration Date'] = pd.to_datetime(df_regression_day_stats['Expiration Date'])

# 將 df_btcusdt_close 設定 date 為索引
df_btcusdt_close.set_index('date', inplace=True)

# 顯示可用的日期範圍
print("價格數據的日期範圍：")
print(f"起始日期：{df_btcusdt_close.index.min()}")
print(f"結束日期：{df_btcusdt_close.index.max()}")

# 篩選出在價格數據日期範圍內的觀察資料
mask = (df_regression_day_stats['Observation Date'] >= df_btcusdt_close.index.min()) & \
       (df_regression_day_stats['Expiration Date'] <= df_btcusdt_close.index.max())
df_regression_day_stats_filtered = df_regression_day_stats[mask].copy()

# 計算當期對數報酬率
df_regression_day_stats_filtered['T Return'] = np.log(
    df_btcusdt_close.loc[df_regression_day_stats_filtered['Expiration Date']]['close'].values / 
    df_btcusdt_close.loc[df_regression_day_stats_filtered['Observation Date']]['close'].values
)

# 計算前期對數報酬率
# 先將資料按觀察日排序
df_regression_day_stats_filtered = df_regression_day_stats_filtered.sort_values('Observation Date')
# 使用 shift 函數來獲取前期的報酬率
df_regression_day_stats_filtered['T-1 Return'] = df_regression_day_stats_filtered['T Return'].shift(1)
df_regression_day_stats_filtered['T-2 Return'] = df_regression_day_stats_filtered['T Return'].shift(2)
df_regression_day_stats_filtered['T-3 Return'] = df_regression_day_stats_filtered['T Return'].shift(3)
df_regression_day_stats_filtered['T-4 Return'] = df_regression_day_stats_filtered['T Return'].shift(4)

# 去除 NaN 值
df_regression_day_stats_filtered = df_regression_day_stats_filtered.dropna()

# 顯示結果
print(f"\n符合日期範圍的資料筆數：{len(df_regression_day_stats_filtered)}")
print("\n加入對數報酬率後的資料：")
print(df_regression_day_stats_filtered)

# 將結果儲存為 CSV
output_filename = f'RND_regression_day_stats_with_returns_兩個點_{today}.csv'
df_regression_day_stats_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")

# 讀取資料
# 檔名日期需自行更改
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_with_returns_兩個點.csv')
df_fear_greed_index = pd.read_csv('Crypto Fear and Greed Index_2020-2024.csv')
df_vix = pd.read_csv('CBOE VIX_2020-2024.csv')

# 將所有 DataFrame 的日期欄位都轉換為 datetime 格式
df_regression_day_stats_with_returns['Observation Date'] = pd.to_datetime(df_regression_day_stats_with_returns['Observation Date'])
df_fear_greed_index['date'] = pd.to_datetime(df_fear_greed_index['date'])
df_vix['日期'] = pd.to_datetime(df_vix['日期'])

# 將 df_fear_greed_index 和 df_vix 的日期欄位設為索引
df_fear_greed_index.set_index('date', inplace=True)
df_vix.set_index('日期', inplace=True)

# 使用 merge 來匹配日期，先合併貪婪指數
df_regression_day_stats_with_returns = pd.merge(
    df_regression_day_stats_with_returns,
    df_fear_greed_index[['value']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 再合併 VIX 指數
df_regression_day_stats_with_returns = pd.merge(
    df_regression_day_stats_with_returns,
    df_vix[['收市']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 重命名欄位
df_regression_day_stats_with_returns.rename(columns={
    'value': 'Fear and Greed Index',
    '收市': 'VIX'
}, inplace=True)

# 檢查合併前的缺失值數量
print("填補前的缺失值數量：")
missing_values_fear_greed = df_regression_day_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_day_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 使用前後最近的值填補 VIX 的空值
df_regression_day_stats_with_returns['VIX'] = df_regression_day_stats_with_returns['VIX'].fillna(method='ffill').fillna(method='bfill')

# 檢查填補後的缺失值數量
print("\n填補後的缺失值數量：")
missing_values_fear_greed = df_regression_day_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_day_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 將結果儲存為 CSV
output_filename = f'RND_regression_day_stats_all_data_兩個點_{today}.csv'
df_regression_day_stats_with_returns.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")


''' 執行迴歸分析_每天_兩個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('刪6個極端值/RND_regression_day_stats_all_data_兩個點_2025-04-28_刪6個極端值.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_day_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv('descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std

# 單因子迴歸分析
# 建立一個 DataFrame 來儲存迴歸結果
univariate_regression_results = pd.DataFrame(columns=['Variable', 'Coefficient', 'p value','Significance', 'R-squared', 'MSE'])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行簡單迴歸
for var in variables_to_standardize:
    if var != 'T Return':  # 排除 Y 變數
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[var]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷顯著水準
        p_value = model.pvalues[1]
        if p_value < 0.01:
            significance = '***'
        elif p_value < 0.05:
            significance = '**'
        elif p_value < 0.1:
            significance = '*'
        else:
            significance = ''

        # 儲存結果
        univariate_regression_results = pd.concat([univariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Coefficient': [model.params[1]],  # 係數
            'p value': [model.pvalues[1]],    # p value
            'Significance': [significance],   # 顯著水準
            'R-squared': [model.rsquared],     # R-squared
            'MSE': [mse]                      # MSE
        })], ignore_index=True)

# 顯示結果
print("\n簡單迴歸分析結果：")
print(univariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
univariate_regression_results.to_csv('univariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n簡單迴歸分析結果已儲存至 univariate regression results.csv")

# 二因子迴歸分析（固定 Skewness）
# 建立一個 DataFrame 來儲存迴歸結果
bivariate_regression_results = pd.DataFrame(columns=[
    'Variable', 
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行二因子迴歸（與 Skewness 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness']:  # 排除 Y 變數和 Skewness
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Skewness', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷另一個變數的顯著水準
        p_value_var = model.pvalues[2]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        bivariate_regression_results = pd.concat([bivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Variable_Coef': [model.params[2]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n二因子迴歸分析結果（固定 Skewness）：")
print(bivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
bivariate_regression_results.to_csv('bivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n二因子迴歸分析結果已儲存至 bivariate regression results.csv")

# 三因子迴歸分析（固定 Skewness 和 Kurtosis）
# 建立一個 DataFrame 來儲存迴歸結果
trivariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行三因子迴歸（與 Skewness, Kurtosis 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis']:  # 排除 Y 變數、Skewness 和 Kurtosis
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Skewness', 'Kurtosis', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷第三個變數的顯著水準
        p_value_var = model.pvalues[3]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        trivariate_regression_results = pd.concat([trivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Variable_Coef': [model.params[3]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n三因子迴歸分析結果（固定 Skewness 和 Kurtosis）：")
print(trivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
trivariate_regression_results.to_csv('trivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n三因子迴歸分析結果已儲存至 trivariate regression results.csv")

# 四因子迴歸分析（固定 Skewness、Kurtosis 和 Std）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Std_Coef', 'Std_p', 'Std_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Skewness, Kurtosis, Std 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis', 'Std']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Skewness', 'Kurtosis', 'Std', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷 Std 的顯著水準
        p_value_std = model.pvalues[3]
        if p_value_std < 0.01:
            sig_std = '***'
        elif p_value_std < 0.05:
            sig_std = '**'
        elif p_value_std < 0.1:
            sig_std = '*'
        else:
            sig_std = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Std_Coef': [model.params[3]],          # Std 係數
            'Std_p': [p_value_std],                 # Std p-value
            'Std_Sig': [sig_std],                   # Std 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Skewness、Kurtosis 和 Std）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

'''
# 針對所有變數進行 ADF 檢定
variables = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Fear and Greed Index',
            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

print("ADF 檢定結果：")
print("-" * 50)
for var in variables:
    result = adfuller(df_regression_week_stats_with_returns[var].dropna())
    print(f"\n變數：{var}")
    print(f"ADF 統計量：{result[0]:.4f}")
    print(f"p-value：{result[1]:.4f}")
    print("臨界值：")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    # 判斷是否為定態序列
    if result[1] < 0.05:
        print("結論：該序列為定態序列 (拒絕單根假設)")
    else:
        print("結論：該序列為非定態序列 (未能拒絕單根假設)")

# 儲存結果
with open(f'adf_results_day_兩個點_{today}.txt', 'w', encoding='utf-8') as f:
    for var in variables:
        result = adfuller(df_regression_day_stats_with_returns[var].dropna())
        f.write(f"變數：{var}\n")
        f.write(f"p-value：{result[1]:.4f}\n")
        f.write("結論：" + ("該序列為定態序列" if result[1] < 0.05 else "該序列為非定態序列") + "\n\n")
'''

###########################################################

# 準備迴歸變數，用這個模型
X_1 = df_regression_day_stats_with_returns[[
    'Kurtosis', 'Median',
    'T-4 Return'
]]
y = df_regression_day_stats_with_returns['T Return']

# 加入常數項
X_1 = sm.add_constant(X_1)

# 執行OLS迴歸
model = sm.OLS(y, X_1).fit()

# 計算 MSE
y_pred = model.predict(X_1)
mse = np.mean((y - y_pred) ** 2)

# 印出迴歸結果
print("迴歸分析結果：")
print(model.summary())
print(f"\nMSE: {mse:.4f}")

# 建立一個 DataFrame 來儲存迴歸結果
regression_results = pd.DataFrame(columns=[
    'Variable',
    'Coefficient',
    'Std Error',
    'T-Stat',
    'P-Value',
    'Significance'
])

# 取得模型結果
variables = ['const', 'Kurtosis', 'Median', 'T-4 Return']
coefficients = model.params
std_errors = model.bse
t_stats = model.tvalues
p_values = model.pvalues

# 判斷顯著性
def get_significance(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''

# 整理結果
for var in variables:
    idx = variables.index(var)
    regression_results = pd.concat([regression_results, pd.DataFrame({
        'Variable': [var],
        'Coefficient': [coefficients[idx]],
        'Std Error': [std_errors[idx]],
        'T-Stat': [t_stats[idx]],
        'P-Value': [p_values[idx]],
        'Significance': [get_significance(p_values[idx])]
    })], ignore_index=True)

# 加入模型整體統計量
model_stats = pd.DataFrame({
    'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
    'Value': [
        model.rsquared,
        model.rsquared_adj,
        model.fvalue,
        model.f_pvalue,
        model.nobs,
        mse
    ]
})

# 顯示結果
print("\n迴歸係數及顯著性：")
print(regression_results.round(4))
print("\n模型統計量：")
print(model_stats.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 儲存結果
regression_results.to_csv('regression_model_coefficients.csv', index=False, encoding='utf-8-sig')
model_stats.to_csv('regression_model_statistics.csv', index=False, encoding='utf-8-sig')

# 基於這個模型的四因子迴歸分析（固定 Kurtosis、Median 和 T-4 Return）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Median_Coef', 'Median_p', 'Median_Sig',
    'T-4 Return_Coef', 'T-4 Return_p', 'T-4 Return_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Kurtosis, Median, T-4 Return 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Kurtosis', 'Median', 'T-4 Return']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_day_stats_with_returns[['Kurtosis', 'Median', 'T-4 Return', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        p_value_kurtosis = model.pvalues[1]
        if p_value_kurtosis < 0.01:
            sig_kurtosis = '***'
        elif p_value_kurtosis < 0.05:
            sig_kurtosis = '**'
        elif p_value_kurtosis < 0.1:
            sig_kurtosis = '*'
        else:
            sig_kurtosis = ''
            
        # 判斷 Median 的顯著水準
        p_value_median = model.pvalues[2]
        if p_value_median < 0.01:
            sig_median = '***'
        elif p_value_median < 0.05:
            sig_median = '**'
        elif p_value_median < 0.1:
            sig_median = '*'
        else:
            sig_median = ''
            
        # 判斷 T-4 Return 的顯著水準
        p_value_t4 = model.pvalues[3]
        if p_value_t4 < 0.01:
            sig_t4 = '***'
        elif p_value_t4 < 0.05:
            sig_t4 = '**'
        elif p_value_t4 < 0.1:
            sig_t4 = '*'
        else:
            sig_t4 = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Kurtosis_Coef': [model.params[1]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurtosis],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurtosis],             # Kurtosis 顯著性
            'Median_Coef': [model.params[2]],     # Median 係數
            'Median_p': [p_value_median],           # Median p-value
            'Median_Sig': [sig_median],             # Median 顯著性
            'T-4 Return_Coef': [model.params[3]],          # T-4 Return 係數
            'T-4 Return_p': [p_value_t4],                 # T-4 Return p-value
            'T-4 Return_Sig': [sig_t4],                   # T-4 Return 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Kurtosis、Median 和 T-4 Return）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

###########################################################

# 儲存迴歸結果
with open(f'regression_results_day_兩個點_{today}.txt', 'w', encoding='utf-8') as f:
    f.write(model.summary().as_text())

# 計算每個變數的VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_1.values, i) for i in range(X_1.shape[1])]

print("\n變異數膨脹因子(VIF)：")
print(vif_data)

# 儲存VIF結果
vif_data.to_csv(f'vif_results_day_兩個點_{today}.csv', index=False, encoding='utf-8-sig')


''' 迴歸分析資料整理_每週_一個點方法 '''
# 讀取 instruments.csv
instruments = pd.read_csv('deribit_data/instruments.csv')

# 建立 DataFrame 存放迴歸資料
df_regression_week = pd.DataFrame()

# 選擇日期，將 type 為 week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = instruments[instruments['type'].isin(['week', 'quarter', 'year'])]['date'].unique()

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates)

# 計算 observation_dates，將 expiration_dates 前一日設定為 observation_dates，存入 observation_dates
observation_dates = expiration_dates - pd.Timedelta(days=7)

# 將結果轉回字串格式
observation_dates = observation_dates.strftime('%Y-%m-%d')
expiration_dates = expiration_dates.strftime('%Y-%m-%d')

# 將 expiration_dates 和 observation_dates 設定為 DataFrame 的欄位
df_regression_week['observation_dates'] = observation_dates
df_regression_week['expiration_dates'] = expiration_dates

# 建立儲存統計資料的 list
stats_data = []

# 對每一組日期進行計算
for obs_date, exp_date in zip(df_regression_week['observation_dates'], df_regression_week['expiration_dates']):
    try:
        # 設定全域變數
        global observation_date, expiration_date
        observation_date = obs_date
        expiration_date = exp_date
        
        # 讀取資料並進行 RND 計算
        call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
        F = find_F2()
        get_FTS()
        df_options_mix = mix_cp_function_v2()
        smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
        fit = RND_function(smooth_IV)
        
        # GPD 尾端擬合
        fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(
            fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
        )
        
        # 計算統計量
        stats = calculate_rnd_statistics(fit, delta_x)
        
        # 整理統計資料
        stats_data.append({
            'Observation Date': obs_date,
            'Expiration Date': exp_date,
            'Mean': stats['mean'],
            'Std': stats['std'],
            'Skewness': stats['skewness'],
            'Kurtosis': stats['kurtosis'],
            '5% Quantile': stats['quantiles'][0.05],
            '25% Quantile': stats['quantiles'][0.25],
            'Median': stats['quantiles'][0.5],
            '75% Quantile': stats['quantiles'][0.75],
            '95% Quantile': stats['quantiles'][0.95]
        })
        
        print(f"成功處理：觀察日 {obs_date}，到期日 {exp_date}")
        
    except Exception as e:
        print(f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}")
        print(f"錯誤訊息：{str(e)}")
        continue

# 將統計資料轉換為 DataFrame
df_regression_week_stats = pd.DataFrame(stats_data)

# 將結果儲存為 CSV
output_filename = f'RND_regression_week_stats_一個點_{today}.csv'
df_regression_week_stats.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n統計資料已儲存至 {output_filename}")

# 顯示統計摘要
print("\n統計資料摘要：")
print(df_regression_week_stats.describe())

# 讀取 BTC 價格資料
df_btcusdt = pd.read_csv('binance_data/BTCUSDT_spot.csv')

# 讀取當中 date 與 close 欄位
df_btcusdt_close = df_btcusdt[['date', 'close']]

# 將 df_btcusdt_close 的日期欄位轉換為 datetime 格式
df_btcusdt_close['date'] = pd.to_datetime(df_btcusdt_close['date'])

# 將 df_regression_day_stats 的日期欄位轉換為 datetime 格式
df_regression_week_stats['Observation Date'] = pd.to_datetime(df_regression_week_stats['Observation Date'])
df_regression_week_stats['Expiration Date'] = pd.to_datetime(df_regression_week_stats['Expiration Date'])

# 將 df_btcusdt_close 設定 date 為索引
df_btcusdt_close.set_index('date', inplace=True)

# 顯示可用的日期範圍
print("價格數據的日期範圍：")
print(f"起始日期：{df_btcusdt_close.index.min()}")
print(f"結束日期：{df_btcusdt_close.index.max()}")

# 篩選出在價格數據日期範圍內的觀察資料
mask = (df_regression_week_stats['Observation Date'] >= df_btcusdt_close.index.min()) & \
       (df_regression_week_stats['Expiration Date'] <= df_btcusdt_close.index.max())
df_regression_week_stats_filtered = df_regression_week_stats[mask].copy()

# 計算當期對數報酬率
df_regression_week_stats_filtered['T Return'] = np.log(
    df_btcusdt_close.loc[df_regression_week_stats_filtered['Expiration Date']]['close'].values / 
    df_btcusdt_close.loc[df_regression_week_stats_filtered['Observation Date']]['close'].values
)

# 計算前期對數報酬率
# 先將資料按觀察日排序
df_regression_week_stats_filtered = df_regression_week_stats_filtered.sort_values('Observation Date')

# 使用 shift 函數來獲取前期的報酬率
df_regression_week_stats_filtered['T-1 Return'] = df_regression_week_stats_filtered['T Return'].shift(1)
df_regression_week_stats_filtered['T-2 Return'] = df_regression_week_stats_filtered['T Return'].shift(2)
df_regression_week_stats_filtered['T-3 Return'] = df_regression_week_stats_filtered['T Return'].shift(3)
df_regression_week_stats_filtered['T-4 Return'] = df_regression_week_stats_filtered['T Return'].shift(4)

# 去除 NaN 值
df_regression_week_stats_filtered = df_regression_week_stats_filtered.dropna()

# 顯示結果
print(f"\n符合日期範圍的資料筆數：{len(df_regression_week_stats_filtered)}")
print("\n加入對數報酬率後的資料：")
print(df_regression_week_stats_filtered)

# 將結果儲存為 CSV
output_filename = f'RND_regression_week_stats_with_returns_一個點_{today}.csv'
df_regression_week_stats_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")

# 讀取資料
# 檔名日期需自行更改
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_with_returns_一個點_2025-01-14.csv')
df_fear_greed_index = pd.read_csv('Crypto Fear and Greed Index_2020-2024.csv')
df_vix = pd.read_csv('CBOE VIX_2020-2024.csv')

# 將所有 DataFrame 的日期欄位都轉換為 datetime 格式
df_regression_week_stats_with_returns['Observation Date'] = pd.to_datetime(df_regression_week_stats_with_returns['Observation Date'])
df_fear_greed_index['date'] = pd.to_datetime(df_fear_greed_index['date'])
df_vix['日期'] = pd.to_datetime(df_vix['日期'])

# 將 df_fear_greed_index 和 df_vix 的日期欄位設為索引
df_fear_greed_index.set_index('date', inplace=True)
df_vix.set_index('日期', inplace=True)

# 使用 merge 來匹配日期，先合併貪婪指數
df_regression_week_stats_with_returns = pd.merge(
    df_regression_week_stats_with_returns,
    df_fear_greed_index[['value']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 再合併 VIX 指數
df_regression_week_stats_with_returns = pd.merge(
    df_regression_week_stats_with_returns,
    df_vix[['收市']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 重命名欄位
df_regression_week_stats_with_returns.rename(columns={
    'value': 'Fear and Greed Index',
    '收市': 'VIX'
}, inplace=True)

# 檢查合併前的缺失值數量
print("填補前的缺失值數量：")
missing_values_fear_greed = df_regression_week_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_week_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 使用前後最近的值填補 VIX 的空值
df_regression_week_stats_with_returns['VIX'] = df_regression_week_stats_with_returns['VIX'].fillna(method='ffill').fillna(method='bfill')

# 檢查填補後的缺失值數量
print("\n填補後的缺失值數量：")
missing_values_fear_greed = df_regression_week_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_week_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 將結果儲存為 CSV
output_filename = f'RND_regression_week_stats_all_data_一個點_{today}.csv'
df_regression_week_stats_with_returns.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")


''' 執行迴歸分析_每週_一個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_一個點_2025-02-02.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_week_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv('descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std

# 單因子迴歸分析
# 建立一個 DataFrame 來儲存迴歸結果
univariate_regression_results = pd.DataFrame(columns=['Variable', 'Coefficient', 'p value','Significance', 'R-squared', 'MSE'])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行簡單迴歸
for var in variables_to_standardize:
    if var != 'T Return':  # 排除 Y 變數
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[var]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷顯著水準
        p_value = model.pvalues[1]
        if p_value < 0.01:
            significance = '***'
        elif p_value < 0.05:
            significance = '**'
        elif p_value < 0.1:
            significance = '*'
        else:
            significance = ''

        # 儲存結果
        univariate_regression_results = pd.concat([univariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Coefficient': [model.params[1]],  # 係數
            'p value': [model.pvalues[1]],    # p value
            'Significance': [significance],   # 顯著水準
            'R-squared': [model.rsquared],    # R-squared
            'MSE': [mse]                      # MSE
        })], ignore_index=True)

# 顯示結果
print("\n簡單迴歸分析結果：")
print(univariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
univariate_regression_results.to_csv('univariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n簡單迴歸分析結果已儲存至 univariate regression results.csv")

# 二因子迴歸分析（固定 Skewness）
# 建立一個 DataFrame 來儲存迴歸結果
bivariate_regression_results = pd.DataFrame(columns=[
    'Variable', 
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行二因子迴歸（與 Skewness 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness']:  # 排除 Y 變數和 Skewness
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Skewness', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷另一個變數的顯著水準
        p_value_var = model.pvalues[2]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        bivariate_regression_results = pd.concat([bivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Variable_Coef': [model.params[2]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n二因子迴歸分析結果（固定 Skewness）：")
print(bivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
bivariate_regression_results.to_csv('bivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n二因子迴歸分析結果已儲存至 bivariate regression results.csv")

# 三因子迴歸分析（固定 Skewness 和 Kurtosis）
# 建立一個 DataFrame 來儲存迴歸結果
trivariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行三因子迴歸（與 Skewness, Kurtosis 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis']:  # 排除 Y 變數、Skewness 和 Kurtosis
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Skewness', 'Kurtosis', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷第三個變數的顯著水準
        p_value_var = model.pvalues[3]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        trivariate_regression_results = pd.concat([trivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Variable_Coef': [model.params[3]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n三因子迴歸分析結果（固定 Skewness 和 Kurtosis）：")
print(trivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
trivariate_regression_results.to_csv('trivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n三因子迴歸分析結果已儲存至 trivariate regression results.csv")

# 四因子迴歸分析（固定 Skewness、Kurtosis 和 Std）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Std_Coef', 'Std_p', 'Std_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Skewness, Kurtosis, Std 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis', 'Std']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Skewness', 'Kurtosis', 'Std', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷 Std 的顯著水準
        p_value_std = model.pvalues[3]
        if p_value_std < 0.01:
            sig_std = '***'
        elif p_value_std < 0.05:
            sig_std = '**'
        elif p_value_std < 0.1:
            sig_std = '*'
        else:
            sig_std = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Std_Coef': [model.params[3]],          # Std 係數
            'Std_p': [p_value_std],                 # Std p-value
            'Std_Sig': [sig_std],                   # Std 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Skewness、Kurtosis 和 Std）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

'''
# 針對 T Return, Skewness, Kurtosis 欄位，將小於 -3 與大於 3 的資料列刪除
# df_regression_week_stats_with_returns = df_regression_week_stats_with_returns[(df_regression_week_stats_with_returns['T Return'] >= -2) & (df_regression_week_stats_with_returns['T Return'] <= 2)]
# df_regression_week_stats_with_returns = df_regression_week_stats_with_returns[(df_regression_week_stats_with_returns['Skewness'] >= -2) & (df_regression_week_stats_with_returns['Skewness'] <= 2)]
#df_regression_week_stats_with_returns = df_regression_week_stats_with_returns[(df_regression_week_stats_with_returns['Kurtosis'] >= -2) & (df_regression_week_stats_with_returns['Kurtosis'] <= 2)]

# 針對所有變數進行 ADF 檢定
variables = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Fear and Greed Index',
            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

print("ADF 檢定結果：")
print("-" * 50)
for var in variables:
    result = adfuller(df_regression_week_stats_with_returns[var].dropna())
    print(f"\n變數：{var}")
    print(f"ADF 統計量：{result[0]:.4f}")
    print(f"p-value：{result[1]:.4f}")
    print("臨界值：")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    # 判斷是否為定態序列
    if result[1] < 0.05:
        print("結論：該序列為定態序列 (拒絕單根假設)")
    else:
        print("結論：該序列為非定態序列 (未能拒絕單根假設)")

# 儲存結果
with open(f'adf_results_week_一個點_{today}.txt', 'w', encoding='utf-8') as f:
    for var in variables:
        result = adfuller(df_regression_week_stats_with_returns[var].dropna())
        f.write(f"變數：{var}\n")
        f.write(f"p-value：{result[1]:.4f}\n")
        f.write("結論：" + ("該序列為定態序列" if result[1] < 0.05 else "該序列為非定態序列") + "\n\n")
'''

###########################################################

# 準備迴歸變數，用這個模型
X_4 = df_regression_week_stats_with_returns[[
     'Skewness', 'Median', 'Fear and Greed Index',
]]
y = df_regression_week_stats_with_returns['T Return']

# 加入常數項
X_4 = sm.add_constant(X_4)

# 執行OLS迴歸
model = sm.OLS(y, X_4).fit()

# 計算 MSE
y_pred = model.predict(X_4)
mse = np.mean((y - y_pred) ** 2)

# 印出迴歸結果
print("迴歸分析結果：")
print(model.summary())
print(f"\nMSE: {mse:.4f}")

# 建立一個 DataFrame 來儲存迴歸結果
regression_results = pd.DataFrame(columns=[
    'Variable',
    'Coefficient',
    'Std Error',
    'T-Stat',
    'P-Value',
    'Significance'
])

# 取得模型結果
variables = ['const', 'Kurtosis', 'Median', 'Fear and Greed Index']
coefficients = model.params
std_errors = model.bse
t_stats = model.tvalues
p_values = model.pvalues

# 判斷顯著性
def get_significance(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''

# 整理結果
for var in variables:
    idx = variables.index(var)
    regression_results = pd.concat([regression_results, pd.DataFrame({
        'Variable': [var],
        'Coefficient': [coefficients[idx]],
        'Std Error': [std_errors[idx]],
        'T-Stat': [t_stats[idx]],
        'P-Value': [p_values[idx]],
        'Significance': [get_significance(p_values[idx])]
    })], ignore_index=True)

# 加入模型整體統計量
model_stats = pd.DataFrame({
    'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
    'Value': [
        model.rsquared,
        model.rsquared_adj,
        model.fvalue,
        model.f_pvalue,
        model.nobs,
        mse
    ]
})

# 顯示結果
print("\n迴歸係數及顯著性：")
print(regression_results.round(4))
print("\n模型統計量：")
print(model_stats.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 儲存結果
regression_results.to_csv('regression_model_coefficients.csv', index=False, encoding='utf-8-sig')
model_stats.to_csv('regression_model_statistics.csv', index=False, encoding='utf-8-sig')


# 基於這個模型的四因子迴歸分析（固定 Kurtosis、Median 和 Fear and Greed Index）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Median_Coef', 'Median_p', 'Median_Sig',
    'Fear and Greed Index_Coef', 'Fear and Greed Index_p', 'Fear and Greed Index_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Kurtosis, Median, Fear and Greed Index 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Kurtosis', 'Median', 'Fear and Greed Index']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Kurtosis', 'Median', 'Fear and Greed Index', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[1]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷 Median 的顯著水準
        p_value_median = model.pvalues[2]
        if p_value_median < 0.01:
            sig_median = '***'
        elif p_value_median < 0.05:
            sig_median = '**'
        elif p_value_median < 0.1:
            sig_median = '*'
        else:
            sig_median = ''
            
        # 判斷 Fear and Greed Index 的顯著水準
        p_value_fgi = model.pvalues[3]
        if p_value_fgi < 0.01:
            sig_fgi = '***'
        elif p_value_fgi < 0.05:
            sig_fgi = '**'
        elif p_value_fgi < 0.1:
            sig_fgi = '*'
        else:
            sig_fgi = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Kurtosis_Coef': [model.params[1]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Median_Coef': [model.params[2]],     # Median 係數
            'Median_p': [p_value_median],           # Median p-value
            'Median_Sig': [sig_median],             # Median 顯著性
            'Fear and Greed Index_Coef': [model.params[3]],          # Fear and Greed Index 係數
            'Fear and Greed Index_p': [p_value_fgi],                 # Fear and Greed Index p-value
            'Fear and Greed Index_Sig': [sig_fgi],                   # Fear and Greed Index 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                            # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Kurtosis、Median 和 Fear and Greed Index）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

###########################################################

# 儲存迴歸結果
with open(f'regression_results_week_一個點_{today}.txt', 'w', encoding='utf-8') as f:
    f.write(model.summary().as_text())

# 計算每個變數的VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_1.values, i) for i in range(X_1.shape[1])]

print("\n變異數膨脹因子(VIF)：")
print(vif_data)

# 儲存VIF結果
vif_data.to_csv(f'vif_results_week_一個點_{today}.csv', index=False, encoding='utf-8-sig')


''' 迴歸分析資料整理_每週_兩個點方法 '''
# 讀取 instruments.csv
instruments = pd.read_csv('deribit_data/instruments.csv')

# 建立 DataFrame 存放迴歸資料
df_regression_week = pd.DataFrame()

# 選擇日期，將 type 為 week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = instruments[instruments['type'].isin(['week', 'quarter', 'year'])]['date'].unique()

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates)

# 計算 observation_dates，將 expiration_dates 前一日設定為 observation_dates，存入 observation_dates
observation_dates = expiration_dates - pd.Timedelta(days=7)

# 將結果轉回字串格式
observation_dates = observation_dates.strftime('%Y-%m-%d')
expiration_dates = expiration_dates.strftime('%Y-%m-%d')

# 將 expiration_dates 和 observation_dates 設定為 DataFrame 的欄位
df_regression_week['observation_dates'] = observation_dates
df_regression_week['expiration_dates'] = expiration_dates

# 建立儲存統計資料的 list
stats_data = []

# 對每一組日期進行計算
for obs_date, exp_date in zip(df_regression_week['observation_dates'], df_regression_week['expiration_dates']):
    try:
        # 設定全域變數
        global observation_date, expiration_date
        observation_date = obs_date
        expiration_date = exp_date
        
        # 讀取資料並進行 RND 計算
        call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
        F = find_F2()
        get_FTS()
        df_options_mix = mix_cp_function_v2()
        smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
        fit = RND_function(smooth_IV)
        
        # GPD 尾端擬合
        fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(
            fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
        )
        
        # 計算統計量
        stats = calculate_rnd_statistics(fit, delta_x)
        
        # 整理統計資料
        stats_data.append({
            'Observation Date': obs_date,
            'Expiration Date': exp_date,
            'Mean': stats['mean'],
            'Std': stats['std'],
            'Skewness': stats['skewness'],
            'Kurtosis': stats['kurtosis'],
            '5% Quantile': stats['quantiles'][0.05],
            '25% Quantile': stats['quantiles'][0.25],
            'Median': stats['quantiles'][0.5],
            '75% Quantile': stats['quantiles'][0.75],
            '95% Quantile': stats['quantiles'][0.95]
        })
        
        print(f"成功處理：觀察日 {obs_date}，到期日 {exp_date}")
        
    except Exception as e:
        print(f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}")
        print(f"錯誤訊息：{str(e)}")
        continue

# 將統計資料轉換為 DataFrame
df_regression_week_stats = pd.DataFrame(stats_data)

# 將結果儲存為 CSV
output_filename = f'RND_regression_week_stats_兩個點_{today}.csv'
df_regression_week_stats.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n統計資料已儲存至 {output_filename}")

# 顯示統計摘要
print("\n統計資料摘要：")
print(df_regression_week_stats.describe())

# 讀取 BTC 價格資料
df_btcusdt = pd.read_csv('binance_data/BTCUSDT_spot.csv')

# 讀取當中 date 與 close 欄位
df_btcusdt_close = df_btcusdt[['date', 'close']]

# 將 df_btcusdt_close 的日期欄位轉換為 datetime 格式
df_btcusdt_close['date'] = pd.to_datetime(df_btcusdt_close['date'])

# 將 df_regression_day_stats 的日期欄位轉換為 datetime 格式
df_regression_week_stats['Observation Date'] = pd.to_datetime(df_regression_week_stats['Observation Date'])
df_regression_week_stats['Expiration Date'] = pd.to_datetime(df_regression_week_stats['Expiration Date'])

# 將 df_btcusdt_close 設定 date 為索引
df_btcusdt_close.set_index('date', inplace=True)

# 顯示可用的日期範圍
print("價格數據的日期範圍：")
print(f"起始日期：{df_btcusdt_close.index.min()}")
print(f"結束日期：{df_btcusdt_close.index.max()}")

# 篩選出在價格數據日期範圍內的觀察資料
mask = (df_regression_week_stats['Observation Date'] >= df_btcusdt_close.index.min()) & \
       (df_regression_week_stats['Expiration Date'] <= df_btcusdt_close.index.max())
df_regression_week_stats_filtered = df_regression_week_stats[mask].copy()

# 計算當期對數報酬率
df_regression_week_stats_filtered['T Return'] = np.log(
    df_btcusdt_close.loc[df_regression_week_stats_filtered['Expiration Date']]['close'].values / 
    df_btcusdt_close.loc[df_regression_week_stats_filtered['Observation Date']]['close'].values
)

# 計算前期對數報酬率
# 先將資料按觀察日排序
df_regression_week_stats_filtered = df_regression_week_stats_filtered.sort_values('Observation Date')
# 使用 shift 函數來獲取前期的報酬率
df_regression_week_stats_filtered['T-1 Return'] = df_regression_week_stats_filtered['T Return'].shift(1)
df_regression_week_stats_filtered['T-2 Return'] = df_regression_week_stats_filtered['T Return'].shift(2)
df_regression_week_stats_filtered['T-3 Return'] = df_regression_week_stats_filtered['T Return'].shift(3)
df_regression_week_stats_filtered['T-4 Return'] = df_regression_week_stats_filtered['T Return'].shift(4)

# 去除 NaN 值
df_regression_week_stats_filtered = df_regression_week_stats_filtered.dropna()

# 顯示結果
print(f"\n符合日期範圍的資料筆數：{len(df_regression_week_stats_filtered)}")
print("\n加入對數報酬率後的資料：")
print(df_regression_week_stats_filtered)

# 將結果儲存為 CSV
output_filename = f'RND_regression_week_stats_with_returns_兩個點_{today}.csv'
df_regression_week_stats_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")

# 讀取資料
# 檔名日期需自行更改
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_with_returns_兩個點_2025-01-14.csv')
df_fear_greed_index = pd.read_csv('Crypto Fear and Greed Index_2020-2024.csv')
df_vix = pd.read_csv('CBOE VIX_2020-2024.csv')

# 將所有 DataFrame 的日期欄位都轉換為 datetime 格式
df_regression_week_stats_with_returns['Observation Date'] = pd.to_datetime(df_regression_week_stats_with_returns['Observation Date'])
df_fear_greed_index['date'] = pd.to_datetime(df_fear_greed_index['date'])
df_vix['日期'] = pd.to_datetime(df_vix['日期'])

# 將 df_fear_greed_index 和 df_vix 的日期欄位設為索引
df_fear_greed_index.set_index('date', inplace=True)
df_vix.set_index('日期', inplace=True)

# 使用 merge 來匹配日期，先合併貪婪指數
df_regression_week_stats_with_returns = pd.merge(
    df_regression_week_stats_with_returns,
    df_fear_greed_index[['value']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 再合併 VIX 指數
df_regression_week_stats_with_returns = pd.merge(
    df_regression_week_stats_with_returns,
    df_vix[['收市']],
    left_on='Observation Date',
    right_index=True,
    how='left'
)

# 重命名欄位
df_regression_week_stats_with_returns.rename(columns={
    'value': 'Fear and Greed Index',
    '收市': 'VIX'
}, inplace=True)

# 檢查合併前的缺失值數量
print("填補前的缺失值數量：")
missing_values_fear_greed = df_regression_week_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_week_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 使用前後最近的值填補 VIX 的空值
df_regression_week_stats_with_returns['VIX'] = df_regression_week_stats_with_returns['VIX'].fillna(method='ffill').fillna(method='bfill')

# 檢查填補後的缺失值數量
print("\n填補後的缺失值數量：")
missing_values_fear_greed = df_regression_week_stats_with_returns['Fear and Greed Index'].isna().sum()
missing_values_vix = df_regression_week_stats_with_returns['VIX'].isna().sum()
print(f"Fear and Greed Index 中的缺失值數量：{missing_values_fear_greed}")
print(f"VIX 中的缺失值數量：{missing_values_vix}")

# 將結果儲存為 CSV
output_filename = f'RND_regression_week_stats_all_data_兩個點_{today}.csv'
df_regression_week_stats_with_returns.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n已將結果儲存至 {output_filename}")


''' 執行迴歸分析_每週_兩個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_兩個點_2025-02-02.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_week_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv(f'descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX',
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std

# 單因子迴歸分析
# 建立一個 DataFrame 來儲存迴歸結果
univariate_regression_results = pd.DataFrame(columns=['Variable', 'Coefficient', 'p value','Significance', 'R-squared', 'MSE'])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行簡單迴歸
for var in variables_to_standardize:
    if var != 'T Return':  # 排除 Y 變數
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[var]
        X = sm.add_constant(X)  # 加入常數項

        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷顯著水準
        p_value = model.pvalues[1]
        if p_value < 0.01:
            significance = '***'
        elif p_value < 0.05:
            significance = '**'
        elif p_value < 0.1:
            significance = '*'
        else:
            significance = ''

        # 儲存結果
        univariate_regression_results = pd.concat([univariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Coefficient': [model.params[1]],  # 係數
            'p value': [model.pvalues[1]],    # p value
            'Significance': [significance],   # 顯著水準
            'R-squared': [model.rsquared],    # R-squared
            'MSE': [mse]                      # MSE
        })], ignore_index=True)

# 顯示結果
print("\n簡單迴歸分析結果：")
print(univariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
univariate_regression_results.to_csv('univariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n簡單迴歸分析結果已儲存至 univariate regression results.csv")

# 二因子迴歸分析（固定 Skewness）
# 建立一個 DataFrame 來儲存迴歸結果
bivariate_regression_results = pd.DataFrame(columns=[
    'Variable', 
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行二因子迴歸（與 Skewness 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness']:  # 排除 Y 變數和 Skewness
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Skewness', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷另一個變數的顯著水準
        p_value_var = model.pvalues[2]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        bivariate_regression_results = pd.concat([bivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Variable_Coef': [model.params[2]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                             # MSE
        })], ignore_index=True)

# 顯示結果
print("\n二因子迴歸分析結果（固定 Skewness）：")
print(bivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
bivariate_regression_results.to_csv('bivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n二因子迴歸分析結果已儲存至 bivariate regression results.csv")

# 三因子迴歸分析（固定 Skewness 和 Kurtosis）
# 建立一個 DataFrame 來儲存迴歸結果
trivariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行三因子迴歸（與 Skewness, Kurtosis 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis']:  # 排除 Y 變數、Skewness 和 Kurtosis
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Skewness', 'Kurtosis', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷第三個變數的顯著水準
        p_value_var = model.pvalues[3]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        trivariate_regression_results = pd.concat([trivariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Variable_Coef': [model.params[3]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                             # MSE
        })], ignore_index=True)

# 顯示結果
print("\n三因子迴歸分析結果（固定 Skewness 和 Kurtosis）：")
print(trivariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
trivariate_regression_results.to_csv('trivariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n三因子迴歸分析結果已儲存至 trivariate regression results.csv")

# 四因子迴歸分析（固定 Skewness、Kurtosis 和 Std）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Skewness_Coef', 'Skewness_p', 'Skewness_Sig',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Std_Coef', 'Std_p', 'Std_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Skewness, Kurtosis, Std 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Skewness', 'Kurtosis', 'Std']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Skewness', 'Kurtosis', 'Std', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        p_value_skew = model.pvalues[1]
        if p_value_skew < 0.01:
            sig_skew = '***'
        elif p_value_skew < 0.05:
            sig_skew = '**'
        elif p_value_skew < 0.1:
            sig_skew = '*'
        else:
            sig_skew = ''
            
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[2]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷 Std 的顯著水準
        p_value_std = model.pvalues[3]
        if p_value_std < 0.01:
            sig_std = '***'
        elif p_value_std < 0.05:
            sig_std = '**'
        elif p_value_std < 0.1:
            sig_std = '*'
        else:
            sig_std = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Skewness_Coef': [model.params[1]],     # Skewness 係數
            'Skewness_p': [p_value_skew],           # Skewness p-value
            'Skewness_Sig': [sig_skew],             # Skewness 顯著性
            'Kurtosis_Coef': [model.params[2]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Std_Coef': [model.params[3]],          # Std 係數
            'Std_p': [p_value_std],                 # Std p-value
            'Std_Sig': [sig_std],                   # Std 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                             # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Skewness、Kurtosis 和 Std）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

'''
# 針對所有變數進行 ADF 檢定
variables = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Fear and Greed Index',
            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

print("ADF 檢定結果：")
print("-" * 50)
for var in variables:
    result = adfuller(df_regression_week_stats_with_returns[var].dropna())
    print(f"\n變數：{var}")
    print(f"ADF 統計量：{result[0]:.4f}")
    print(f"p-value：{result[1]:.4f}")
    print("臨界值：")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    # 判斷是否為定態序列
    if result[1] < 0.05:
        print("結論：該序列為定態序列 (拒絕單根假設)")
    else:
        print("結論：該序列為非定態序列 (未能拒絕單根假設)")

# 儲存結果
with open(f'adf_results_week_兩個點_{today}.txt', 'w', encoding='utf-8') as f:
    for var in variables:
        result = adfuller(df_regression_week_stats_with_returns[var].dropna())
        f.write(f"變數：{var}\n")
        f.write(f"p-value：{result[1]:.4f}\n")
        f.write("結論：" + ("該序列為定態序列" if result[1] < 0.05 else "該序列為非定態序列") + "\n\n")
'''

###########################################################

# 準備迴歸變數，用這個模型
X_4 = df_regression_week_stats_with_returns[[
     'Skewness', 'Median', 'Fear and Greed Index',
]]
y = df_regression_week_stats_with_returns['T Return']

# 加入常數項
X_4 = sm.add_constant(X_4)

# 執行OLS迴歸
model = sm.OLS(y, X_4).fit()

# 計算 MSE
y_pred = model.predict(X_4)
mse = np.mean((y - y_pred) ** 2)

# 印出迴歸結果
print("迴歸分析結果：")
print(model.summary())
print(f"\nMSE: {mse:.4f}")

# 建立一個 DataFrame 來儲存迴歸結果
regression_results = pd.DataFrame(columns=[
    'Variable',
    'Coefficient',
    'Std Error',
    'T-Stat',
    'P-Value',
    'Significance'
])

# 取得模型結果
variables = ['const', 'Skewness', 'Median', 'Fear and Greed Index']
coefficients = model.params
std_errors = model.bse
t_stats = model.tvalues
p_values = model.pvalues

# 判斷顯著性
def get_significance(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''

# 整理結果
for var in variables:
    idx = variables.index(var)
    regression_results = pd.concat([regression_results, pd.DataFrame({
        'Variable': [var],
        'Coefficient': [coefficients[idx]],
        'Std Error': [std_errors[idx]],
        'T-Stat': [t_stats[idx]],
        'P-Value': [p_values[idx]],
        'Significance': [get_significance(p_values[idx])]
    })], ignore_index=True)

# 加入模型整體統計量
model_stats = pd.DataFrame({
    'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
    'Value': [
        model.rsquared,
        model.rsquared_adj,
        model.fvalue,
        model.f_pvalue,
        model.nobs,
        mse
    ]
})

# 顯示結果
print("\n迴歸係數及顯著性：")
print(regression_results.round(4))
print("\n模型統計量：")
print(model_stats.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 儲存結果
regression_results.to_csv('regression_model_coefficients.csv', index=False, encoding='utf-8-sig')
model_stats.to_csv('regression_model_statistics.csv', index=False, encoding='utf-8-sig')


# 基於這個模型的四因子迴歸分析（固定 Kurtosis、Median 和 Fear and Greed Index）
# 建立一個 DataFrame 來儲存迴歸結果
quadvariate_regression_results = pd.DataFrame(columns=[
    'Variable',
    'Kurtosis_Coef', 'Kurtosis_p', 'Kurtosis_Sig',
    'Median_Coef', 'Median_p', 'Median_Sig',
    'Fear and Greed Index_Coef', 'Fear and Greed Index_p', 'Fear and Greed Index_Sig',
    'Variable_Coef', 'Variable_p', 'Variable_Sig',
    'R-squared', 'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 對每個 X 變數進行四因子迴歸（與 Kurtosis, Median, Fear and Greed Index 配對）
for var in variables_to_standardize:
    if var not in ['T Return', 'Kurtosis', 'Median', 'Fear and Greed Index']:  # 排除 Y 變數和固定的三個因子
        # 準備 X 變數
        X = df_regression_week_stats_with_returns[['Kurtosis', 'Median', 'Fear and Greed Index', var]]
        X = sm.add_constant(X)  # 加入常數項
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        p_value_kurt = model.pvalues[1]
        if p_value_kurt < 0.01:
            sig_kurt = '***'
        elif p_value_kurt < 0.05:
            sig_kurt = '**'
        elif p_value_kurt < 0.1:
            sig_kurt = '*'
        else:
            sig_kurt = ''
            
        # 判斷 Median 的顯著水準
        p_value_median = model.pvalues[2]
        if p_value_median < 0.01:
            sig_median = '***'
        elif p_value_median < 0.05:
            sig_median = '**'
        elif p_value_median < 0.1:
            sig_median = '*'
        else:
            sig_median = ''
            
        # 判斷 Fear and Greed Index 的顯著水準
        p_value_fgi = model.pvalues[3]
        if p_value_fgi < 0.01:
            sig_fgi = '***'
        elif p_value_fgi < 0.05:
            sig_fgi = '**'
        elif p_value_fgi < 0.1:
            sig_fgi = '*'
        else:
            sig_fgi = ''
            
        # 判斷第四個變數的顯著水準
        p_value_var = model.pvalues[4]
        if p_value_var < 0.01:
            sig_var = '***'
        elif p_value_var < 0.05:
            sig_var = '**'
        elif p_value_var < 0.1:
            sig_var = '*'
        else:
            sig_var = ''

        # 儲存結果
        quadvariate_regression_results = pd.concat([quadvariate_regression_results, pd.DataFrame({
            'Variable': [var],
            'Kurtosis_Coef': [model.params[1]],     # Kurtosis 係數
            'Kurtosis_p': [p_value_kurt],           # Kurtosis p-value
            'Kurtosis_Sig': [sig_kurt],             # Kurtosis 顯著性
            'Median_Coef': [model.params[2]],     # Median 係數
            'Median_p': [p_value_median],           # Median p-value
            'Median_Sig': [sig_median],             # Median 顯著性
            'Fear and Greed Index_Coef': [model.params[3]],          # Fear and Greed Index 係數
            'Fear and Greed Index_p': [p_value_fgi],                 # Fear and Greed Index p-value
            'Fear and Greed Index_Sig': [sig_fgi],                   # Fear and Greed Index 顯著性
            'Variable_Coef': [model.params[4]],      # 變數係數
            'Variable_p': [p_value_var],             # 變數 p-value
            'Variable_Sig': [sig_var],               # 變數顯著性
            'R-squared': [model.rsquared],           # R-squared
            'MSE': [mse]                             # MSE
        })], ignore_index=True)

# 顯示結果
print("\n四因子迴歸分析結果（固定 Kurtosis、Median 和 Fear and Greed Index）：")
print(quadvariate_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

# 將結果儲存為 CSV
quadvariate_regression_results.to_csv('quadvariate regression results.csv', index=False, encoding='utf-8-sig')
print(f"\n四因子迴歸分析結果已儲存至 quadvariate regression results.csv")

###########################################################

# 儲存迴歸結果
with open(f'regression_results_week_兩個點_{today}.txt', 'w', encoding='utf-8') as f:
    f.write(model.summary().as_text())

# 計算VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 計算每個變數的VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_1.columns
vif_data["VIF"] = [variance_inflation_factor(X_1.values, i) for i in range(X_1.shape[1])]

print("\n變異數膨脹因子(VIF)：")
print(vif_data)

# 儲存VIF結果
vif_data.to_csv(f'vif_results_week_兩個點_{today}.csv', index=False, encoding='utf-8-sig')




''' Function '''
# 讀取資料
def read_data_v2(expiration_date):
    #formatted_date = datetime.strptime(expiration_date, "%Y-%m-%d").strftime("%d%b%y").upper()
    call_iv = pd.read_csv(f"deribit_data/iv/call/call_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    put_iv = pd.read_csv(f"deribit_data/iv/put/put_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    df_idx = pd.read_csv(f"deribit_data/BTC-index/BTC_index_{expiration_date}.csv", index_col="Unnamed: 0")

    call_iv.columns = call_iv.columns.astype(int)
    put_iv.columns = put_iv.columns.astype(int)

    call_price = pd.read_csv(f"deribit_data/BTC-call/call_strike_{expiration_date}.csv", index_col="Unnamed: 0")
    put_price = pd.read_csv(f"deribit_data/BTC-put/put_strike_{expiration_date}.csv", index_col="Unnamed: 0")

    call_price.columns = call_price.columns.astype(int)
    put_price.columns = put_price.columns.astype(int)
    
    #df_F = find_F_df(call_iv, put_iv, call_price, put_price, df_idx)

    return call_iv, put_iv, call_price, put_price, df_idx#, df_F


# 找出 F
def find_F1(K, type):
    global observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    def calculate_call_price(F, K, sigma, T, S0):
        d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return ( norm.cdf(d1) - K / F * norm.cdf(d2) ) * S0

    def calculate_put_price(F, K, sigma, T, S0):
        d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return ( K / F * norm.cdf(-d2) - norm.cdf(-d1) ) * S0

    def newton_method(real_price, K, sigma, T, S, type, tolerance=1e-6, max_iterations=1000):
        F = S*0.8
        for _ in range(max_iterations):
            if type=="C":
                guess_price = calculate_call_price(F, K, sigma, T, S)
            elif type=="P":
                guess_price = calculate_put_price(F, K, sigma, T, S)

            F_new = F + abs(guess_price - real_price) * 0.5
            if abs(real_price - guess_price) < tolerance:
                return F_new
            F = F_new
            
        return F
    
    if type=="C":
        price = call_price[K].loc[observation_date]  
        sigma = call_iv[K].loc[observation_date]
    if type=="P":
        price = put_price[K].loc[observation_date]  
        sigma = put_iv[K].loc[observation_date]
    
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observation_date)).days /365 
    S = df_idx["index_price"].loc[observation_date]

    return newton_method(price, K, sigma, T, S, type)


# 找出 F
def find_F2():
    global observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    S = df_idx["index_price"].loc[observation_date]
    df = call_price.loc[observation_date][call_price.loc[observation_date]!=0]
    result_c = df[(df.index >= S*0.9) & (df.index <= S*1.5)]
    result_c = pd.DataFrame(result_c)
    result_c.columns = ["C"]

    df = put_price.loc[observation_date][put_price.loc[observation_date]!=0]
    result_p = df[(df.index >= S*0.8) & (df.index <= S*1.1)]
    result_p = pd.DataFrame(result_p)
    result_p.columns = ["P"]

    F_values_c = [find_F1(K, "C") for K in result_c.index]
    F_values_p =  [find_F1(K, "P") for K in result_p.index]
    F = np.array(F_values_c+F_values_p).mean()
    
    return F


# 找出 F、T、S
def get_FTS():
    global observation_date, expiration_date
    F = find_F2()
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observation_date)).days/365
    S = df_idx["index_price"].loc[observation_date]
    return {"F": F, "T": T, "S": S}


# 定義混合買權、賣權隱含波動率函數
def mix_cp_function_v2():
    global observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    mix = pd.concat([call_iv.loc[observation_date], put_iv.loc[observation_date]], axis=1)
    mix.columns = ["C", "P"]
    mix = mix.replace(0, np.nan)

    # atm
    atm = mix.loc[(mix.index <= F*1.1) & (mix.index >= F*0.9)]
    atm["mixIV"] = atm[["C","P"]].mean(axis=1)

    # otm
    otm = pd.DataFrame(pd.concat([mix.loc[mix.index < F*0.9, 'P'], mix.loc[mix.index > F*1.1, 'C'] ], axis=0), columns=["mixIV"])

    # mix
    mix_cp = pd.concat([atm, otm], axis=0).sort_index()
    mix_cp[["C","P"]] = mix
    mix_cp = mix_cp.dropna(subset=["mixIV"])
    mix_cp = mix_cp.loc[:F*2.5]
    
    return mix_cp


# 定義 UnivariateSpline 函數
def UnivariateSpline_function_v2(mix_cp, power=4, s=None, w=None):
    global observation_date, expiration_date, delta_x
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]
    spline = UnivariateSpline(mix_cp.index, mix_cp["mixIV"], k=power, s=s, w=w)
    
    min_K = 0
    max_K = int(max(mix_cp.index)*1.2)
    dK = delta_x
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth_IV = pd.DataFrame([K_fine, Vol_fine], index=["K", "mixIV"]).T
    smooth_IV["C"] = call.future(F, smooth_IV["K"], T, smooth_IV["mixIV"], S)

    return smooth_IV


# 定義 LSQUnivariateSpline 函數
def UnivariateSpline_function_v3(mix_cp, power=4, s=None, w=None):
    global observation_date, expiration_date, delta_x
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]
    
    # 在 F 的位置加入 knot
    knots = np.array([F])
    spline = LSQUnivariateSpline(mix_cp.index, mix_cp["mixIV"], knots, k=power)
    
    min_K = 0
    max_K = int(max(mix_cp.index)*1.2)
    dK = delta_x
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth_IV = pd.DataFrame([K_fine, Vol_fine], index=["K", "mixIV"]).T
    smooth_IV["C"] = call.future(F, smooth_IV["K"], T, smooth_IV["mixIV"], S)
    
    return smooth_IV


# 定義繪製隱含波動率圖表的函數
def plot_implied_volatility(mix_cp):
    global observation_date, expiration_date
    basicinfo = get_FTS()
    futures_price = basicinfo["F"]

    # 繪製買權履約價與隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp['C'], color='orange')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Strike Price for call options')
    plt.show()

    # 繪製賣權履約價與隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp['P'], color='blue')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Strike Price for put options')
    plt.show()

    # 繪製買權與賣權的隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp['C'], color='orange', label='Call IV')
    plt.scatter(mix_cp.index, mix_cp['P'], color='blue', label='Put IV')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5, label='Futures Price')
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f'F = {int(futures_price)}', transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製買權與賣權的隱含波動率的散布圖，顯示出 0.9F ~ 1.1F 的 mixIV
    plt.figure(figsize=(10, 6), dpi=200)
    # 0.9F 到 1.1F 內的數據
    atm_mask = (mix_cp.index >= futures_price * 0.9) & (mix_cp.index <= futures_price * 1.1)
    plt.scatter(mix_cp.index[atm_mask], mix_cp['C'][atm_mask], color='orange', label='Call IV')
    plt.scatter(mix_cp.index[atm_mask], mix_cp['P'][atm_mask], color='blue', label='Put IV')
    plt.scatter(mix_cp.index[atm_mask], mix_cp['mixIV'][atm_mask], color='green', label='Mix IV')
    # 0.9F 到 1.1F 外的數據
    otm_mask_low = mix_cp.index < futures_price * 0.9
    otm_mask_high = mix_cp.index > futures_price * 1.1
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp['C'][otm_mask_low], color='orange', alpha=0.5, edgecolors='none')
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp['C'][otm_mask_high], color='orange', alpha=0.5, edgecolors='none')
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp['P'][otm_mask_low], color='blue', alpha=0.5, edgecolors='none')
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp['P'][otm_mask_high], color='blue', alpha=0.5, edgecolors='none')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5, label='Futures Price')
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f'F = {int(futures_price)}', transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製 mix 後之買權與賣權的隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    # 0.9F 到 1.1F 內的數據
    atm_mask = (mix_cp.index >= futures_price * 0.9) & (mix_cp.index <= futures_price * 1.1)
    plt.scatter(mix_cp.index[atm_mask], mix_cp['mixIV'][atm_mask], color='green', label='Mix IV')
    # 0.9F 到 1.1F 外的數據
    otm_mask_low = mix_cp.index < futures_price * 0.9
    otm_mask_high = mix_cp.index > futures_price * 1.1
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp['C'][otm_mask_high], color='orange', label='Call IV')
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp['P'][otm_mask_low], color='blue', label='Put IV')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5, label='Futures Price')
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f'F = {int(futures_price)}', transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# RND
def RND_function(smooth_IV):
    
    # 方法一 直接微分
    smooth_IV["cdf"] = np.gradient(smooth_IV['C'], smooth_IV['K'])+1
    smooth_IV["pdf"] = np.gradient(np.gradient(smooth_IV['C'], smooth_IV['K']), smooth_IV['K'])

    # 方法二
    dk = smooth_IV["K"].iloc[1] - smooth_IV["K"].iloc[0]
    smooth_IV["RND"] =  (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2*smooth_IV["C"]) / ((dk)**2) #np.exp(r*T) *
    smooth_IV = smooth_IV.dropna()

    # RND 平滑
    smooth_IV["RND"] = savgol_filter(smooth_IV["RND"], 500, 3) # 平滑

    smooth_IV['right_cumulative'] = 1 - smooth_IV['cdf']

    # 只保存 mix_cp.index.min() <= K <= mix_cp.index.max() 的數據
    smooth_IV = smooth_IV[(smooth_IV['K'] >= df_options_mix.index.min()) & (smooth_IV['K'] <= df_options_mix.index.max())]

    # 過濾無效數據
    smooth_IV = smooth_IV[(smooth_IV['right_cumulative'].notna()) & 
              (smooth_IV['cdf'].notna()) &
              (smooth_IV['right_cumulative'] < 1) & 
              (smooth_IV['right_cumulative'] > 0) &
              (smooth_IV['cdf'] < 1) &
              (smooth_IV['cdf'] > 0)]

    # 將欄位 K 名稱改為 strike_price；將 mixIV 改為 fit_imp_vol；將 C 改為 fit_call；將 cdf 改為 left_cumulative；將 RND 改為 RND_density
    smooth_IV = smooth_IV.rename(columns={'K': 'strike_price', 'mixIV': 'fit_imp_vol', 'C': 'fit_call', 'cdf': 'left_cumulative', 'RND': 'RND_density'})

    fit = smooth_IV

    return fit

# RND
def RND_function(smooth_IV):
    """
    計算風險中性機率密度 (Risk-Neutral Density) 及相關分配函數
    
    參數:
        smooth_IV: 包含平滑隱含波動率的資料框
    
    回傳:
        處理後的資料框，包含 RND 密度和累積分配函數
    """
    # 計算步長
    dk = smooth_IV["K"].iloc[1] - smooth_IV["K"].iloc[0]
    
    # 計算累積分配函數 (CDF)
    smooth_IV["left_cumulative"] = np.gradient(smooth_IV['C'], smooth_IV['K']) + 1
    smooth_IV["right_cumulative"] = 1 - smooth_IV["left_cumulative"]
    
    # 計算機率密度函數 (PDF)
    smooth_IV["pdf"] = np.gradient(np.gradient(smooth_IV['C'], smooth_IV['K']), smooth_IV['K'])
    
    # 使用有限差分法計算 RND
    smooth_IV["RND_density"] = (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2*smooth_IV["C"]) / (dk**2)
    
    # 移除計算過程中產生的 NaN 值
    smooth_IV = smooth_IV.dropna()
    
    # 使用 Savitzky-Golay 濾波器平滑 RND
    smooth_IV["RND_density"] = savgol_filter(smooth_IV["RND_density"], 500, 3)
    
    # 只保留有效範圍內的資料
    if 'df_options_mix' in globals():
        smooth_IV = smooth_IV[(smooth_IV['K'] >= df_options_mix.index.min()) & 
                              (smooth_IV['K'] <= df_options_mix.index.max())]
    
    # 過濾無效資料
    smooth_IV = smooth_IV[(smooth_IV['right_cumulative'].notna()) & 
                          (smooth_IV['left_cumulative'].notna()) &
                          (smooth_IV['right_cumulative'] < 1) & 
                          (smooth_IV['right_cumulative'] > 0) &
                          (smooth_IV['left_cumulative'] < 1) &
                          (smooth_IV['left_cumulative'] > 0)]
    
    # 重新命名欄位以提高可讀性
    smooth_IV = smooth_IV.rename(columns={
        'K': 'strike_price', 
        'mixIV': 'fit_imp_vol', 
        'C': 'fit_call'
    })
    
    return smooth_IV

# 定義繪製擬合曲線的函數
def plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date):
    # 繪製隱含波動率微笑擬合圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(df_options_mix.index, df_options_mix['mixIV'],color='green', label='Mix IV')
    plt.plot(fit['strike_price'], fit['fit_imp_vol'], color='orange', label='Fitted IV')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility Smile of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製買權曲線
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['fit_call'], color='orange', label='Fitted Call Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Price')
    plt.title(f'Call Curve of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製經驗風險中性密度 (PDF)
    plt.figure(figsize=(10, 6), dpi=200)
    # 設定 y 軸格式為 10^n
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(fit['strike_price'], fit['RND_density'], color='orange', label='Empirical RND')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.title(f'Empirical Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製經驗風險中性累積分佈函數 (CDF)
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['left_cumulative'], color='orange', label='CDF')
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# 定義擬合 GPD 的函數，選 1 個點，比較斜率與 CDF
def fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95):
    """使用斜率和CDF擬合GPD尾部
    
    Args:
        fit (DataFrame): 包含RND和CDF資料的DataFrame
        initial_i (int): 初始搜尋步數
        delta_x (float): x軸間距
        alpha_1L (float): 左尾接合點的累積機率
        alpha_1R (float): 右尾接合點的累積機率
    """
    # 檢查左尾累積機率
    if fit['left_cumulative'].iloc[0] > alpha_1L:
        alpha_1L = fit['left_cumulative'].iloc[0] + 0.0005
        print(f"警告：left_cumulative[0] ({fit['left_cumulative'].iloc[0]:.4f}) 大於 alpha_1L，將接合點設為 {alpha_1L}。")

    #--------------------
    # 右尾擬合
    #--------------------
    # 1. 找到接合點
    loc_right = fit.iloc[(fit['left_cumulative'] - alpha_1R).abs().argsort()[:1]]
    right_end = loc_right['strike_price'].values[0]
    right_missing_tail = loc_right['right_cumulative'].values[0]
    right_sigma = right_missing_tail / loc_right['RND_density'].values[0]

    # 2. 尋找斜率變化點
    i = initial_i
    loc_index = fit.index.get_loc(loc_right.index[0])
    while True:
        try:
            if loc_index - i < 0:
                print(f"警告：i={i} 太大，導致索引為負。")
                break
            current_density = fit.iloc[loc_index-i]['RND_density']
            right_slope = (current_density - loc_right['RND_density'].values[0]) / (i * delta_x)
            if right_slope < 0:
                break
            i += 1
        except (KeyError, IndexError):
            print("警告：索引錯誤。")
            break

    # 3. GPD擬合
    def right_objective(x):
        xi, scale = x
        X_alpha = right_end
        slope_error = ((right_missing_tail * gpd.pdf(X_alpha + delta_x, xi, loc=X_alpha, scale=scale) - 
                       loc_right['RND_density'].values[0]) / delta_x - right_slope)
        cdf_error = (gpd.cdf(X_alpha, xi, loc=X_alpha, scale=scale) - loc_right['right_cumulative'].values[0])
        return (1e12 * slope_error**2) + (1e12 * cdf_error**2)

    right_fit = minimize(right_objective, [0, right_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    right_xi, right_sigma = right_fit.x

    # 4. 擴展並填充右尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, 160010, delta_x)}), how='outer')
    fit['right_extra_density'] = right_missing_tail * gpd.pdf(fit['strike_price'], right_xi, loc=right_end, scale=right_sigma)
    fit['full_density'] = fit['RND_density'].fillna(0)
    fit.loc[fit['strike_price'] >= right_end, 'full_density'] = fit.loc[fit['strike_price'] >= right_end, 'right_extra_density'].fillna(0)

    #--------------------
    # 左尾擬合
    #--------------------
    # 1. 準備資料
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    loc_left = fit.iloc[(fit['left_cumulative'] - alpha_1L).abs().argsort()[:1]]
    left_end = loc_left['strike_price'].values[0]
    left_missing_tail = loc_left['left_cumulative'].values[0]
    left_sigma = left_missing_tail / loc_left['RND_density'].values[0]

    # 2. 尋找斜率變化點
    i = initial_i
    loc_index = fit.index.get_loc(loc_left.index[0])
    max_index = len(fit) - 1
    while True:
        try:
            if loc_index + i > max_index:
                print(f"警告：i={i} 太大，導致索引超出範圍。")
                break
            current_density = fit.iloc[loc_index+i]['RND_density']
            left_slope = (current_density - loc_left['RND_density'].values[0]) / (i * delta_x)
            if left_slope > 0:
                break
            i += 1
        except (KeyError, IndexError):
            print("警告：索引錯誤。")
            break

    # 3. GPD擬合
    def left_objective(x):
        xi, scale = x
        X_alpha = loc_left['reverse_strike'].values[0]
        slope_error = ((left_missing_tail * gpd.pdf(X_alpha + delta_x, xi, loc=X_alpha, scale=scale) - 
                       loc_left['RND_density'].values[0]) / delta_x - left_slope)
        cdf_error = (gpd.cdf(X_alpha, xi, loc=X_alpha, scale=scale) - loc_left['left_cumulative'].values[0])
        return (1e12 * slope_error**2) + (1e12 * cdf_error**2)

    left_fit = minimize(left_objective, [0, left_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    left_xi, left_sigma = left_fit.x

    # 4. 擴展並填充左尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = left_missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, 
                                                           loc=loc_left['reverse_strike'].values[0], scale=left_sigma)
    fit.loc[fit['strike_price'] <= left_end, 'full_density'] = fit.loc[fit['strike_price'] <= left_end, 'left_extra_density'].fillna(0)

    #--------------------
    # 後處理
    #--------------------
    fit = fit.sort_values('strike_price')
    fit['full_density'] = fit['full_density'].interpolate(method='cubic')
    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    return fit, left_end, right_end


# 定義擬合 GPD 的函數，選 2 個點，比較 PDF
def fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98):
    """使用兩點PDF比較來擬合GPD尾部
    
    Args:
        fit (DataFrame): 包含RND和CDF資料的DataFrame
        delta_x (float): x軸間距
        alpha_2L (float): 左尾第二個接合點的累積機率
        alpha_1L (float): 左尾第一個接合點的累積機率
        alpha_1R (float): 右尾第一個接合點的累積機率
        alpha_2R (float): 右尾第二個接合點的累積機率
    """
    #--------------------
    # 右尾擬合
    #--------------------
    # 1. 找到接合點
    loc_right = fit.iloc[(fit['left_cumulative'] - alpha_1R).abs().argsort()[:1]]
    right_missing_tail = loc_right['right_cumulative'].values[0]
    right_sigma = right_missing_tail / loc_right['RND_density'].values[0]
    
    # 2. 找到第二個接合點
    X_alpha_2R = fit.iloc[(fit['right_cumulative'] - alpha_2R).abs().argsort()[:1]]['strike_price'].values[0]

    # 3. GPD擬合
    def right_objective(x):
        xi, scale = x
        X_alpha_1R = loc_right['strike_price'].values[0]
        density_error_1R = (right_missing_tail * gpd.pdf(X_alpha_1R + delta_x, xi, loc=X_alpha_1R, scale=scale) - 
                          loc_right['RND_density'].values[0])
        density_error_2R = (right_missing_tail * gpd.pdf(X_alpha_2R + delta_x, xi, loc=X_alpha_2R, scale=scale) - 
                          loc_right['RND_density'].values[0])
        return (1e12 * density_error_1R**2) + (1e12 * density_error_2R**2)

    right_fit = minimize(right_objective, [0, right_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    right_xi, right_sigma = right_fit.x

    # 4. 擴展並填充右尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, 160010, delta_x)}), how='outer')
    fit['right_extra_density'] = right_missing_tail * gpd.pdf(fit['strike_price'], right_xi, 
                                                             loc=loc_right['strike_price'].values[0], scale=right_sigma)
    fit['full_density'] = np.where(fit['strike_price'] > loc_right['strike_price'].values[0], 
                                  fit['right_extra_density'], fit['RND_density'])

    #--------------------
    # 左尾擬合
    #--------------------
    # 1. 準備資料
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    loc_left = fit.iloc[(fit['left_cumulative'] - alpha_1L).abs().argsort()[:1]]
    left_missing_tail = loc_left['left_cumulative'].values[0]
    left_sigma = left_missing_tail / loc_left['RND_density'].values[0]
    
    # 2. 找到第二個接合點
    X_alpha_2L = fit.iloc[(fit['left_cumulative'] - alpha_2L).abs().argsort()[:1]]['reverse_strike'].values[0]

    # 3. GPD擬合
    def left_objective(x):
        xi, scale = x
        X_alpha_1L = loc_left['reverse_strike'].values[0]
        density_error_1L = (left_missing_tail * gpd.pdf(X_alpha_1L + delta_x, xi, loc=X_alpha_1L, scale=scale) - 
                          loc_left['RND_density'].values[0])
        density_error_2L = (left_missing_tail * gpd.pdf(X_alpha_2L + delta_x, xi, loc=X_alpha_2L, scale=scale) - 
                          loc_left['RND_density'].values[0])
        return (1e12 * density_error_1L**2) + (1e12 * density_error_2L**2)

    left_fit = minimize(left_objective, [0, left_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    left_xi, left_sigma = left_fit.x

    # 4. 擴展並填充左尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = left_missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, 
                                                           loc=loc_left['reverse_strike'].values[0], scale=left_sigma)
    fit.loc[fit['strike_price'] < loc_left['strike_price'].values[0], 'full_density'] = \
        fit.loc[fit['strike_price'] < loc_left['strike_price'].values[0], 'left_extra_density']

    #--------------------
    # 後處理
    #--------------------
    fit = fit.sort_values('strike_price')
    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    # 找出指定累積機率對應的界限
    lower_bound = fit.loc[(fit['full_density_cumulative'] - alpha_1L).abs().idxmin(), 'strike_price']
    upper_bound = fit.loc[(fit['full_density_cumulative'] - alpha_1R).abs().idxmin(), 'strike_price']

    return fit, lower_bound, upper_bound


# 定義繪製擬合 GPD 的函數
def plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date):
    # RND
    plt.figure(figsize=(10, 6), dpi=200)
    
    # 設定 y 軸格式為 10^n
    from matplotlib.ticker import ScalarFormatter
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 原始 RND
    plt.plot(fit['strike_price'], fit['full_density'], label='Empirical RND', color='royalblue')
    
    # 找出 CDF 為 5% 和 95% 的點
    left_point = fit.loc[(fit['left_cumulative'] - 0.05).abs().idxmin()]
    right_point = fit.loc[(fit['left_cumulative'] - 0.95).abs().idxmin()]
    
    # 左尾 GPD
    left_tail = fit[fit['strike_price'] <= upper_bound]
    plt.plot(left_tail['strike_price'], left_tail['left_extra_density'], 
             label='Left tail GPD', color='orange', linestyle=':', linewidth=2)
    
    # 右尾 GPD
    right_tail = fit[fit['strike_price'] >= lower_bound]
    plt.plot(right_tail['strike_price'], right_tail['right_extra_density'], 
             label='Right tail GPD', color='green', linestyle=':', linewidth=2)
    
    # 在 5% 和 95% 的點加上黑色空心圓圈
    plt.plot(left_point['strike_price'], left_point['full_density'], 'o', 
             color='black', fillstyle='none', markersize=10)
    plt.plot(right_point['strike_price'], right_point['full_density'], 'o', 
             color='black', fillstyle='none', markersize=10)
    
    # 添加文字標註
    plt.annotate(r'$\alpha_{1L}=0.05$', 
                xy=(left_point['strike_price'], left_point['full_density']),
                xytext=(-65, 5), textcoords='offset points',
                fontsize=12)
    plt.annotate(r'$\alpha_{1R}=0.95$', 
                xy=(right_point['strike_price'], right_point['full_density']),
                xytext=(3, 5), textcoords='offset points',
                fontsize=12)
    
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# 定義繪製完整密度累積分佈函數的函數
def plot_full_density_cdf(fit, observation_date, expiration_date):
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['full_density_cumulative'], label='CDF')
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# 定義計算和繪製 RND 統計量的函數
def calculate_rnd_statistics(fit, delta_x):
    # 計算統計量
    RND_mean = np.sum(fit['strike_price'] * fit['full_density'] * delta_x)
    RND_std = np.sqrt(np.sum((fit['strike_price'] - RND_mean)**2 * fit['full_density'] * delta_x))
    
    fit['std_strike'] = (fit['strike_price'] - RND_mean) / RND_std
    RND_skew = np.sum(fit['std_strike']**3 * fit['full_density'] * delta_x)
    RND_kurt = np.sum(fit['std_strike']**4 * fit['full_density'] * delta_x) - 3

    # 計算分位數
    # fit['left_cumulative'] = np.cumsum(fit['full_density'] * delta_x)
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quants = [fit.loc[(fit['left_cumulative'] - q).abs().idxmin(), 'strike_price'] for q in quantiles]

    # 返回數據
    return {
        'mean': RND_mean,
        'std': RND_std,
        'skewness': RND_skew,
        'kurtosis': RND_kurt,
        'quantiles': dict(zip(quantiles, quants)),
        'rnd_data': fit[['strike_price', 'full_density']]
    }


# 定義繪製 RND 圖形及分位數的函數
def plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date):
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['full_density'], label='Empirical RND')
    for quant in quants:
        plt.axvline(x=quant, linestyle='--', color='gray')
    plt.xlabel('Strike Price')
    plt.ylabel('RND')
    plt.title(f'Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# 定義處理多個日期的函數，使用兩點的方法
def process_multiple_dates_two_points(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_1L=0.02, alpha_2L=0.05, alpha_1R=0.95, alpha_2R=0.98)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data


# 定義處理多個日期的函數，使用一點的方法
def process_multiple_dates_one_point(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data


# 繪製多個日期的 RND
def plot_multiple_rnd(all_rnd_data, observation_dates, expiration_date):
    plt.figure(figsize=(12, 8), dpi=100)
       
    for date in observation_dates:
        fit = all_rnd_data[date]
        plt.plot(fit['strike_price'], fit['full_density'], label=date)
       
    plt.xlabel('Strike Price')
    plt.ylabel('RND')
    plt.title(f'Multiple Dates Risk-Neutral Density Curve (Expiration Date: {expiration_date})')
    plt.legend(title='Observation Date', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# 生成日期列表
def generate_dates(start_date, end_date, interval_days=1):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    return date_list


# 使用積分反推買權價格
def calculate_call_option_price_discrete(fit, strike_price):
    # 假設 'x' 是資產價格，'pdf' 是對應的概率密度
    x_values = fit['strike_price'].values
    pdf_values = fit['full_density'].values
    
    # 計算期權價格的離散積分
    call_payoffs = np.maximum(x_values - strike_price, 0)
    call_price = np.trapz(call_payoffs * pdf_values, x_values)
    
    return call_price


# 計算所有大於 future_price 的行權價的買權價格，每隔 50 個計算一次
def calculate_call_option_prices_above_future_price(fit, future_price, step=50):
    # 假設 'x' 是資產價格，'pdf' 是對應的概率密度
    x_values = fit['strike_price'].values
    pdf_values = fit['full_density'].values
    
    # 找到所有大於 future_price 的行權價
    strike_prices = x_values[x_values > future_price]
    
    # 每隔 step 個行權價計算一次
    selected_strike_prices = [sp for sp in strike_prices if sp % step <= 0.01]
    # 將 selected_strike_prices 轉換為整數
    selected_strike_prices = [int(sp) for sp in selected_strike_prices]
    
    # 計算每個選定行權價的買權價格
    call_option_prices = {}
    for strike_price in selected_strike_prices:
        call_payoffs = np.maximum(x_values - strike_price, 0)
        call_price = np.trapz(call_payoffs * pdf_values, x_values)
        call_option_prices[strike_price] = call_price
    
    return call_option_prices, x_values, strike_prices, selected_strike_prices



# 定義處理多個日期的函數，求買權價格，使用兩點的方法
def find_call_option_prices_above_future_price_multiple_dates_two_points(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}
    all_call_option_prices = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_1L=0.02, alpha_2L=0.05, alpha_1R=0.95, alpha_2R=0.98)
            call_option_prices = calculate_call_option_prices_above_future_price(fit, future_price, step=1000)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
            all_call_option_prices[observation_date] = call_option_prices
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data, all_call_option_prices



# 定義處理多個日期的函數，求買權價格，使用一點的方法
def find_call_option_prices_above_future_price_multiple_dates_one_point(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}
    all_call_option_prices = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95)
            call_option_prices = calculate_call_option_prices_above_future_price(fit, future_price, step=1000)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
            all_call_option_prices[observation_date] = call_option_prices
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data, all_call_option_prices