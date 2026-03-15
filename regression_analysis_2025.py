# 基本數據處理與分析套件
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 繪圖套件
import matplotlib.pyplot as plt
import seaborn as sns

# 統計相關套件
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

# 系統套件
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)



''' 執行迴歸分析_每天_一個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_all_data_一個點_2025-02-25.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_1 = df_regression_day_stats_with_returns[[
    'Skewness', 'Median',
    'T-4 Return'
]]
y = df_regression_day_stats_with_returns['T Return']

# 加入常數項
X_1 = sm.add_constant(X_1)

# Out-of-sample Analysis
data = pd.concat([y, X_1], axis=1)
T = len(data)
initial_window = T * 0.6

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Skewness', 'Median', 'T-4 Return']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.tight_layout()
plt.show()


''' 執行迴歸分析_每天_兩個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_all_data_兩個點_2025-02-25.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_1 = df_regression_day_stats_with_returns[[
    'Skewness', 'Median',
    'T-4 Return'
]]
y = df_regression_day_stats_with_returns['T Return']

# 加入常數項
X_1 = sm.add_constant(X_1)

# Out-of-sample Analysis
data = pd.concat([y, X_1], axis=1)
T = len(data)
initial_window = T * 0.6

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Skewness', 'Median', 'T-4 Return']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.tight_layout()
plt.show()


''' 執行迴歸分析_每週_一個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_一個點_2025-02-25.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_4 = df_regression_week_stats_with_returns[[
     'Skewness', 'Median', 'Fear and Greed Index',
]]
y = df_regression_week_stats_with_returns['T Return']

# 加入常數項
X_4 = sm.add_constant(X_4)

# Out-of-sample Analysis
data = pd.concat([y, X_4], axis=1)
T = len(data)
initial_window = int(T * 0.8)

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Skewness', 'Median', 'Fear and Greed Index']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.tight_layout()
plt.show()


''' 執行迴歸分析_每週_兩個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_兩個點_2025-02-25.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX',
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_4 = df_regression_week_stats_with_returns[[
     'Skewness', 'Median', 'Fear and Greed Index',
]]
y = df_regression_week_stats_with_returns['T Return']

# 加入常數項
X_4 = sm.add_constant(X_4)

# Out-of-sample Analysis
data = pd.concat([y, X_4], axis=1)
T = len(data)
initial_window = int(T * 0.8)

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Skewness', 'Median', 'Fear and Greed Index']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.tight_layout()
plt.show()



''' Function '''
# 樣本外分析
def out_of_sample_analysis(data, initial_window, target_col, feature_cols):
    """
    執行樣本外分析
    
    參數：
    data: DataFrame, 包含目標變數和特徵
    initial_window: int, 初始訓練窗格大小 (s₀)
    target_col: str, 目標變數的欄位名稱 (R_t+1)
    feature_cols: list, 特徵變數的欄位名稱清單 (X_t)
    
    回傳：
    DataFrame: 包含實際值、預測值和歷史平均值
    float: R²_OS 值
    """
    
    # 初始化結果儲存
    results = []
    
    # 取得總樣本長度
    T = len(data)
    initial_window = int(initial_window)
    
    # 對每個預測時點進行迭代
    for t in range(initial_window, T-1):
        # 取得訓練資料
        train_data = data.iloc[:t]
        
        # 計算歷史平均值作為基準
        historical_mean = train_data[target_col].mean()
        
        # 擬合線性迴歸模型
        model = LinearRegression()
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        model.fit(X_train, y_train)
        
        # 進行預測
        X_predict = data.iloc[t:t+1][feature_cols]
        predicted_value = model.predict(X_predict)[0]
        
        # 取得實際值
        actual_value = data.iloc[t+1][target_col]
        
        # 儲存結果
        results.append({
            'time': data.index[t+1],
            'actual': actual_value,
            'predicted': predicted_value,
            'historical_mean': historical_mean
        })
    
    # 轉換結果為 DataFrame
    results_df = pd.DataFrame(results)
    
    # 計算 R²_OS
    numerator = np.sum((results_df['actual'] - results_df['predicted'])**2)
    denominator = np.sum((results_df['actual'] - results_df['historical_mean'])**2)
    R2_OS = 1 - numerator/denominator
    
    return results_df, R2_OS