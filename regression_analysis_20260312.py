# 基本數據處理與分析套件
import pandas as pd
import numpy as np
import statsmodels.api as sm
# 日期時間處理
from datetime import datetime
# 系統與工具套件
import nest_asyncio
import warnings


nest_asyncio.apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)
today = datetime.now().strftime('%Y-%m-%d')


''' 執行迴歸分析_每天_一個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv("output/RND_regression_all_1pt_1d_20260312.csv")

# 設定數值變數名稱
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 處理 MissingDataError: 移除 inf 並刪除包含 NaN 的列
df_regression_day_stats_with_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
df_regression_day_stats_with_returns.dropna(subset=numeric_columns, inplace=True)

# 對所有數值變數進行敘述統計
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