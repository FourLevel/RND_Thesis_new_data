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
import glob
import concurrent.futures

nest_asyncio.apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option("display.float_format", "{:.4f}".format)
initial_i = 1
delta_x = 0.1


"""
Functions
"""
def bs_call_future(F, K, T, sigma, S):
    """
    Black-76 模型計算買權理論價格（Deribit 反向合約版本）。
    """
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (norm.cdf(d1) - norm.cdf(d2) * K / F) * S


def read_data(expiration_date):
    """
    讀取指定到期日的選擇權資料。
    """
    call_iv = pd.read_csv(f"data/my_data/BTC_iv/call_iv/call_iv_{expiration_date}.csv", index_col=0) / 100
    put_iv = pd.read_csv(f"data/my_data/BTC_iv/put_iv/put_iv_{expiration_date}.csv", index_col=0) / 100
    df_idx = pd.read_csv("data/my_data/BTC_price/BTC_price.csv")
    df_idx["date"] = pd.to_datetime(df_idx["date"]).dt.strftime("%Y-%m-%d")
    df_idx = df_idx.set_index("date")

    call_iv.columns = call_iv.columns.astype(int)
    put_iv.columns = put_iv.columns.astype(int)

    call_price = pd.read_csv(f"data/my_data/BTC_call/call_strike_{expiration_date}.csv", index_col=0)
    put_price = pd.read_csv(f"data/my_data/BTC_put/put_strike_{expiration_date}.csv", index_col=0)

    call_price.columns = call_price.columns.astype(int)
    put_price.columns = put_price.columns.astype(int)

    return call_iv, put_iv, call_price, put_price, df_idx


def find_F(observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx):
    """
    從多個接近 ATM 的選擇權反推隱含遠期價格 F，取平均。
    對每個選定的履約價，透過牛頓法由市場價格反推單一隱含 F，最後取平均。
    """
    def _calculate_call_price(F, K, sigma, T, S0):
        d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (norm.cdf(d1) - K / F * norm.cdf(d2)) * S0

    def _calculate_put_price(F, K, sigma, T, S0):
        d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (K / F * norm.cdf(-d2) - norm.cdf(-d1)) * S0

    def _implied_F_single(K, opt_type, T, S, tolerance=1e-6, max_iterations=1000):
        """
        由單一選擇權的市場價格，透過牛頓法反推隱含遠期價格 F。
        """
        if opt_type == "C":
            price = call_price[K].loc[observation_date]
            sigma = call_iv[K].loc[observation_date]
        else:
            price = put_price[K].loc[observation_date]
            sigma = put_iv[K].loc[observation_date]

        F = S * 0.8
        for _ in range(max_iterations):
            if opt_type == "C":
                guess_price = _calculate_call_price(F, K, sigma, T, S)
            else:
                guess_price = _calculate_put_price(F, K, sigma, T, S)

            F_new = F + abs(guess_price - price) * 0.5
            if abs(price - guess_price) < tolerance:
                return F_new
            F = F_new

        return F

    S = df_idx["close"].loc[observation_date]
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observation_date)).days / 365

    # 篩選接近 ATM 且有成交的買權（0.9S ~ 1.5S）
    df = call_price.loc[observation_date][call_price.loc[observation_date] != 0]
    result_c = df[(df.index >= S * 0.9) & (df.index <= S * 1.5)]

    # 篩選接近 ATM 且有成交的賣權（0.8S ~ 1.1S）
    df = put_price.loc[observation_date][put_price.loc[observation_date] != 0]
    result_p = df[(df.index >= S * 0.8) & (df.index <= S * 1.1)]

    # 對每個履約價反推隱含 F，最後取平均
    F_values_c = [_implied_F_single(K, "C", T, S) for K in result_c.index]
    F_values_p = [_implied_F_single(K, "P", T, S) for K in result_p.index]
    F_avg = np.array(F_values_c + F_values_p).mean()

    return F_avg


def get_FTS(observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx):
    """
    取得隱含遠期價格 F、到期時間 T、現貨價格 S。
    """
    F_avg = find_F(observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx)
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observation_date)).days / 365
    S = df_idx["close"].loc[observation_date]
    return {"F": F_avg, "T": T, "S": S}


def mix_cp_function(observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx):
    """
    混合買權、賣權隱含波動率，ATM 區取平均，OTM 區各取一邊。
    """
    basicinfo = get_FTS(observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx)
    F = basicinfo["F"]

    mix = pd.concat([call_iv.loc[observation_date], put_iv.loc[observation_date]], axis=1)
    mix.columns = ["C", "P"]
    mix = mix.replace(0, np.nan)

    # ATM 區域：0.9F ~ 1.1F，取 Call/Put IV 平均
    atm = mix.loc[(mix.index <= F * 1.1) & (mix.index >= F * 0.9)].copy()
    atm["mixIV"] = atm[["C", "P"]].mean(axis=1)

    # OTM 區域：低於 0.9F 用 Put IV，高於 1.1F 用 Call IV
    otm = pd.DataFrame(
        pd.concat([mix.loc[mix.index < F * 0.9, "P"], mix.loc[mix.index > F * 1.1, "C"]], axis=0),
        columns=["mixIV"],
    )

    # 合併並過濾
    mix_cp = pd.concat([atm, otm], axis=0).sort_index()
    mix_cp[["C", "P"]] = mix
    mix_cp = mix_cp.dropna(subset=["mixIV"])
    mix_cp = mix_cp.loc[:F * 2.5]

    return mix_cp, basicinfo


def UnivariateSpline_function(mix_cp, basicinfo, power=4, s=None, w=None):
    """
    使用 LSQUnivariateSpline 平滑隱含波動率曲線，並計算理論買權價格。
    """
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    # 在 F 的位置加入 knot
    knots = np.array([F])
    spline = LSQUnivariateSpline(mix_cp.index, mix_cp["mixIV"], knots, k=power)

    min_K = 0
    max_K = int(max(mix_cp.index) * 1.2)
    dK = delta_x
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth_IV = pd.DataFrame([K_fine, Vol_fine], index=["K", "mixIV"]).T
    smooth_IV["C"] = bs_call_future(F, smooth_IV["K"], T, smooth_IV["mixIV"], S)

    return smooth_IV


def RND_function(smooth_IV, mix_cp=None):
    """
    計算風險中性機率密度 (Risk-Neutral Density) 及相關分配函數。

    參數:
        smooth_IV: 包含平滑隱含波動率的 DataFrame（欄位：K, mixIV, C）
        mix_cp: 原始含買賣權 IV 的資料 (用於過濾)

    回傳:
        處理後的 DataFrame，包含 RND 密度和累積分配函數
    """
    # 計算步長
    dk = smooth_IV["K"].iloc[1] - smooth_IV["K"].iloc[0]

    # 計算累積分配函數 (CDF)
    smooth_IV["left_cumulative"] = np.gradient(smooth_IV["C"], smooth_IV["K"]) + 1
    smooth_IV["right_cumulative"] = 1 - smooth_IV["left_cumulative"]

    # 計算機率密度函數 (PDF)
    smooth_IV["pdf"] = np.gradient(np.gradient(smooth_IV["C"], smooth_IV["K"]), smooth_IV["K"])

    # 使用有限差分法計算 RND
    smooth_IV["RND_density"] = (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2 * smooth_IV["C"]) / (dk ** 2)

    # 移除計算過程中產生的 NaN 值
    smooth_IV = smooth_IV.dropna()

    # 使用 Savitzky-Golay 濾波器平滑 RND
    smooth_IV["RND_density"] = savgol_filter(smooth_IV["RND_density"], 500, 3)

    # 只保留有效範圍內的資料
    if mix_cp is not None:
        smooth_IV = smooth_IV[
            (smooth_IV["K"] >= mix_cp.index.min()) & (smooth_IV["K"] <= mix_cp.index.max())
        ]

    # 過濾無效資料
    smooth_IV = smooth_IV[
        (smooth_IV["right_cumulative"].notna())
        & (smooth_IV["left_cumulative"].notna())
        & (smooth_IV["right_cumulative"] < 1)
        & (smooth_IV["right_cumulative"] > 0)
        & (smooth_IV["left_cumulative"] < 1)
        & (smooth_IV["left_cumulative"] > 0)
    ]

    # 重新命名欄位
    smooth_IV = smooth_IV.rename(columns={"K": "strike_price", "mixIV": "fit_imp_vol", "C": "fit_call"})

    return smooth_IV


def plot_implied_volatility(mix_cp, basicinfo, observation_date, expiration_date):
    """
    繪製隱含波動率圖表（共 5 張子圖）。
    """
    futures_price = basicinfo["F"]

    # 1. 買權 IV
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp["C"], color="orange")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility vs Strike Price for call options")
    plt.show()

    # 2. 賣權 IV
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp["P"], color="blue")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility vs Strike Price for put options")
    plt.show()

    # 3. 買權 + 賣權 IV
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp["C"], color="orange", label="Call IV")
    plt.scatter(mix_cp.index, mix_cp["P"], color="blue", label="Put IV")
    plt.axvline(x=futures_price, color="black", linestyle="--", alpha=0.5, label="Futures Price")
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f"F = {int(futures_price)}", transform=plt.gca().transData)
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.title(f"Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})")
    plt.legend()
    plt.show()

    # 4. 買權 + 賣權 IV，顯示 0.9F ~ 1.1F 的 mixIV
    plt.figure(figsize=(10, 6), dpi=200)
    atm_mask = (mix_cp.index >= futures_price * 0.9) & (mix_cp.index <= futures_price * 1.1)
    plt.scatter(mix_cp.index[atm_mask], mix_cp["C"][atm_mask], color="orange", label="Call IV")
    plt.scatter(mix_cp.index[atm_mask], mix_cp["P"][atm_mask], color="blue", label="Put IV")
    plt.scatter(mix_cp.index[atm_mask], mix_cp["mixIV"][atm_mask], color="green", label="Mix IV")
    otm_mask_low = mix_cp.index < futures_price * 0.9
    otm_mask_high = mix_cp.index > futures_price * 1.1
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp["C"][otm_mask_low], color="orange", alpha=0.5, edgecolors="none")
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp["C"][otm_mask_high], color="orange", alpha=0.5, edgecolors="none")
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp["P"][otm_mask_low], color="blue", alpha=0.5, edgecolors="none")
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp["P"][otm_mask_high], color="blue", alpha=0.5, edgecolors="none")
    plt.axvline(x=futures_price, color="black", linestyle="--", alpha=0.5, label="Futures Price")
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f"F = {int(futures_price)}", transform=plt.gca().transData)
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.title(f"Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})")
    plt.legend()
    plt.show()

    # 5. 混合後的 IV（OTM Call + ATM Mix + OTM Put）
    plt.figure(figsize=(10, 6), dpi=200)
    atm_mask = (mix_cp.index >= futures_price * 0.9) & (mix_cp.index <= futures_price * 1.1)
    plt.scatter(mix_cp.index[atm_mask], mix_cp["mixIV"][atm_mask], color="green", label="Mix IV")
    otm_mask_low = mix_cp.index < futures_price * 0.9
    otm_mask_high = mix_cp.index > futures_price * 1.1
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp["C"][otm_mask_high], color="orange", label="Call IV")
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp["P"][otm_mask_low], color="blue", label="Put IV")
    plt.axvline(x=futures_price, color="black", linestyle="--", alpha=0.5, label="Futures Price")
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f"F = {int(futures_price)}", transform=plt.gca().transData)
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.title(f"Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})")
    plt.legend()
    plt.show()


def plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date):
    """
    繪製 RND 擬合結果圖表（IV Smile、Call 曲線、RND 密度、CDF）。
    """

    # 1. 隱含波動率微笑擬合圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(df_options_mix.index, df_options_mix["mixIV"], color="green", label="Mix IV")
    plt.plot(fit["strike_price"], fit["fit_imp_vol"], color="orange", label="Fitted IV")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title(f"Implied Volatility Smile of BTC options on {observation_date} (expired on {expiration_date})")
    plt.legend()
    plt.show()

    # 2. 買權曲線
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit["strike_price"], fit["fit_call"], color="orange", label="Fitted Call Price")
    plt.xlabel("Strike Price")
    plt.ylabel("Price")
    plt.title(f"Call Curve of BTC options on {observation_date} (expired on {expiration_date})")
    plt.legend()
    plt.show()

    # 3. 經驗風險中性密度 (PDF)
    plt.figure(figsize=(10, 6), dpi=200)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.plot(fit["strike_price"], fit["RND_density"], color="orange", label="Empirical RND")
    plt.xlabel("Strike Price")
    plt.ylabel("Density")
    plt.title(f"Empirical Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})")
    plt.legend()
    plt.show()

    # 4. 經驗風險中性累積分佈函數 (CDF)
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit["strike_price"], fit["left_cumulative"], color="orange", label="CDF")
    plt.xlabel("Strike Price")
    plt.ylabel("Probability")
    plt.title(f"Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})")
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
        raise ValueError(
            f"left_cumulative[0] ({fit['left_cumulative'].iloc[0]:.4f}) 大於 alpha_1L ({alpha_1L})，"
            f"RND 資料不足以進行左尾 GPD 擬合，跳過此日期。"
        )

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
    # 檢查左尾累積機率
    if fit['left_cumulative'].iloc[0] > alpha_1L:
        raise ValueError(
            f"left_cumulative[0] ({fit['left_cumulative'].iloc[0]:.4f}) 大於 alpha_1L ({alpha_1L})，"
            f"RND 資料不足以進行左尾 GPD 擬合，跳過此日期。"
        )

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

def process_single_date(args):
    """
    處理單一到期日的核心計算邏輯（設計為支援平行運算）。
    """
    exp_date, lookback_days, gpd_method, delta_x = args
    obs_date = (pd.to_datetime(exp_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    try:
        call_iv, put_iv, call_price, put_price, df_idx = read_data(exp_date)

        if obs_date not in call_price.index or obs_date not in put_price.index or \
           obs_date not in call_iv.index or obs_date not in put_iv.index:
            return None, f"跳過：觀察日 {obs_date} 的資料不完整（{exp_date}）"

        df_options_mix, basicinfo = mix_cp_function(obs_date, exp_date, call_iv, put_iv, call_price, put_price, df_idx)
        smooth_IV = UnivariateSpline_function(df_options_mix, basicinfo, power=4)
        fit = RND_function(smooth_IV, df_options_mix)

        # GPD 尾端擬合
        if gpd_method == "1pt":
            fit, _, _ = fit_gpd_tails_use_slope_and_cdf_with_one_point(
                fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
            )
        else:
            fit, _, _ = fit_gpd_tails_use_pdf_with_two_points(
                fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
            )

        stats = calculate_rnd_statistics(fit, delta_x)

        result = {
            "Observation Date": obs_date,
            "Expiration Date": exp_date,
            "Mean": stats["mean"],
            "Std": stats["std"],
            "Skewness": stats["skewness"],
            "Kurtosis": stats["kurtosis"],
            "5% Quantile": stats["quantiles"][0.05],
            "25% Quantile": stats["quantiles"][0.25],
            "Median": stats["quantiles"][0.5],
            "75% Quantile": stats["quantiles"][0.75],
            "95% Quantile": stats["quantiles"][0.95],
        }
        return result, f"成功處理：觀察日 {obs_date}，到期日 {exp_date}"

    except Exception as e:
        return None, f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}，錯誤：{e}"


def build_regression_data(lookback_days=1, gpd_method="1pt", test_exp_dates=None):
    """
    建立迴歸分析資料。

    Args:
        lookback_days (int): 觀察日與到期日的間隔天數（1=每日，7=每週）
        gpd_method (str): GPD 擬合方法（"1pt"=一個點，"2pt"=兩個點）
        test_exp_dates (list): 測試用，僅處理指定的到期日（如 ["2020-03-20"]）。None 則處理全部。

    Returns:
        DataFrame: 包含 RND 統計量、報酬率、Fear and Greed Index、VIX 的完整迴歸資料
    """
    today = datetime.now().strftime("%Y%m%d")

    # 決定要處理的到期日
    if test_exp_dates is not None:
        all_expiration_dates = sorted(test_exp_dates)
        print(f"測試模式：處理 {len(all_expiration_dates)} 個指定到期日（GPD：{gpd_method}，間隔：{lookback_days}天）")
    else:
        files = glob.glob("data/my_data/BTC_call/call_strike_*.csv")
        all_expiration_dates = sorted(
            [os.path.basename(f).replace("call_strike_", "").replace(".csv", "") for f in files]
        )
        print(f"共找到 {len(all_expiration_dates)} 個到期日（GPD：{gpd_method}，間隔：{lookback_days}天）")

    # --------------------------------------------------
    # 計算每個到期日的 RND 統計量 (平行運算)
    # --------------------------------------------------
    stats_data = []
    
    # 準備 multiprocessing 的參數列表
    tasks = [(exp_date, lookback_days, gpd_method, delta_x) for exp_date in all_expiration_dates]

    import time
    start_time = time.time()
    
    # 使用 ProcessPoolExecutor 進行平行運算
    max_workers = min(os.cpu_count() or 4, 12)  # 避免開過多進程，最多取 12
    print(f"啟動平行運算，使用 {max_workers} 個核心...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_date, tasks)
        
        for res, msg in results:
            if msg:
                print(msg)
            if res is not None:
                stats_data.append(res)
                
    elapsed = time.time() - start_time
    print(f"基礎計算花費時間：{elapsed:.2f} 秒")

    df_stats = pd.DataFrame(stats_data)

    # 儲存 RND 統計資料
    output_rnd = f"output/regression_raw_data/RND_stats_{gpd_method}_{lookback_days}d_{today}.csv"
    df_stats.to_csv(output_rnd, index=False, encoding="utf-8-sig")
    print(f"\nRND 統計資料已儲存至 {output_rnd}")
    print(f"\n統計摘要：\n{df_stats.describe()}")

    # --------------------------------------------------
    # 合併 BTC 報酬率
    # --------------------------------------------------
    df_stats["Observation Date"] = pd.to_datetime(df_stats["Observation Date"]).dt.normalize()
    df_stats["Expiration Date"] = pd.to_datetime(df_stats["Expiration Date"]).dt.normalize()

    df_btcusdt = pd.read_csv("data/my_data/BTCUSDT_spot/BTCUSDT_spot.csv")
    df_btcusdt["Date"] = pd.to_datetime(df_btcusdt["Date"]).dt.normalize()
    df_btcusdt = df_btcusdt.set_index("Date")

    mask = (df_stats["Observation Date"] >= df_btcusdt.index.min()) & \
           (df_stats["Expiration Date"] <= df_btcusdt.index.max())
    df_filtered = df_stats[mask].copy()

    df_filtered["T Return"] = np.log(
        df_btcusdt.loc[df_filtered["Expiration Date"]]["Close"].values /
        df_btcusdt.loc[df_filtered["Observation Date"]]["Close"].values
    )

    df_filtered = df_filtered.sort_values("Observation Date")
    
    if len(df_filtered) > 4:
        for lag in range(1, 5):
            df_filtered[f"T-{lag} Return"] = df_filtered["T Return"].shift(lag)
        df_filtered = df_filtered.dropna()
    else:
        print(f"\n資料筆數不足 5 筆({len(df_filtered)}筆)，跳過 lag return 計算。")
    
    print(f"\n加入報酬率後的資料筆數：{len(df_filtered)}")

    # --------------------------------------------------
    # 合併 Fear and Greed Index 與 VIX
    # --------------------------------------------------
    df_fgi = pd.read_csv("data/my_data/Crypto_Fear_and_Greed_Index.csv")
    df_fgi["date"] = pd.to_datetime(df_fgi["date"]).dt.normalize()
    df_fgi = df_fgi.set_index("date")

    df_vix = pd.read_csv("data/my_data/VIX_index.csv")
    df_vix["DATE"] = pd.to_datetime(df_vix["DATE"], format="%m/%d/%Y").dt.normalize()
    df_vix = df_vix.set_index("DATE")

    df_filtered = pd.merge(
        df_filtered, df_fgi[["value"]],
        left_on="Observation Date", right_index=True, how="left"
    )
    df_filtered.rename(columns={"value": "Fear and Greed Index"}, inplace=True)

    df_filtered = pd.merge(
        df_filtered, df_vix[["CLOSE"]],
        left_on="Observation Date", right_index=True, how="left"
    )
    df_filtered.rename(columns={"CLOSE": "VIX"}, inplace=True)

    df_filtered["VIX"] = df_filtered["VIX"].ffill().bfill()

    print(f"\nFear and Greed Index 缺失值：{df_filtered['Fear and Greed Index'].isna().sum()}")
    print(f"VIX 缺失值：{df_filtered['VIX'].isna().sum()}")

    # 儲存最終結果
    output_all = f"output/regression_raw_data/RND_regression_all_{gpd_method}_{lookback_days}d_{today}.csv"
    df_filtered.to_csv(output_all, index=False, encoding="utf-8-sig")
    print(f"\n完整迴歸資料已儲存至 {output_all}")

    return df_filtered


if __name__ == "__main__":
    
    """
    建立迴歸分析資料
    """
    build_regression_data(lookback_days=1, gpd_method="1pt")  # 每日 + 一個點
    build_regression_data(lookback_days=1, gpd_method="2pt")  # 每日 + 兩個點
    build_regression_data(lookback_days=7, gpd_method="1pt")  # 每週 + 一個點
    build_regression_data(lookback_days=7, gpd_method="2pt")  # 每週 + 兩個點