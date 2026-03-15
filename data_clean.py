# data_clean.py - 使用 MAD 方法清理單一 CSV 檔案的極端值 & 日期對齊

import pandas as pd
import numpy as np

def calculate_mad_outliers(data, threshold=3):
    """
    使用中位數絕對偏差法 (MAD) 識別極端值
    
    Parameters:
    -----------
    data : array-like
        要分析的數據列
    threshold : float
        MAD 倍數，預設為 3（標準作法）
        
    Returns:
    --------
    dict : 包含極端值的詳細資訊
        - 'outlier_mask': bool array，True 表示該行是極端值
        - 'median': 中位數
        - 'mad': 中位數絕對偏差
        - 'lower_bound': 下界閾值
        - 'upper_bound': 上界閾值
        - 'outlier_count': 極端值個數
        - 'outlier_percentage': 極端值比例 (%)
    """
    # 移除 NaN 值進行計算
    data_clean = data.dropna()
    
    # 計算中位數
    median = np.median(data_clean)
    
    # 計算絕對偏差
    absolute_deviation = np.abs(data_clean - median)
    
    # 計算 MAD（絕對偏差的中位數）
    mad = np.median(absolute_deviation)
    
    # 設定閾值
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    
    # 識別極端值
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    # 統計資訊
    outlier_count = outlier_mask.sum()
    outlier_percentage = (outlier_count / len(data)) * 100 if len(data) > 0 else 0
    
    return {
        'outlier_mask': outlier_mask,
        'median': median,
        'mad': mad,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': outlier_count,
        'outlier_percentage': outlier_percentage
    }


def clean_data_mad(input_path, output_path, mad_threshold=3):
    """
    使用 MAD 方法清理 Skewness 和 Kurtosis 的極端值
    
    Parameters:
    -----------
    input_path : str
        輸入 CSV 檔案路徑
    output_path : str
        輸出 CSV 檔案路徑
    mad_threshold : float
        MAD 倍數閾值，預設為 3
    """
    
    # 讀取資料
    print(f"讀取資料檔案: {input_path}")
    df = pd.read_csv(input_path)
    
    original_count = len(df)
    print(f"原始資料筆數: {original_count}")
    print("\n" + "="*80)
    
    # 分別處理 Skewness 和 Kurtosis
    print("\n【Skewness 極端值分析】")
    print("-" * 80)
    skewness_results = calculate_mad_outliers(df['Skewness'], mad_threshold)
    
    print(f"中位數 (Median):        {skewness_results['median']:.6f}")
    print(f"MAD (Median Abs Dev):   {skewness_results['mad']:.6f}")
    print(f"下界閾值 (Lower Bound): {skewness_results['lower_bound']:.6f}")
    print(f"上界閾值 (Upper Bound): {skewness_results['upper_bound']:.6f}")
    print(f"極端值個數:             {skewness_results['outlier_count']}")
    print(f"極端值比例:             {skewness_results['outlier_percentage']:.2f}%")
    
    print("\n【Kurtosis 極端值分析】")
    print("-" * 80)
    kurtosis_results = calculate_mad_outliers(df['Kurtosis'], mad_threshold)
    
    print(f"中位數 (Median):        {kurtosis_results['median']:.6f}")
    print(f"MAD (Median Abs Dev):   {kurtosis_results['mad']:.6f}")
    print(f"下界閾值 (Lower Bound): {kurtosis_results['lower_bound']:.6f}")
    print(f"上界閾值 (Upper Bound): {kurtosis_results['upper_bound']:.6f}")
    print(f"極端值個數:             {kurtosis_results['outlier_count']}")
    print(f"極端值比例:             {kurtosis_results['outlier_percentage']:.2f}%")
    
    # 合併兩個條件：同時滿足 Skewness 或 Kurtosis 極端值的列
    combined_outliers = skewness_results['outlier_mask'] | kurtosis_results['outlier_mask']
    total_outliers = combined_outliers.sum()
    
    print("\n【整體極端值統計】")
    print("-" * 80)
    print(f"Skewness 極端值： {skewness_results['outlier_count']} 列")
    print(f"Kurtosis 極端值： {kurtosis_results['outlier_count']} 列")
    print(f"至少一項為極端值： {total_outliers} 列")
    print(f"極端值比例:       {(total_outliers/len(df))*100:.2f}%")
    
    # 剔除極端值
    df_cleaned = df[~combined_outliers].reset_index(drop=True)

    # 最後再清一次空值（給迴歸分析用）
    before_dropna_count = len(df_cleaned)
    df_cleaned = df_cleaned.dropna().reset_index(drop=True)
    removed_by_dropna = before_dropna_count - len(df_cleaned)

    cleaned_count = len(df_cleaned)
    
    print("\n【清理結果】")
    print("-" * 80)
    print(f"清理前筆數: {original_count}")
    print(f"清理後筆數: {cleaned_count}")
    print(f"移除筆數:   {original_count - cleaned_count}")
    print(f"因空值額外移除: {removed_by_dropna}")
    print(f"保留率:     {(cleaned_count/original_count)*100:.2f}%")
    print("=" * 80)
    
    # 保存清理後的資料
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n✓ 清理後的資料已保存至: {output_path}")
    
    return df_cleaned, {
        'skewness': skewness_results,
        'kurtosis': kurtosis_results,
        'original_count': original_count,
        'cleaned_count': cleaned_count
    }


def align_dates(reference_file, target_file, output_file):
    """
    根據參考檔案的日期對齊目標檔案
    
    Parameters:
    -----------
    reference_file : str
        參考檔案路徑（已清理的 1pt_1d 檔案）
    target_file : str
        目標檔案路徑（2pt_1d 檔案）
    output_file : str
        輸出檔案路徑（對齊後的 2pt_1d 檔案）
    """
    
    print(f"\n{'='*80}")
    print("【日期對齊】")
    print(f"{'='*80}\n")
    
    # 讀取參考檔案（已清理的 1pt_1d）
    print(f"1. 讀取參考檔案 (已清理的 1pt_1d): {reference_file}")
    df_reference = pd.read_csv(reference_file)
    print(f"   參考檔案筆數: {len(df_reference)}")
    
    # 提取日期組合
    reference_dates = df_reference[['Observation Date', 'Expiration Date']].copy()
    reference_dates = reference_dates.drop_duplicates().reset_index(drop=True)
    print(f"   獨特日期組合: {len(reference_dates)}")
    
    # 讀取目標檔案（2pt_1d）
    print(f"\n2. 讀取目標檔案 (2pt_1d): {target_file}")
    df_target = pd.read_csv(target_file)
    original_target_count = len(df_target)
    print(f"   目標檔案原筆數: {original_target_count}")
    
    # 對齊：使用 merge 保留只在參考檔案中出現的日期組合
    df_aligned = df_target.merge(
        reference_dates,
        on=['Observation Date', 'Expiration Date'],
        how='inner'
    ).reset_index(drop=True)
    
    aligned_count = len(df_aligned)
    removed_count = original_target_count - aligned_count
    
    print(f"\n【對齊結果】")
    print("-" * 80)
    print(f"對齊前筆數: {original_target_count}")
    print(f"對齊後筆數: {aligned_count}")
    print(f"移除筆數:   {removed_count}")
    print(f"保留率:     {(aligned_count/original_target_count)*100:.2f}%")
    print("=" * 80)
    
    # 保存對齊後的資料
    df_aligned.to_csv(output_file, index=False)
    print(f"\n✓ 對齊後的資料已保存至: {output_file}")
    
    return df_aligned


# ============================================================================
# 主程式
# ============================================================================

if __name__ == '__main__':
    
    # ========== Step 1: 清理 1pt_1d 資料 ==========
    print("\n【Step 1: 清理 1pt_1d 資料】\n")
    
    input_1pt_1d = "output/regression_raw_data/RND_regression_all_1pt_1d_20260312.csv"
    output_1pt_1d_cleaned = "output/regression_cleaned_data/RND_regression_all_1pt_1d_20260312.csv"
    
    df_1pt_1d_cleaned, results_1pt_1d = clean_data_mad(
        input_path=input_1pt_1d,
        output_path=output_1pt_1d_cleaned,
        mad_threshold=5
    )
    
    # ========== Step 2: 對齊 2pt_1d 資料 ==========
    print("\n\n【Step 2: 對齊 2pt_1d 資料】")
    
    reference_1pt_1d = output_1pt_1d_cleaned
    target_2pt_1d = "output/regression_raw_data/RND_regression_all_2pt_1d_20260312.csv"
    output_2pt_1d_aligned = "output/regression_cleaned_data/RND_regression_all_2pt_1d_20260312.csv"
    
    df_2pt_1d_aligned = align_dates(
        reference_file=reference_1pt_1d,
        target_file=target_2pt_1d,
        output_file=output_2pt_1d_aligned
    )
    
    # ========== Step 3: 清理 1pt_7d 資料 ==========
    print("\n\n【Step 3: 清理 1pt_7d 資料】\n")
    
    input_1pt_7d = "output/regression_raw_data/RND_regression_all_1pt_7d_20260312.csv"
    output_1pt_7d_cleaned = "output/regression_cleaned_data/RND_regression_all_1pt_7d_20260312.csv"
    
    df_1pt_7d_cleaned, results_1pt_7d = clean_data_mad(
        input_path=input_1pt_7d,
        output_path=output_1pt_7d_cleaned,
        mad_threshold=5
    )
    
    # ========== Step 4: 對齊 2pt_7d 資料 ==========
    print("\n\n【Step 4: 對齊 2pt_7d 資料】")
    
    reference_1pt_7d = output_1pt_7d_cleaned
    target_2pt_7d = "output/regression_raw_data/RND_regression_all_2pt_7d_20260312.csv"
    output_2pt_7d_aligned = "output/regression_cleaned_data/RND_regression_all_2pt_7d_20260312.csv"
    
    df_2pt_7d_aligned = align_dates(
        reference_file=reference_1pt_7d,
        target_file=target_2pt_7d,
        output_file=output_2pt_7d_aligned
    )
    
    print("\n✓ 資料清理與對齊完成！")