import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import shapiro, kstest, boxcox
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score, classification_report
)
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_PATH = "D:/Aurel/2025/raw/"

class DataPreprocessor:

    def __init__(self):
        self.main_data = None
        self.od_matrix = None
        self.od_yearly = {}
        self.imputation_results = {}
        self.outlier_results = {}
        self.transformation_results = {}
        self.stl_outlier_results = None  

    def load_data(self):
        print("加载数据")
        try:
            self.main_data = pd.read_csv(BASE_PATH + 'main_data_advanced.csv')
            self.od_matrix = pd.read_csv(BASE_PATH + 'od_matrix.csv')

            for year in range(2019, 2024):
                filename = f'od_matrix_{year}.csv'
                self.od_yearly[year] = pd.read_csv(BASE_PATH + filename)

        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise

    def descriptive_statistics(self):
        numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
        desc_stats = self.main_data[numeric_cols].describe().T
        desc_stats['missing_rate'] = self.main_data[numeric_cols].isnull().sum() / len(self.main_data)
        desc_stats['skewness'] = self.main_data[numeric_cols].skew()
        desc_stats['kurtosis'] = self.main_data[numeric_cols].kurtosis()
        desc_stats.to_csv('descriptive_statistics.csv', encoding='utf-8-sig')
        return desc_stats

    def compare_imputation_methods(self):
        print("缺失值插补对比")
        numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
        data_numeric = self.main_data[numeric_cols].copy()
        mask = data_numeric.isnull()
        complete_data = data_numeric.dropna()

        if len(complete_data) < 100:
            test_data = data_numeric.copy()
        else:
            test_data = complete_data.sample(min(100, len(complete_data)), random_state=42)

        np.random.seed(42)
        missing_mask = np.random.random(test_data.shape) < 0.1
        test_data_missing = test_data.copy()
        test_data_missing[missing_mask] = np.nan

        methods = {}
        methods['Mean'] = test_data_missing.fillna(test_data_missing.mean())
        methods['Median'] = test_data_missing.fillna(test_data_missing.median())

        knn_imputer = KNNImputer(n_neighbors=5)
        methods['KNN'] = pd.DataFrame(
            knn_imputer.fit_transform(test_data_missing),
            columns=test_data_missing.columns,
            index=test_data_missing.index
        )

        mice_imputer = IterativeImputer(max_iter=10, random_state=42)
        methods['MICE'] = pd.DataFrame(
            mice_imputer.fit_transform(test_data_missing),
            columns=test_data_missing.columns,
            index=test_data_missing.index
        )

        methods['Forward_Fill'] = test_data_missing.fillna(method='ffill').fillna(method='bfill')

        results = []
        for method_name, imputed_data in methods.items():
            rmse = np.sqrt(np.mean((test_data[missing_mask] - imputed_data[missing_mask]) ** 2))
            mae = np.mean(np.abs(test_data[missing_mask] - imputed_data[missing_mask]))
            results.append({'Method': method_name, 'RMSE': rmse, 'MAE': mae, 'Time': 'N/A'})

        results_df = pd.DataFrame(results)
        results_df.to_csv('imputation_comparison.csv', index=False, encoding='utf-8-sig')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        results_df.plot(x='Method', y='RMSE', kind='bar', ax=axes[0], legend=False)
        results_df.plot(x='Method', y='MAE', kind='bar', ax=axes[1], legend=False)
        plt.tight_layout()
        plt.savefig('imputation_comparison.png', dpi=300)

        self.imputation_results = results_df
        return results_df, methods

    def detect_outliers(self):
        print("异常值检测")
        numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
        test_cols = numeric_cols[:5]
        outlier_summary = []

        for col in test_cols:
            data = self.main_data[col].dropna()
            if len(data) == 0:
                continue
            col_results = {'Variable': col}
            z_scores = np.abs(stats.zscore(data))
            col_results['Grubbs'] = (z_scores > 3).sum()

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            col_results['IQR'] = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            col_results['IsolationForest'] = (iso_forest.fit_predict(data.values.reshape(-1, 1)) == -1).sum()

            lof = LocalOutlierFactor(contamination=0.1)
            col_results['LOF'] = (lof.fit_predict(data.values.reshape(-1, 1)) == -1).sum()

            outlier_summary.append(col_results)

        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df.to_csv('outlier_detection_report.csv', index=False, encoding='utf-8-sig')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, col in enumerate(test_cols[:4]):
            ax = axes[idx // 2, idx % 2]
            data = self.main_data[col].dropna()
            ax.boxplot(data, vert=False)
            ax.set_title(col)

        plt.tight_layout()
        plt.savefig('outlier_detection_boxplots.png', dpi=300)

        self.outlier_results = outlier_df
        return outlier_df


    def detect_outliers_stl(self, ts_col='跨境数据传输总量_TB', date_col='年份', period=12):
        print("STL 异常值检测")

        if ts_col not in self.main_data.columns or date_col not in self.main_data.columns:
            print("STL 检测失败：缺少时间列或目标列")
            return None

        df = self.main_data.copy()
        df[date_col] = pd.to_datetime(df[date_col], format='%Y', errors='ignore')
        df = df.sort_values(date_col)
        df = df[[date_col, ts_col]].dropna()

        df = df.set_index(date_col)
        ts = df[ts_col]

        stl = STL(ts, period=period)
        result = stl.fit()

        resid = result.resid
        mad = np.median(np.abs(resid - np.median(resid)))
        threshold = 3 * mad

        outliers = np.abs(resid) > threshold
        outlier_df = pd.DataFrame({
            'value': ts,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'resid': resid,
            'is_outlier': outliers
        })

        outlier_df.to_csv("stl_outlier_results.csv", encoding="utf-8-sig")

        plt.figure(figsize=(12, 6))
        plt.plot(ts, label='原始数据')
        plt.scatter(ts.index[outliers], ts[outliers], color='red', label='异常点')
        plt.title("STL 异常值检测")
        plt.legend()
        plt.tight_layout()
        plt.savefig("stl_outlier_detection.png", dpi=300)

        self.stl_outlier_results = outlier_df
        return outlier_df

    def test_normality(self):
        print("正态性检验")
        numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
        test_cols = numeric_cols[:8]
        normality_results = []

        for col in test_cols:
            data = self.main_data[col].dropna()
            if len(data) < 3:
                continue
            shapiro_stat, shapiro_p = shapiro(data)
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            normality_results.append({
                'Variable': col,
                'Shapiro_Statistic': shapiro_stat,
                'Shapiro_p_value': shapiro_p,
                'KS_Statistic': ks_stat,
                'KS_p_value': ks_p,
                'Is_Normal_Shapiro': shapiro_p > 0.05,
                'Is_Normal_KS': ks_p > 0.05
            })

        normality_df = pd.DataFrame(normality_results)
        normality_df.to_csv('normality_test_results.csv', index=False, encoding='utf-8-sig')

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for idx, col in enumerate(test_cols[:8]):
            ax = axes[idx // 4, idx % 4]
            data = self.main_data[col].dropna()
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(col)

        plt.tight_layout()
        plt.savefig('qq_plots.png', dpi=300)

        return normality_df

    def compare_transformations(self):
        print("数据变换对比")
        numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
        test_cols = numeric_cols[:5]
        transformation_results = []

        for col in test_cols:
            data = self.main_data[col].dropna()
            if len(data) < 3 or (data <= 0).any():
                continue

            col_results = {'Variable': col}
            col_results['Original_Skew'] = stats.skew(data)

            standardized = StandardScaler().fit_transform(data.values.reshape(-1, 1)).flatten()
            col_results['Standardized_Skew'] = stats.skew(standardized)

            normalized = MinMaxScaler().fit_transform(data.values.reshape(-1, 1)).flatten()
            col_results['Normalized_Skew'] = stats.skew(normalized)

            log_transformed = np.log1p(data)
            col_results['Log_Skew'] = stats.skew(log_transformed)

            boxcox_transformed, _ = boxcox(data)
            col_results['BoxCox_Skew'] = stats.skew(boxcox_transformed)

            pt = PowerTransformer(method='yeo-johnson')
            yj_transformed = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
            col_results['YeoJohnson_Skew'] = stats.skew(yj_transformed)

            transformation_results.append(col_results)

        trans_df = pd.DataFrame(transformation_results)
        trans_df.to_csv('transformation_comparison.csv', index=False, encoding='utf-8-sig')

        self.transformation_results = trans_df
        return trans_df

    def check_multicollinearity(self):
        print("VIF检验")
        numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
        data_clean = self.main_data[numeric_cols].dropna()
        if len(data_clean) == 0:
            return None

        test_cols = numeric_cols[:10]
        X = data_clean[test_cols]

        vif_data = []
        for i, col in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data.append({'Variable': col, 'VIF': vif, 'Has_Multicollinearity': vif > 10})
            except:
                vif_data.append({'Variable': col, 'VIF': np.nan, 'Has_Multicollinearity': False})

        vif_df = pd.DataFrame(vif_data)
        vif_df.to_csv('vif_results.csv', index=False, encoding='utf-8-sig')
        return vif_df

    def validate_consistency(self):
        print("一致性验证")
        consistency_report = []

        years = sorted(self.main_data['年份'].unique())
        year_range = range(min(years), max(years) + 1)
        missing_years = set(year_range) - set(years)
        consistency_report.append({
            'Check_Type': 'Time_Continuity',
            'Status': 'Pass' if len(missing_years) == 0 else 'Fail',
            'Details': f'Missing years: {missing_years}' if missing_years else 'Complete'
        })

        if '入境数据量_TB' in self.main_data.columns and '出境数据量_TB' in self.main_data.columns:
            if '跨境数据传输总量_TB' in self.main_data.columns:
                total = self.main_data['入境数据量_TB'] + self.main_data['出境数据量_TB']
                diff = np.abs(total - self.main_data['跨境数据传输总量_TB'])
                logical_error = (diff > 1).sum()
                consistency_report.append({
                    'Check_Type': 'Logical_Relation',
                    'Status': 'Pass' if logical_error == 0 else 'Warning',
                    'Details': f'{logical_error} records with logical errors'
                })

        cities_main = set(self.main_data['城市'].unique())
        cities_od = set(self.od_matrix['起点城市'].unique()) | set(self.od_matrix['终点城市'].unique())
        missing_cities = cities_od - cities_main
        consistency_report.append({
            'Check_Type': 'Cross_Table_Validation',
            'Status': 'Pass' if len(missing_cities) == 0 else 'Warning',
            'Details': f'Cities in OD but not in main: {missing_cities}' if missing_cities else 'Consistent'
        })

        consistency_df = pd.DataFrame(consistency_report)
        consistency_df.to_csv('consistency_validation_report.csv', index=False, encoding='utf-8-sig')
        return consistency_df

    def generate_quality_report(self):
        report = []
        report.append("数据质量评估报告")
        report.append(f"生成时间: {pd.Timestamp.now()}")
        report.append(f"主数据表: {self.main_data.shape}")
        report.append(f"OD矩阵表: {self.od_matrix.shape}")

        missing_rate = self.main_data.isnull().sum() / len(self.main_data)
        high_missing = missing_rate[missing_rate > 0.1]
        report.append(f"高缺失率变量(>10%): {len(high_missing)}")

        if hasattr(self, 'outlier_results'):
            avg_outliers = self.outlier_results.iloc[:, 1:].mean().mean()
            report.append(f"平均异常值比例: {avg_outliers:.2f}%")

        report.append("建议: 使用MICE/KNN、IQR/IsolationForest、Yeo-Johnson、移除高VIF变量")
        report_text = "\n".join(report)

        with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        return report_text

    def run_full_pipeline(self):
        print("开始执行")
        self.load_data()
        self.descriptive_statistics()
        self.compare_imputation_methods()
        self.detect_outliers()
        self.detect_outliers_stl()  
        self.test_normality()
        self.compare_transformations()
        self.check_multicollinearity()
        self.validate_consistency()
        self.generate_quality_report()
        print("完成")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_full_pipeline()


class ExploratoryDataAnalysis:
    
    def __init__(self, main_data, od_matrix, od_yearly):
        self.main_data = main_data
        self.od_matrix = od_matrix
        self.od_yearly = od_yearly
        self.results = {}


class ExploratoryDataAnalysis:
    
    def __init__(self, main_data, od_matrix, od_yearly):
        self.main_data = main_data
        self.od_matrix = od_matrix
        self.od_yearly = od_yearly
        self.results = {}

def descriptive_statistics_advanced(self):
    """描述性统计分析"""
    print("=" * 80)
    print("描述性统计分析")
    print("=" * 80)
    
    numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
    key_indicators = [
        '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元',
        '数据交易额_亿元', '5G基站数量', '算力规模_PFLOPS'
    ]
    
    available_indicators = [col for col in key_indicators if col in numeric_cols]
    
    stats_results = []
    for col in available_indicators:
        data = self.main_data[col].dropna()
        if len(data) == 0:
            continue
            
        stats_dict = {
            'Indicator': col,
            'Count': len(data),
            'Mean': np.mean(data),
            'Std': np.std(data),
            'CV': np.std(data) / np.mean(data) if np.mean(data) != 0 else np.nan,
            'Min': np.min(data),
            'Q1': np.percentile(data, 25),
            'Median': np.median(data),
            'Q3': np.percentile(data, 75),
            'Max': np.max(data),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data)
        }
        
        if len(data) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            stats_dict['Shapiro_p'] = shapiro_p
            stats_dict['Is_Normal'] = shapiro_p > 0.05
        
        stats_results.append(stats_dict)
    
    stats_df = pd.DataFrame(stats_results)
    print("\n核心指标描述性统计:")
    print(stats_df.round(3))
    
    stats_df.to_csv('chapter5_descriptive_stats.csv', index=False, encoding='utf-8-sig')
    
    self.results['descriptive_stats'] = stats_df
    return stats_df

ExploratoryDataAnalysis.descriptive_statistics_advanced = descriptive_statistics_advanced

def temporal_trend_analysis(self):
    print("\n时间趋势特征分析")
    print("-" * 80)
    
    if '年份' not in self.main_data.columns:
        print("警告: 缺少年份字段")
        return None
    
    yearly_data = self.main_data.groupby('年份').agg({
        '跨境数据传输总量_TB': 'sum',
        'GDP_亿元': 'sum',
        '数字经济核心产业增加值_亿元': 'sum',
        '数据交易额_亿元': 'sum'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    indicators = [
        ('跨境数据传输总量_TB', '跨境数据传输总量(TB)'),
        ('GDP_亿元', 'GDP(亿元)'),
        ('数字经济核心产业增加值_亿元', '数字经济增加值(亿元)'),
        ('数据交易额_亿元', '数据交易额(亿元)')
    ]
    
    for idx, (col, title) in enumerate(indicators):
        if col not in yearly_data.columns:
            continue
            
        ax = axes[idx // 2, idx % 2]
        ax.plot(yearly_data['年份'], yearly_data[col], marker='o', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('年份')
        ax.set_ylabel('数值')
        ax.grid(True, alpha=0.3)
        
        z = np.polyfit(yearly_data['年份'], yearly_data[col], 1)
        p = np.poly1d(z)
        ax.plot(yearly_data['年份'], p(yearly_data['年份']), 
               linestyle='--', color='red', alpha=0.7, label='趋势线')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('temporal_trend_analysis.png', dpi=300, bbox_inches='tight')
    print("时间趋势图已保存: temporal_trend_analysis.png")
    
    growth_rates = []
    for col, title in indicators:
        if col in yearly_data.columns and len(yearly_data[col]) > 1:
            start_val = yearly_data[col].iloc[0]
            end_val = yearly_data[col].iloc[-1]
            years = len(yearly_data) - 1
            if start_val > 0:
                cagr = ((end_val / start_val) ** (1/years) - 1) * 100
                growth_rates.append({
                    'Indicator': title,
                    'Start_Value': start_val,
                    'End_Value': end_val,
                    'CAGR_%': cagr
                })
    
    growth_df = pd.DataFrame(growth_rates)
    print("\n年均复合增长率(CAGR):")
    print(growth_df.round(2))
    
    growth_df.to_csv('temporal_growth_rates.csv', index=False, encoding='utf-8-sig')
    
    self.results['temporal_trends'] = yearly_data
    self.results['growth_rates'] = growth_df
    return yearly_data, growth_df

ExploratoryDataAnalysis.temporal_trend_analysis = temporal_trend_analysis


def time_series_decomposition(self):
    print("\n时间序列分解(Trend-Seasonal-Residual)")
    print("-" * 80)
    
    if '年份' not in self.main_data.columns or '城市' not in self.main_data.columns:
        print("警告: 缺少必要字段")
        return None
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns:
        print(f"警告: 缺少{target_col}字段")
        return None
    
    cities = self.main_data['城市'].unique()[:3]
    
    fig, axes = plt.subplots(len(cities), 3, figsize=(15, 4*len(cities)))
    if len(cities) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, city in enumerate(cities):
        city_data = self.main_data[self.main_data['城市'] == city].sort_values('年份')
        
        if len(city_data) < 4:
            continue
        
        ts_data = city_data.set_index('年份')[target_col].dropna()
        
        if len(ts_data) >= 4:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(ts_data, model='additive', period=1)
                
                decomposition.trend.plot(ax=axes[idx, 0])
                axes[idx, 0].set_title(f'{city} - 趋势')
                axes[idx, 0].set_ylabel('数值')
                
                decomposition.seasonal.plot(ax=axes[idx, 1])
                axes[idx, 1].set_title(f'{city} - 季节性')
                axes[idx, 1].set_ylabel('数值')
                
                decomposition.resid.plot(ax=axes[idx, 2])
                axes[idx, 2].set_title(f'{city} - 残差')
                axes[idx, 2].set_ylabel('数值')
                
            except Exception as e:
                print(f"{city}分解失败: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
    print("时间序列分解图已保存: time_series_decomposition.png")
    
    return None

ExploratoryDataAnalysis.time_series_decomposition = time_series_decomposition


def spatial_distribution_analysis(self):
    print("\n空间分布特征分析")
    print("-" * 80)
    
    if '城市' not in self.main_data.columns:
        print("警告: 缺少城市字段")
        return None
    
    latest_year = self.main_data['年份'].max()
    latest_data = self.main_data[self.main_data['年份'] == latest_year]
    
    key_indicators = [
        '跨境数据传输总量_TB', 'GDP_亿元', 
        '数字经济核心产业增加值_亿元', '数据交易额_亿元'
    ]
    
    available_indicators = [col for col in key_indicators if col in latest_data.columns]
    
    city_summary = latest_data.groupby('城市')[available_indicators].sum()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, col in enumerate(available_indicators[:4]):
        ax = axes[idx // 2, idx % 2]
        
        top_cities = city_summary[col].nlargest(10)
        top_cities.plot(kind='barh', ax=ax)
        ax.set_title(f'{col}空间分布({latest_year}年)')
        ax.set_xlabel('数值')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('spatial_distribution_barplot.png', dpi=300, bbox_inches='tight')
    print("空间分布柱状图已保存: spatial_distribution_barplot.png")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    city_data_normalized = (city_summary - city_summary.min()) / (city_summary.max() - city_summary.min())
    
    sns.heatmap(city_data_normalized.T, annot=False, cmap='YlOrRd', 
               cbar_kws={'label': '标准化值'}, ax=ax)
    ax.set_title(f'城市指标空间分布热力图({latest_year}年)')
    ax.set_xlabel('城市')
    ax.set_ylabel('指标')
    
    plt.tight_layout()
    plt.savefig('spatial_distribution_heatmap.png', dpi=300, bbox_inches='tight')
    print("空间分布热力图已保存: spatial_distribution_heatmap.png")
    
    city_summary.to_csv('spatial_distribution_summary.csv', encoding='utf-8-sig')
    
    self.results['spatial_distribution'] = city_summary
    return city_summary

ExploratoryDataAnalysis.spatial_distribution_analysis = spatial_distribution_analysis



def city_disparity_analysis(self):
    print("\n城市间差异分析")
    print("-" * 80)
    
    if '城市' not in self.main_data.columns:
        print("警告: 缺少城市字段")
        return None
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns:
        print(f"警告: 缺少{target_col}字段")
        return None
    
    city_stats = self.main_data.groupby('城市')[target_col].agg([
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('CV', lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan)
    ]).round(3)
    
    print("\n城市间差异统计:")
    print(city_stats)
    
    city_stats.to_csv('city_disparity_statistics.csv', encoding='utf-8-sig')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    city_means = self.main_data.groupby('城市')[target_col].mean().sort_values(ascending=False)
    city_means.plot(kind='bar', ax=axes[0])
    axes[0].set_title('城市平均数据传输量')
    axes[0].set_ylabel(target_col)
    axes[0].tick_params(axis='x', rotation=45)
    
    city_cv = city_stats['CV'].sort_values(ascending=False)
    city_cv.plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('城市变异系数(CV)')
    axes[1].set_ylabel('变异系数')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('city_disparity_analysis.png', dpi=300, bbox_inches='tight')
    print("城市差异分析图已保存: city_disparity_analysis.png")
    
    self.results['city_disparity'] = city_stats
    return city_stats

ExploratoryDataAnalysis.city_disparity_analysis = city_disparity_analysis


def calculate_gini_coefficient(self):
    print("\n基尼系数 - 城市间数据流动不平等")
    print("-" * 80)
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns:
        print(f"警告: 缺少{target_col}字段")
        return None
    
    yearly_gini = []
    for year in sorted(self.main_data['年份'].unique()):
        year_data = self.main_data[self.main_data['年份'] == year][target_col].dropna()
        
        if len(year_data) == 0:
            continue
        
        sorted_data = np.sort(year_data)
        n = len(sorted_data)
        cumsum = np.cumsum(sorted_data)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n
        
        yearly_gini.append({'Year': year, 'Gini': gini})
    
    gini_df = pd.DataFrame(yearly_gini)
    print("\n历年基尼系数:")
    print(gini_df)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(gini_df['Year'], gini_df['Gini'], marker='o', linewidth=2)
    ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='警戒线(0.4)')
    ax.set_title('基尼系数演化趋势')
    ax.set_xlabel('年份')
    ax.set_ylabel('基尼系数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gini_coefficient_trend.png', dpi=300, bbox_inches='tight')
    print("基尼系数趋势图已保存: gini_coefficient_trend.png")
    
    gini_df.to_csv('gini_coefficient.csv', index=False, encoding='utf-8-sig')
    
    self.results['gini'] = gini_df
    return gini_df

ExploratoryDataAnalysis.calculate_gini_coefficient = calculate_gini_coefficient



def calculate_theil_index(self):
    print("\n泰尔指数 - 区域内外差异分解")
    print("-" * 80)
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns or '城市' not in self.main_data.columns:
        print("警告: 缺少必要字段")
        return None
    
    bay_area_cities = ['香港', '澳门', '广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆']
    
    self.main_data['Region'] = self.main_data['城市'].apply(
        lambda x: '大湾区' if x in bay_area_cities else '其他'
    )
    
    yearly_theil = []
    for year in sorted(self.main_data['年份'].unique()):
        year_data = self.main_data[self.main_data['年份'] == year]
        
        total_mean = year_data[target_col].mean()
        total_sum = year_data[target_col].sum()
        
        if total_sum == 0:
            continue
        
        theil_within = 0
        theil_between = 0
        
        for region in year_data['Region'].unique():
            region_data = year_data[year_data['Region'] == region][target_col]
            region_mean = region_data.mean()
            region_sum = region_data.sum()
            region_share = region_sum / total_sum
            
            if region_share > 0 and region_mean > 0:
                theil_between += region_share * np.log(region_mean / total_mean)
                
                for val in region_data:
                    if val > 0:
                        theil_within += (val / total_sum) * np.log(val / region_mean)
        
        yearly_theil.append({
            'Year': year,
            'Theil_Total': theil_within + theil_between,
            'Theil_Within': theil_within,
            'Theil_Between': theil_between
        })
    
    theil_df = pd.DataFrame(yearly_theil)
    print("\n历年泰尔指数:")
    print(theil_df)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(theil_df['Year'], theil_df['Theil_Total'], marker='o', label='总体不平等')
    ax.plot(theil_df['Year'], theil_df['Theil_Within'], marker='s', label='区域内不平等')
    ax.plot(theil_df['Year'], theil_df['Theil_Between'], marker='^', label='区域间不平等')
    ax.set_title('泰尔指数分解')
    ax.set_xlabel('年份')
    ax.set_ylabel('泰尔指数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('theil_index_decomposition.png', dpi=300, bbox_inches='tight')
    print("泰尔指数图已保存: theil_index_decomposition.png")
    
    theil_df.to_csv('theil_index.csv', index=False, encoding='utf-8-sig')
    
    self.results['theil'] = theil_df
    return theil_df

ExploratoryDataAnalysis.calculate_theil_index = calculate_theil_index


def plot_lorenz_curve(self):
    print("\n洛伦兹曲线 - 不平等程度可视化")
    print("-" * 80)
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns:
        print(f"警告: 缺少{target_col}字段")
        return None
    
    latest_year = self.main_data['年份'].max()
    latest_data = self.main_data[self.main_data['年份'] == latest_year][target_col].dropna()
    
    sorted_data = np.sort(latest_data)
    n = len(sorted_data)
    cum_data = np.cumsum(sorted_data)
    cum_data_norm = cum_data / cum_data[-1]
    cum_pop = np.arange(1, n+1) / n
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='完全平等线')
    ax.plot(cum_pop, cum_data_norm, linewidth=2, label='洛伦兹曲线')
    ax.fill_between(cum_pop, cum_data_norm, alpha=0.3)
    ax.set_title(f'洛伦兹曲线 ({latest_year}年)')
    ax.set_xlabel('累计人口比例')
    ax.set_ylabel('累计数据传输量比例')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lorenz_curve.png', dpi=300, bbox_inches='tight')
    print("洛伦兹曲线已保存: lorenz_curve.png")
    
    return None

ExploratoryDataAnalysis.plot_lorenz_curve = plot_lorenz_curve


def correlation_analysis(self):
    print("\n" + "=" * 80)
    print("相关性分析")
    print("=" * 80)
    
    numeric_cols = self.main_data.select_dtypes(include=[np.number]).columns
    key_vars = [
        '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元',
        '数据交易额_亿元', '5G基站数量', '算力规模_PFLOPS',
        '研发经费投入_亿元', '高新技术企业数'
    ]
    
    available_vars = [col for col in key_vars if col in numeric_cols]
    corr_data = self.main_data[available_vars].dropna()
    
    if len(corr_data) == 0:
        print("警告: 无足够数据进行相关性分析")
        return None
    
    pearson_corr = corr_data.corr(method='pearson')
    spearman_corr = corr_data.corr(method='spearman')
    kendall_corr = corr_data.corr(method='kendall')
    
    print("\nPearson相关系数矩阵:")
    print(pearson_corr.round(3))
    
    pearson_corr.to_csv('correlation_pearson.csv', encoding='utf-8-sig')
    spearman_corr.to_csv('correlation_spearman.csv', encoding='utf-8-sig')
    kendall_corr.to_csv('correlation_kendall.csv', encoding='utf-8-sig')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, ax=axes[0], cbar_kws={'label': '相关系数'})
    axes[0].set_title('Pearson相关系数')
    
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, ax=axes[1], cbar_kws={'label': '相关系数'})
    axes[1].set_title('Spearman相关系数')
    
    sns.heatmap(kendall_corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, ax=axes[2], cbar_kws={'label': '相关系数'})
    axes[2].set_title('Kendall相关系数')
    
    plt.tight_layout()
    plt.savefig('correlation_matrices.png', dpi=300, bbox_inches='tight')
    print("\n相关系数矩阵图已保存: correlation_matrices.png")
    
    self.results['correlations'] = {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'kendall': kendall_corr
    }
    
    return pearson_corr, spearman_corr, kendall_corr

ExploratoryDataAnalysis.correlation_analysis = correlation_analysis


def partial_correlation_analysis(self):
    print("\n偏相关分析(控制GDP影响)")
    print("-" * 80)
    
    if 'GDP_亿元' not in self.main_data.columns:
        print("警告: 缺少GDP字段")
        return None
    
    key_vars = [
        '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元',
        '数据交易额_亿元', '5G基站数量'
    ]
    
    available_vars = [col for col in key_vars if col in self.main_data.columns]
    data_clean = self.main_data[available_vars].dropna()
    
    if len(data_clean) < 30:
        print("警告: 数据量不足")
        return None
    
    from scipy.stats import pearsonr
    
    def partial_corr(data, x, y, z):
        """计算偏相关系数"""
        df = data[[x, y, z]].dropna()
        
        rx_z = df[x] - np.polyval(np.polyfit(df[z], df[x], 1), df[z])
        ry_z = df[y] - np.polyval(np.polyfit(df[z], df[y], 1), df[z])
        
        corr, p_value = pearsonr(rx_z, ry_z)
        return corr, p_value
    
    control_var = 'GDP_亿元'
    other_vars = [v for v in available_vars if v != control_var]
    
    partial_corr_results = []
    for i, var1 in enumerate(other_vars):
        for var2 in other_vars[i+1:]:
            corr, p_val = partial_corr(data_clean, var1, var2, control_var)
            partial_corr_results.append({
                'Variable_1': var1,
                'Variable_2': var2,
                'Control': control_var,
                'Partial_Corr': corr,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })
    
    partial_corr_df = pd.DataFrame(partial_corr_results)
    print("\n偏相关分析结果:")
    print(partial_corr_df.round(4))
    
    partial_corr_df.to_csv('partial_correlation.csv', index=False, encoding='utf-8-sig')
    
    self.results['partial_correlation'] = partial_corr_df
    return partial_corr_df

ExploratoryDataAnalysis.partial_correlation_analysis = partial_correlation_analysis

def distance_correlation(self):
    print("\n距离相关性分析(非线性关系)")
    print("-" * 80)
    
    key_vars = [
        '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元'
    ]
    
    available_vars = [col for col in key_vars if col in self.main_data.columns]
    data_clean = self.main_data[available_vars].dropna()
    
    if len(data_clean) < 10:
        print("警告: 数据量不足")
        return None
    
    def distance_corr(x, y):
        n = len(x)
        a = np.abs(np.subtract.outer(x, x))
        b = np.abs(np.subtract.outer(y, y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor
    
    dist_corr_results = []
    
    for i, var1 in enumerate(available_vars):
        for var2 in available_vars[i+1:]:
            x = data_clean[var1].values
            y = data_clean[var2].values
            dcor = distance_corr(x, y)
            dist_corr_results.append({
                'Variable_1': var1,
                'Variable_2': var2,
                'Distance_Corr': dcor
            })

    dist_corr_df = pd.DataFrame(dist_corr_results)
    print("\n距离相关系数:")
    print(dist_corr_df.round(4))

    dist_corr_df.to_csv('distance_correlation.csv', index=False, encoding='utf-8-sig')

    self.results['distance_correlation'] = dist_corr_df
    return dist_corr_df

ExploratoryDataAnalysis.distance_correlation = distance_correlation

def maximal_information_coefficient(self):
    print("\n最大信息系数(MIC) - 复杂关系发现")
    print("-" * 80)
    
    key_vars = [
        '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元',
        '数据交易额_亿元', '5G基站数量'
    ]
    
    available_vars = [col for col in key_vars if col in self.main_data.columns]
    data_clean = self.main_data[available_vars].dropna()
    
    if len(data_clean) < 10:
        print("警告: 数据量不足")
        return None
    
    def qcut_codes(arr, q):
        """将连续变量离散化为q个分位数"""
        s = pd.Series(arr)
        try:
            cats = pd.qcut(s, q=q, duplicates='drop')
            return cats.cat.codes.to_numpy()
        except:
            cats = pd.cut(s, bins=min(q, len(np.unique(s))), labels=False)
            return cats
    
    def compute_mi(x, y, bins=10):
        try:
            x_binned = qcut_codes(x, bins).reshape(-1, 1)
            y_binned = qcut_codes(y, bins)
            
            mi = mutual_info_regression(x_binned, y_binned, discrete_features=True)[0]
            return mi
        except Exception as e:
            print(f"计算互信息时出错: {e}")
            return np.nan
    
    results = []
    for i, var1 in enumerate(available_vars):
        for var2 in available_vars[i+1:]:
            x = data_clean[var1].values
            y = data_clean[var2].values
            
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                print(f"警告: {var1}或{var2}数据变化不足")
                continue
            
            mi_value = compute_mi(x, y, bins=min(10, len(data_clean)//5))
            
            if not np.isnan(mi_value):
                results.append({
                    'Variable_1': var1,
                    'Variable_2': var2,
                    'Mutual_Information': mi_value
                })
    
    if not results:
        print("警告: 没有有效的互信息计算结果")
        return None
    
    mi_df = pd.DataFrame(results)
    print("\n基于互信息的复杂关系分析结果：")
    print(mi_df.round(4))
    
    mi_df.to_csv("mic_mutual_information_analysis.csv", index=False, encoding="utf-8-sig")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(mi_df) > 0:
        mi_df_sorted = mi_df.sort_values('Mutual_Information', ascending=True)
        y_pos = np.arange(len(mi_df_sorted))
        
        bars = ax.barh(y_pos, mi_df_sorted['Mutual_Information'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{row['Variable_1']} vs {row['Variable_2']}" for _, row in mi_df_sorted.iterrows()],
            fontsize=9
        )
        
        for i, (_, row) in enumerate(mi_df_sorted.iterrows()):
            ax.text(row['Mutual_Information'] + 0.001, i, f'{row["Mutual_Information"]:.4f}', 
                   va='center', fontsize=8)
        
        ax.set_xlabel("互信息值")
        ax.set_title("变量间互信息分析")
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("mic_analysis.png", dpi=300, bbox_inches='tight')
    print("MIC分析图已保存: mic_analysis.png")
    
    self.results['mic'] = mi_df
    return mi_df

ExploratoryDataAnalysis.maximal_information_coefficient = maximal_information_coefficient

def multiple_testing_correction(self):
    print("\n多重检验校正(Bonferroni & FDR)")
    print("-" * 80)
    
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    
    key_vars = [
        '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元',
        '数据交易额_亿元', '5G基站数量', '算力规模_PFLOPS'
    ]
    
    available_vars = [col for col in key_vars if col in self.main_data.columns]
    data_clean = self.main_data[available_vars].dropna()
    
    if len(data_clean) < 10:
        print("警告: 数据量不足")
        return None
    
    p_values = []
    corr_results = []
    
    for i, var1 in enumerate(available_vars):
        for var2 in available_vars[i+1:]:
            x = data_clean[var1].values
            y = data_clean[var2].values
            
            corr, p_val = pearsonr(x, y)
            p_values.append(p_val)
            corr_results.append({
                'Variable_1': var1,
                'Variable_2': var2,
                'Correlation': corr,
                'P_Value': p_val
            })
    
    bonferroni_reject, bonferroni_pvals, _, _ = multipletests(
        p_values, alpha=0.05, method='bonferroni'
    )
    
    fdr_reject, fdr_pvals, _, _ = multipletests(
        p_values, alpha=0.05, method='fdr_bh'
    )
    
    for i, result in enumerate(corr_results):
        result['Bonferroni_Corrected_P'] = bonferroni_pvals[i]
        result['Bonferroni_Significant'] = bonferroni_reject[i]
        result['FDR_Corrected_P'] = fdr_pvals[i]
        result['FDR_Significant'] = fdr_reject[i]
    
    correction_df = pd.DataFrame(corr_results)
    print("\n多重检验校正结果:")
    print(correction_df.round(4))
    
    print(f"\n原始显著数: {sum([r['P_Value'] < 0.05 for r in corr_results])}")
    print(f"Bonferroni校正后显著数: {sum(bonferroni_reject)}")
    print(f"FDR校正后显著数: {sum(fdr_reject)}")
    
    correction_df.to_csv('multiple_testing_correction.csv', index=False, encoding='utf-8-sig')
    
    self.results['multiple_testing'] = correction_df
    return correction_df

ExploratoryDataAnalysis.multiple_testing_correction = multiple_testing_correction



def spatial_autocorrelation_analysis(self):
    print("\n" + "=" * 80)
    print("空间自相关分析")
    print("=" * 80)
    
    from scipy.spatial.distance import pdist, squareform
    
    if '城市' not in self.main_data.columns:
        print("警告: 缺少城市字段")
        return None
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns:
        print(f"警告: 缺少{target_col}字段")
        return None
    
    city_coords = {
        '香港': (114.1733, 22.3200),
        '澳门': (113.5439, 22.1987),
        '广州': (113.2644, 23.1291),
        '深圳': (114.0579, 22.5431),
        '珠海': (113.5767, 22.2769),
        '佛山': (113.1220, 23.0218),
        '惠州': (114.4152, 23.1115),
        '东莞': (113.7518, 23.0209),
        '中山': (113.3826, 22.5171),
        '江门': (113.0816, 22.5789),
        '肇庆': (112.4656, 23.0473)
    }
    
    latest_year = self.main_data['年份'].max()
    latest_data = self.main_data[self.main_data['年份'] == latest_year]
    
    city_values = {}
    for city in latest_data['城市'].unique():
        if city in city_coords:
            value = latest_data[latest_data['城市'] == city][target_col].mean()
            if not np.isnan(value):
                city_values[city] = value
    
    if len(city_values) < 3:
        print("警告: 城市数量不足")
        return None
    
    cities = list(city_values.keys())
    coords = np.array([city_coords[city] for city in cities])
    values = np.array([city_values[city] for city in cities])
    
    dist_matrix = squareform(pdist(coords))
    threshold = np.percentile(dist_matrix[dist_matrix > 0], 50)
    
    W = (dist_matrix > 0) & (dist_matrix <= threshold)
    W = W.astype(float)
    row_sums = W.sum(axis=1)
    W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0, None]
    
    n = len(values)
    mean_val = np.mean(values)
    
    numerator = 0
    denominator = np.sum((values - mean_val) ** 2)
    
    for i in range(n):
        for j in range(n):
            numerator += W[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
    
    S0 = np.sum(W)
    moran_i = (n / S0) * (numerator / denominator)
    
    E_I = -1 / (n - 1)
    
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    
    b2 = (n * np.sum((values - mean_val) ** 4)) / (np.sum((values - mean_val) ** 2) ** 2)
    
    Var_I = ((n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * (S0 ** 2))) - 
             (b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * (S0 ** 2)))) / \
            ((n - 1) * (n - 2) * (n - 3) * (S0 ** 2)) - (E_I ** 2)
    
    z_score = (moran_i - E_I) / np.sqrt(Var_I)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"\n全局Moran's I统计量: {moran_i:.4f}")
    print(f"期望值: {E_I:.4f}")
    print(f"Z得分: {z_score:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        if moran_i > 0:
            print("结论: 存在显著正空间自相关(集聚)")
        else:
            print("结论: 存在显著负空间自相关(离散)")
    else:
        print("结论: 不存在显著空间自相关")
    
    moran_results = {
        'Moran_I': moran_i,
        'Expected_I': E_I,
        'Z_Score': z_score,
        'P_Value': p_value,
        'Significant': p_value < 0.05,
        'Pattern': 'Clustered' if (p_value < 0.05 and moran_i > 0) else 
                  ('Dispersed' if (p_value < 0.05 and moran_i < 0) else 'Random')
    }
    
    moran_df = pd.DataFrame([moran_results])
    moran_df.to_csv('moran_i_results.csv', index=False, encoding='utf-8-sig')
    
    self.results['moran_i'] = moran_results
    return moran_results, W, cities, values

ExploratoryDataAnalysis.spatial_autocorrelation_analysis = spatial_autocorrelation_analysis




def lisa_cluster_analysis(self):
    print("\n5.3.2 LISA Cluster Analysis - 局部空间自相关")
    print("-" * 80)

    try:
        import libpysal
        import esda
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 libpysal 和 esda 库")
        return None

    df = self.main_data.copy()
    required_cols = ['城市', '跨境数据传输总量_TB']
    if not all(col in df.columns for col in required_cols):
        print("警告: 缺少必要字段")
        return None

    latest_year = df['年份'].max()
    df_latest = df[df['年份'] == latest_year].reset_index(drop=True)

    city_order = df_latest['城市'].tolist()
    values = df_latest['跨境数据传输总量_TB'].values

    w = libpysal.weights.DistanceBand.from_array(
        df_latest[['经度', '纬度']].values,
        threshold=1.5
    )
    w.transform = 'r'

    mi_local = esda.Moran_Local(values, w)

    df_latest['Local_I'] = mi_local.Is
    df_latest['p_value'] = mi_local.p_sim
    df_latest['Z_Score'] = mi_local.z_sim

    df_latest['Cluster_Type'] = 'Not Significant'
    df_latest.loc[(mi_local.q == 1) & (mi_local.p_sim < 0.05), 'Cluster_Type'] = 'HH'
    df_latest.loc[(mi_local.q == 2) & (mi_local.p_sim < 0.05), 'Cluster_Type'] = 'LH'
    df_latest.loc[(mi_local.q == 3) & (mi_local.p_sim < 0.05), 'Cluster_Type'] = 'LL'
    df_latest.loc[(mi_local.q == 4) & (mi_local.p_sim < 0.05), 'Cluster_Type'] = 'HL'

    self.results['lisa'] = df_latest

    plt.figure(figsize=(10, 8))
    import numpy as np

    z = (values - values.mean()) / values.std()
    w_z = mi_local.z_sim

    plt.scatter(z, w_z, s=80, c='gray', alpha=0.6)

    for i, city in enumerate(df_latest['城市']):
        plt.text(z[i] * 1.02, w_z[i] * 1.02, city, fontsize=10)

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    ax = plt.gca()
    ax.set_xlabel('标准化值')
    ax.set_ylabel('LISA指数')
    ax.set_title('LISA聚类散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("5.3.2_LISA_cluster_scatter.png", dpi=300)
    plt.close()

    print("LISA聚类散点图已生成: 5.3.2_LISA_cluster_scatter.png")
    return df_latest

ExploratoryDataAnalysis.lisa_cluster_analysis = lisa_cluster_analysis


def lisa_cluster_analysis(self):
    print("\n5.3.2 LISA Cluster Analysis - 局部空间自相关")
    print("-" * 80)

    df = self.main_data.copy()
    required_cols = ['城市', '跨境数据传输总量_TB']
    
    if '经度' not in df.columns or '纬度' not in df.columns:
        print("警告: 缺少经度/纬度字段，使用城市坐标字典")
        # 使用预定义的城市坐标
        city_coords = {
            '香港': (114.1733, 22.3200),
            '澳门': (113.5439, 22.1987),
            '广州': (113.2644, 23.1291),
            '深圳': (114.0579, 22.5431),
            '珠海': (113.5767, 22.2769),
            '佛山': (113.1220, 23.0218),
            '惠州': (114.4152, 23.1115),
            '东莞': (113.7518, 23.0209),
            '中山': (113.3826, 22.5171),
            '江门': (113.0816, 22.5789),
            '肇庆': (112.4656, 23.0473)
        }
    else:
        city_coords = None
    
    if not all(col in df.columns for col in required_cols):
        print("警告: 缺少必要字段")
        return None

    latest_year = df['年份'].max()
    df_latest = df[df['年份'] == latest_year].reset_index(drop=True)
    
    if city_coords:
        df_latest = df_latest[df_latest['城市'].isin(city_coords.keys())].copy()
        df_latest['经度'] = df_latest['城市'].map(lambda x: city_coords[x][0])
        df_latest['纬度'] = df_latest['城市'].map(lambda x: city_coords[x][1])
    
    if len(df_latest) < 3:
        print("警告: 有效城市数量不足")
        return None
    
    cities = df_latest['城市'].values
    coords = df_latest[['经度', '纬度']].values
    values = df_latest['跨境数据传输总量_TB'].values
    
    n = len(values)
    
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    threshold = np.percentile(dist_matrix[dist_matrix > 0], 90)
    
    W = (dist_matrix > 0) & (dist_matrix <= threshold)
    W = W.astype(float)
    
    row_sums = W.sum(axis=1)
    W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0, None]
    
    z = (values - values.mean()) / values.std()
    
    local_i = np.zeros(n)
    for i in range(n):
        w_z = np.sum(W[i] * z)
        local_i[i] = z[i] * w_z
    
    n_permutations = 999
    p_values = np.zeros(n)
    
    print("正在进行随机排列检验...")
    for i in range(n):
        count = 0
        observed = local_i[i]
        
        for _ in range(n_permutations):
            perm_z = z.copy()
            perm_indices = np.random.permutation(n)
            perm_z = perm_z[perm_indices]
            
            w_z_perm = np.sum(W[i] * perm_z)
            perm_i = z[i] * w_z_perm
            
            if perm_i >= observed:
                count += 1
        
        p_values[i] = count / n_permutations
    
    E_I = -W.sum(axis=1) / (n - 1)
    
    m2 = np.sum((z - z.mean()) ** 2) / n
    b2 = np.sum((z - z.mean()) ** 4) / n / (m2 ** 2)
    
    var_I = np.zeros(n)
    for i in range(n):
        wi = W[i].sum()
        wi2 = np.sum(W[i] ** 2)
        var_I[i] = wi2 * (n - b2) / (n - 1) - E_I[i] ** 2
    
    z_scores = (local_i - E_I) / np.sqrt(var_I)
    
    spatial_lag = W @ values
    
    cluster_types = []
    for i in range(n):
        if p_values[i] < 0.05:
            if values[i] > values.mean() and spatial_lag[i] > spatial_lag.mean():
                cluster_types.append('HH')  # 高-高
            elif values[i] < values.mean() and spatial_lag[i] < spatial_lag.mean():
                cluster_types.append('LL')  # 低-低
            elif values[i] > values.mean() and spatial_lag[i] < spatial_lag.mean():
                cluster_types.append('HL')  # 高-低
            elif values[i] < values.mean() and spatial_lag[i] > spatial_lag.mean():
                cluster_types.append('LH')  # 低-高
            else:
                cluster_types.append('Not Significant')
        else:
            cluster_types.append('Not Significant')
    
    df_latest['Local_I'] = local_i
    df_latest['p_value'] = p_values
    df_latest['Z_Score'] = z_scores
    df_latest['Cluster_Type'] = cluster_types
    
    self.results['lisa'] = df_latest
    
    plt.figure(figsize=(10, 8))
    
    w_z = (spatial_lag - spatial_lag.mean()) / spatial_lag.std()
    
    color_map = {
        'HH': 'red',
        'LL': 'blue',
        'LH': 'orange',
        'HL': 'purple',
        'Not Significant': 'gray'
    }
    colors = [color_map[ct] for ct in cluster_types]
    
    plt.scatter(z, w_z, s=100, c=colors, alpha=0.6, edgecolors='black')
    
    for i, city in enumerate(cities):
        plt.text(z[i] * 1.05, w_z[i] * 1.05, city, fontsize=9)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    
    plt.xlabel('标准化值')
    plt.ylabel('空间滞后标准化值')
    plt.title('LISA聚类散点图')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='HH (高-高)'),
        Patch(facecolor='blue', label='LL (低-低)'),
        Patch(facecolor='orange', label='LH (低-高)'),
        Patch(facecolor='purple', label='HL (高-低)'),
        Patch(facecolor='gray', label='Not Significant')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("5.3.2_LISA_cluster_scatter.png", dpi=300)
    plt.close()
    
    print("\nLISA聚类结果:")
    cluster_counts = pd.Series(cluster_types).value_counts()
    print(cluster_counts)
    
    print("\nLISA聚类散点图已生成: 5.3.2_LISA_cluster_scatter.png")
    
    return df_latest

ExploratoryDataAnalysis.lisa_cluster_analysis = lisa_cluster_analysis


def lisa_evolution_analysis(self):
    print("\n" + "=" * 80)
    print("LISA演化分析 - 多年份对比")
    print("=" * 80)
    
    df = self.main_data.copy()
    required_cols = ['城市', '年份', '跨境数据传输总量_TB']
    
    if '经度' not in df.columns or '纬度' not in df.columns:
        print("使用城市坐标字典")
        city_coords = {
            '香港': (114.1733, 22.3200),
            '澳门': (113.5439, 22.1987),
            '广州': (113.2644, 23.1291),
            '深圳': (114.0579, 22.5431),
            '珠海': (113.5767, 22.2769),
            '佛山': (113.1220, 23.0218),
            '惠州': (114.4152, 23.1115),
            '东莞': (113.7518, 23.0209),
            '中山': (113.3826, 22.5171),
            '江门': (113.0816, 22.5789),
            '肇庆': (112.4656, 23.0473)
        }
    else:
        city_coords = None
    
    if not all(col in df.columns for col in required_cols):
        print("警告: 缺少必要字段")
        return None
    
    years = sorted(df['年份'].unique())
    if len(years) < 2:
        print("警告: 需要至少2个年份的数据")
        return None
    
    all_lisa_results = []
    yearly_cluster_counts = []
    
    from scipy.spatial.distance import cdist
    
    for year in years:
        print(f"处理年份: {year}")
        df_year = df[df['年份'] == year].reset_index(drop=True)
        
        # 添加坐标
        if city_coords:
            df_year = df_year[df_year['城市'].isin(city_coords.keys())].copy()
            df_year['经度'] = df_year['城市'].map(lambda x: city_coords[x][0])
            df_year['纬度'] = df_year['城市'].map(lambda x: city_coords[x][1])
        
        if len(df_year) < 3:
            continue
        
        coords = df_year[['经度', '纬度']].values
        values = df_year['跨境数据传输总量_TB'].values
        
        if np.std(values) == 0:
            continue
        
        try:
            n = len(values)
            
            dist_matrix = cdist(coords, coords, metric='euclidean')
            threshold = np.percentile(dist_matrix[dist_matrix > 0], 90)
            
            W = (dist_matrix > 0) & (dist_matrix <= threshold)
            W = W.astype(float)
            row_sums = W.sum(axis=1)
            W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0, None]
            
            z = (values - values.mean()) / values.std()
            
            local_i = np.zeros(n)
            for i in range(n):
                w_z = np.sum(W[i] * z)
                local_i[i] = z[i] * w_z
            
            n_permutations = 199
            p_values = np.zeros(n)
            
            for i in range(n):
                count = 0
                observed = local_i[i]
                
                for _ in range(n_permutations):
                    perm_z = z[np.random.permutation(n)]
                    w_z_perm = np.sum(W[i] * perm_z)
                    perm_i = z[i] * w_z_perm
                    
                    if perm_i >= observed:
                        count += 1
                
                p_values[i] = count / n_permutations
            
            E_I = -W.sum(axis=1) / (n - 1)
            m2 = np.sum((z - z.mean()) ** 2) / n
            b2 = np.sum((z - z.mean()) ** 4) / n / (m2 ** 2)
            
            var_I = np.zeros(n)
            for i in range(n):
                wi2 = np.sum(W[i] ** 2)
                var_I[i] = wi2 * (n - b2) / (n - 1) - E_I[i] ** 2
            
            z_scores = (local_i - E_I) / np.sqrt(var_I)
            
            spatial_lag = W @ values
            
            cluster_types = []
            for i in range(n):
                if p_values[i] < 0.05:
                    if values[i] > values.mean() and spatial_lag[i] > spatial_lag.mean():
                        cluster_types.append('HH')
                    elif values[i] < values.mean() and spatial_lag[i] < spatial_lag.mean():
                        cluster_types.append('LL')
                    elif values[i] > values.mean() and spatial_lag[i] < spatial_lag.mean():
                        cluster_types.append('HL')
                    elif values[i] < values.mean() and spatial_lag[i] > spatial_lag.mean():
                        cluster_types.append('LH')
                    else:
                        cluster_types.append('Not Significant')
                else:
                    cluster_types.append('Not Significant')
            
            df_year['Local_I'] = local_i
            df_year['p_value'] = p_values
            df_year['Z_Score'] = z_scores
            df_year['Cluster_Type'] = cluster_types
            df_year['Year'] = year
            
            all_lisa_results.append(df_year)
            
            cluster_counts = pd.Series(cluster_types).value_counts().to_dict()
            cluster_counts['Year'] = year
            yearly_cluster_counts.append(cluster_counts)
            
        except Exception as e:
            print(f"年份 {year} 分析失败: {str(e)}")
            continue
    
    if not all_lisa_results:
        print("警告: 无有效的LISA分析结果")
        return None
    
    all_results_df = pd.concat(all_lisa_results, ignore_index=True)
    cluster_counts_df = pd.DataFrame(yearly_cluster_counts).fillna(0)
    
    all_results_df.to_csv('lisa_evolution_all_results.csv', index=False, encoding='utf-8-sig')
    cluster_counts_df.to_csv('lisa_evolution_cluster_counts.csv', index=False, encoding='utf-8-sig')
    
    print("\n演化分析汇总:")
    print(cluster_counts_df)
    
    self.results['lisa_evolution'] = {
        'all_results': all_results_df,
        'cluster_counts': cluster_counts_df,
        'years': years
    }
    
    self._plot_lisa_evolution(all_results_df, cluster_counts_df, years)
    
    print("\nLISA演化分析完成")
    
    return all_results_df, cluster_counts_df

ExploratoryDataAnalysis.lisa_evolution_analysis = lisa_evolution_analysis



def _plot_lisa_evolution(self, all_results_df, cluster_counts_df, years):
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    cluster_types = ['HH', 'LH', 'LL', 'HL', 'Not Significant']
    cluster_colors = ['red', 'orange', 'blue', 'purple', 'gray']
    
    for i, cluster_type in enumerate(cluster_types):
        if cluster_type in cluster_counts_df.columns:
            plt.plot(cluster_counts_df['Year'], cluster_counts_df[cluster_type], 
                    marker='o', label=cluster_type, color=cluster_colors[i], linewidth=2)
    
    plt.xlabel('年份')
    plt.ylabel('城市数量')
    plt.title('LISA聚类类型演化趋势')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    significant_ratio = []
    for year in years:
        if year in all_results_df['Year'].unique():
            year_data = all_results_df[all_results_df['Year'] == year]
            sig_count = (year_data['Cluster_Type'] != 'Not Significant').sum()
            total_count = len(year_data)
            if total_count > 0:
                significant_ratio.append(sig_count / total_count * 100)
            else:
                significant_ratio.append(0)
        else:
            significant_ratio.append(0)
    
    plt.plot(years, significant_ratio, marker='s', color='green', linewidth=2)
    plt.xlabel('年份')
    plt.ylabel('显著比例(%)')
    plt.title('显著聚类比例演化')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    mean_local_i = []
    for year in years:
        if year in all_results_df['Year'].unique():
            year_data = all_results_df[all_results_df['Year'] == year]
            mean_local_i.append(year_data['Local_I'].mean())
        else:
            mean_local_i.append(0)
    
    plt.plot(years, mean_local_i, marker='^', color='blue', linewidth=2)
    plt.xlabel('年份')
    plt.ylabel('平均Local I')
    plt.title('平均Local I指数演化')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    cities = all_results_df['城市'].unique()[:5]
    city_data = {}
    for city in cities:
        city_data[city] = []
        for year in years:
            year_city_data = all_results_df[(all_results_df['Year'] == year) & 
                                          (all_results_df['城市'] == city)]
            if len(year_city_data) > 0:
                city_data[city].append(year_city_data['Local_I'].iloc[0])
            else:
                city_data[city].append(np.nan)
    
    for city, values in city_data.items():
        plt.plot(years, values, marker='o', label=city, linewidth=2)
    
    plt.xlabel('年份')
    plt.ylabel('Local I指数')
    plt.title('重点城市Local I指数演化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lisa_evolution_analysis.png', dpi=300, bbox_inches='tight')
    print("LISA演化分析图已保存: lisa_evolution_analysis.png")
    
    plt.figure(figsize=(12, 8))
    if len(years) <= 4:
        n_cols = len(years)
    else:
        n_cols = 4
        n_rows = (len(years) + 3) // 4
    
    for i, year in enumerate(years[:4]):
        plt.subplot(1, n_cols, i+1)
        year_data = all_results_df[all_results_df['Year'] == year]
        
        if len(year_data) > 0:
            z = (year_data['跨境数据传输总量_TB'] - year_data['跨境数据传输总量_TB'].mean()) / year_data['跨境数据传输总量_TB'].std()
            w_z = year_data['Z_Score']
            
            colors = []
            for cluster in year_data['Cluster_Type']:
                if cluster == 'HH':
                    colors.append('red')
                elif cluster == 'LH':
                    colors.append('orange')
                elif cluster == 'LL':
                    colors.append('blue')
                elif cluster == 'HL':
                    colors.append('purple')
                else:
                    colors.append('gray')
            
            plt.scatter(z, w_z, s=60, c=colors, alpha=0.7)
            plt.axhline(0, color='black', linewidth=1, alpha=0.5)
            plt.axvline(0, color='black', linewidth=1, alpha=0.5)
            plt.xlabel('标准化值')
            plt.ylabel('LISA Z-Score')
            plt.title(f'LISA散点图 ({year}年)')
    
    plt.tight_layout()
    plt.savefig('lisa_evolution_scatter_grid.png', dpi=300, bbox_inches='tight')
    print("LISA演化散点网格图已保存: lisa_evolution_scatter_grid.png")

ExploratoryDataAnalysis.lisa_evolution_analysis = lisa_evolution_analysis
ExploratoryDataAnalysis._plot_lisa_evolution = _plot_lisa_evolution


def spatial_pattern_evolution(self):
    print("\n空间格局演化分析")
    print("-" * 80)
    
    if '年份' not in self.main_data.columns or '城市' not in self.main_data.columns:
        print("警告: 缺少必要字段")
        return None
    
    target_col = '跨境数据传输总量_TB'
    if target_col not in self.main_data.columns:
        print(f"警告: 缺少{target_col}字段")
        return None
    
    years = sorted(self.main_data['年份'].unique())
    
    pattern_evolution = []
    for year in years:
        year_data = self.main_data[self.main_data['年份'] == year]
        
        city_values = year_data.groupby('城市')[target_col].mean()
        
        if len(city_values) > 1:
            cv = city_values.std() / city_values.mean() if city_values.mean() != 0 else np.nan
            max_min_ratio = city_values.max() / city_values.min() if city_values.min() != 0 else np.nan
            
            pattern_evolution.append({
                'Year': year,
                'Mean': city_values.mean(),
                'Std': city_values.std(),
                'CV': cv,
                'Max_Min_Ratio': max_min_ratio,
                'Cities': len(city_values)
            })
    
    evolution_df = pd.DataFrame(pattern_evolution)
    print("\n空间格局演化:")
    print(evolution_df.round(3))
    
    evolution_df.to_csv('spatial_pattern_evolution.csv', index=False, encoding='utf-8-sig')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(evolution_df['Year'], evolution_df['Mean'], marker='o')
    axes[0, 0].set_title('均值演化')
    axes[0, 0].set_xlabel('年份')
    axes[0, 0].set_ylabel('均值')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(evolution_df['Year'], evolution_df['Std'], marker='o', color='orange')
    axes[0, 1].set_title('标准差演化')
    axes[0, 1].set_xlabel('年份')
    axes[0, 1].set_ylabel('标准差')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(evolution_df['Year'], evolution_df['CV'], marker='o', color='green')
    axes[1, 0].set_title('变异系数演化')
    axes[1, 0].set_xlabel('年份')
    axes[1, 0].set_ylabel('CV')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(evolution_df['Year'], evolution_df['Max_Min_Ratio'], marker='o', color='red')
    axes[1, 1].set_title('最大最小值比演化')
    axes[1, 1].set_xlabel('年份')
    axes[1, 1].set_ylabel('比率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spatial_pattern_evolution.png', dpi=300, bbox_inches='tight')
    print("空间格局演化图已保存: spatial_pattern_evolution.png")
    
    self.results['pattern_evolution'] = evolution_df
    return evolution_df

ExploratoryDataAnalysis.spatial_pattern_evolution = spatial_pattern_evolution


def generate_eda_report(self):
    print("\n" + "=" * 80)
    print("生成综合报告")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("探索性数据分析 - 综合报告")
    report.append("=" * 80)
    report.append(f"\n生成时间: {pd.Timestamp.now()}")
    
    report.append("\n描述性统计分析")
    report.append("-" * 80)
    if 'descriptive_stats' in self.results:
        report.append(f"分析指标数: {len(self.results['descriptive_stats'])}")
    
    if 'gini' in self.results:
        latest_gini = self.results['gini'].iloc[-1]['Gini']
        report.append(f"最新基尼系数: {latest_gini:.4f}")
        if latest_gini > 0.4:
            report.append("  警告: 基尼系数超过0.4警戒线")
    
    if 'theil' in self.results:
        latest_theil = self.results['theil'].iloc[-1]
        report.append(f"泰尔指数总体: {latest_theil['Theil_Total']:.4f}")
        report.append(f"  区域内: {latest_theil['Theil_Within']:.4f}")
        report.append(f"  区域间: {latest_theil['Theil_Between']:.4f}")
    
    report.append("\n相关性分析")
    report.append("-" * 80)
    if 'correlations' in self.results:
        pearson = self.results['correlations']['pearson']
        high_corr = pearson.abs().unstack()
        high_corr = high_corr[high_corr < 1].nlargest(3)
        report.append("最强相关关系(Top 3):")
        for idx, val in high_corr.items():
            report.append(f"  {idx[0]} vs {idx[1]}: {val:.4f}")
    
    if 'partial_correlation' in self.results:
        sig_partial = self.results['partial_correlation'][
            self.results['partial_correlation']['Significant']
        ]
        report.append(f"显著偏相关关系数: {len(sig_partial)}")
    
    report.append("\n空间自相关分析")
    report.append("-" * 80)
    if 'moran_i' in self.results:
        moran = self.results['moran_i']
        report.append(f"全局Moran's I: {moran['Moran_I']:.4f}")
        report.append(f"Z得分: {moran['Z_Score']:.4f}")
        report.append(f"空间格局: {moran['Pattern']}")
    
    if 'lisa' in self.results:
        lisa = self.results['lisa']
        hh_count = (lisa['Cluster_Type'] == 'HH').sum()
        ll_count = (lisa['Cluster_Type'] == 'LL').sum()
        report.append(f"高-高聚类城市数: {hh_count}")
        report.append(f"低-低聚类城市数: {ll_count}")
    
    if 'lisa_evolution' in self.results:
        evolution = self.results['lisa_evolution']
        report.append("\nLISA演化分析")
        report.append("-" * 80)
        report.append(f"分析年份数: {len(evolution['years'])}")
        report.append(f"城市平均Local I变化: {evolution['all_results']['Local_I'].mean():.4f}")
        if 'cluster_counts' in evolution:
            latest_counts = evolution['cluster_counts'].iloc[-1]
            report.append(f"最新年份显著聚类比例: {(latest_counts.get('HH', 0) + latest_counts.get('LH', 0) + latest_counts.get('LL', 0) + latest_counts.get('HL', 0)) / sum([latest_counts.get(c, 0) for c in ['HH', 'LH', 'LL', 'HL', 'Not Significant']]) * 100:.1f}%")
    
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open('chapter5_eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n综合报告已保存: chapter5_eda_report.txt")
    
    return report_text

ExploratoryDataAnalysis.generate_eda_report = generate_eda_report



def run_full_eda(self):
    print("\n" + "=" * 80)
    print("探索性数据分析 - 完整流程")
    print("=" * 80)
    
    self.descriptive_statistics_advanced()
    self.temporal_trend_analysis()
    self.time_series_decomposition()
    self.spatial_distribution_analysis()
    self.city_disparity_analysis()
    self.calculate_gini_coefficient()
    self.calculate_theil_index()
    self.plot_lorenz_curve()
    self.correlation_analysis()
    self.partial_correlation_analysis()
    self.distance_correlation()
    self.maximal_information_coefficient()
    self.multiple_testing_correction()
    self.spatial_autocorrelation_analysis()
    self.lisa_cluster_analysis()
    self.lisa_evolution_analysis()
    self.spatial_pattern_evolution()
    self.generate_eda_report()
    
    print("\n" + "=" * 80)
    print("所有分析完成 ")
    print("=" * 80)

ExploratoryDataAnalysis.run_full_eda = run_full_eda
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    eda = ExploratoryDataAnalysis(
        main_data=preprocessor.main_data,
        od_matrix=preprocessor.od_matrix,
        od_yearly=preprocessor.od_yearly
    )

    eda.run_full_eda()


class ClusteringClassificationAnalysis:
    
    def __init__(self, main_data, od_matrix, od_yearly):
        self.main_data = main_data
        self.od_matrix = od_matrix
        self.od_yearly = od_yearly
        self.results = {}
        self.clustering_data = None
        self.labels_dict = {}
        
        from sklearn.base import clone as sklearn_clone
        self.clone = sklearn_clone
        
    def prepare_clustering_data(self):
        print("\n" + "=" * 80)
        print("准备聚类分析数据（城市-年份面板数据）")
        print("=" * 80)
        
        cluster_features = [
            '跨境数据传输总量_TB', 'GDP_亿元', '数字经济核心产业增加值_亿元',
            '数据交易额_亿元', '5G基站数量', '算力规模_PFLOPS',
            '研发经费投入_亿元', '高新技术企业数'
        ]
        
        df_panel = self.main_data.copy()
        
        available_features = [f for f in cluster_features if f in df_panel.columns]
        print(f"可用特征数: {len(available_features)}")
        print(f"特征列表: {available_features}")
        
        X = df_panel[available_features].copy()
        X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        city_year_id = df_panel['城市'].astype(str) + '_' + df_panel['年份'].astype(str)
        
        self.clustering_data = {
            'X': X,
            'X_scaled': X_scaled,
            'features': available_features,
            'cities': df_panel['城市'].values,
            'years': df_panel['年份'].values,
            'city_year_id': city_year_id.values,
            'scaler': scaler,
            'df_panel': df_panel
        }
        
        print(f"聚类数据形状: {X_scaled.shape}")
        print(f"样本数量（城市-年份对）: {len(X_scaled)}")
        print(f"城市数量: {df_panel['城市'].nunique()}")
        print(f"年份数量: {df_panel['年份'].nunique()}")
        
        return self.clustering_data
    
    def determine_optimal_clusters(self):
        print("\n聚类数确定分析")
        print("-" * 80)
        
        if self.clustering_data is None:
            self.prepare_clustering_data()
        
        X = self.clustering_data['X_scaled']
        
        max_k = min(15, len(X) // 5) 
        k_range = range(2, max_k + 1)
        
        print(f"测试聚类数范围: 2 到 {max_k}")
        
        sse = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)
        
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        ch_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            score = calinski_harabasz_score(X, labels)
            ch_scores.append(score)
        
        db_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            score = davies_bouldin_score(X, labels)
            db_scores.append(score)
        
        gap_stats = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            
            W_k = kmeans.inertia_
            
            n_refs = 10
            ref_dispersions = []
            for _ in range(n_refs):
                X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
                kmeans_ref = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
                kmeans_ref.fit(X_ref)
                ref_dispersions.append(kmeans_ref.inertia_)
            
            E_W_k = np.mean(ref_dispersions)
            gap = np.log(E_W_k) - np.log(W_k)
            gap_stats.append(gap)
        
        bic_scores = []
        aic_scores = []
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
        
        optimal_k_results = pd.DataFrame({
            'n_clusters': list(k_range),
            'SSE': sse,
            'Silhouette': silhouette_scores,
            'Calinski_Harabasz': ch_scores,
            'Davies_Bouldin': db_scores,
            'Gap_Statistic': gap_stats,
            'BIC': bic_scores,
            'AIC': aic_scores
        })
        
        optimal_k_results.to_csv('7.1_optimal_clusters_metrics.csv', index=False, encoding='utf-8-sig')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(k_range, sse, 'bo-', linewidth=2)
        axes[0, 0].set_xlabel('聚类数 k')
        axes[0, 0].set_ylabel('SSE')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2)
        axes[0, 1].set_xlabel('聚类数 k')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(k_range, ch_scores, 'ro-', linewidth=2)
        axes[0, 2].set_xlabel('聚类数 k')
        axes[0, 2].set_ylabel('Calinski-Harabasz Index')
        axes[0, 2].set_title('Calinski-Harabasz Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(k_range, db_scores, 'mo-', linewidth=2)
        axes[1, 0].set_xlabel('聚类数 k')
        axes[1, 0].set_ylabel('Davies-Bouldin Index')
        axes[1, 0].set_title('Davies-Bouldin Score (越小越好)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(k_range, gap_stats, 'co-', linewidth=2)
        axes[1, 1].set_xlabel('聚类数 k')
        axes[1, 1].set_ylabel('Gap Statistic')
        axes[1, 1].set_title('Gap Statistic (越大越好)')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(k_range, bic_scores, 'o-', label='BIC', linewidth=2)
        axes[1, 2].plot(k_range, aic_scores, 's-', label='AIC', linewidth=2)
        axes[1, 2].set_xlabel('聚类数 k')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('BIC/AIC for GMM (越小越好)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('7.1_optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        print("最优聚类数分析图已保存: 7.1_optimal_clusters_analysis.png")
        
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        optimal_k_ch = k_range[np.argmax(ch_scores)]
        optimal_k_db = k_range[np.argmin(db_scores)]
        optimal_k_gap = k_range[np.argmax(gap_stats)]
        optimal_k_bic = k_range[np.argmin(bic_scores)]
        
        print("\n推荐最优聚类数:")
        print(f"  Silhouette Score: k = {optimal_k_silhouette}")
        print(f"  Calinski-Harabasz: k = {optimal_k_ch}")
        print(f"  Davies-Bouldin: k = {optimal_k_db}")
        print(f"  Gap Statistic: k = {optimal_k_gap}")
        print(f"  BIC: k = {optimal_k_bic}")
        
        optimal_ks = [optimal_k_silhouette, optimal_k_ch, optimal_k_db, optimal_k_gap, optimal_k_bic]
        recommended_k = max(set(optimal_ks), key=optimal_ks.count)
        print(f"\n综合推荐聚类数: k = {recommended_k}")
        
        self.results['optimal_k'] = {
            'metrics': optimal_k_results,
            'recommended': recommended_k,
            'individual_recommendations': {
                'silhouette': optimal_k_silhouette,
                'calinski_harabasz': optimal_k_ch,
                'davies_bouldin': optimal_k_db,
                'gap_statistic': optimal_k_gap,
                'bic': optimal_k_bic
            }
        }
        
        return optimal_k_results, recommended_k
    
    def perform_multiple_clustering(self, n_clusters=None):
        print("\n多算法聚类分析")
        print("-" * 80)
        
        if self.clustering_data is None:
            self.prepare_clustering_data()
        
        X = self.clustering_data['X_scaled']
        city_year_id = self.clustering_data['city_year_id']
        
        if n_clusters is None:
            if 'optimal_k' in self.results:
                n_clusters = self.results['optimal_k']['recommended']
            else:
                n_clusters = 4
        
        print(f"使用聚类数: k = {n_clusters}")
        
        clustering_results = {}
        
        print("\n1. K-Means聚类...")
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        labels_kmeans = kmeans.fit_predict(X)
        clustering_results['KMeans'] = {
            'labels': labels_kmeans,
            'model': kmeans,
            'silhouette': silhouette_score(X, labels_kmeans),
            'calinski_harabasz': calinski_harabasz_score(X, labels_kmeans),
            'davies_bouldin': davies_bouldin_score(X, labels_kmeans)
        }
        
        print("2. 层次聚类(Ward)...")
        hierarchical_ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels_ward = hierarchical_ward.fit_predict(X)
        clustering_results['Hierarchical_Ward'] = {
            'labels': labels_ward,
            'model': hierarchical_ward,
            'silhouette': silhouette_score(X, labels_ward),
            'calinski_harabasz': calinski_harabasz_score(X, labels_ward),
            'davies_bouldin': davies_bouldin_score(X, labels_ward)
        }
        
        print("3. 层次聚类(Average)...")
        hierarchical_avg = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        labels_avg = hierarchical_avg.fit_predict(X)
        clustering_results['Hierarchical_Average'] = {
            'labels': labels_avg,
            'model': hierarchical_avg,
            'silhouette': silhouette_score(X, labels_avg),
            'calinski_harabasz': calinski_harabasz_score(X, labels_avg),
            'davies_bouldin': davies_bouldin_score(X, labels_avg)
        }
        
        print("4. DBSCAN聚类...")
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, -1], axis=0)
        eps = np.percentile(distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels_dbscan = dbscan.fit_predict(X)
        
        n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        if n_clusters_dbscan > 1:
            mask = labels_dbscan != -1
            if mask.sum() > 1:
                clustering_results['DBSCAN'] = {
                    'labels': labels_dbscan,
                    'model': dbscan,
                    'n_clusters': n_clusters_dbscan,
                    'n_noise': (labels_dbscan == -1).sum(),
                    'silhouette': silhouette_score(X[mask], labels_dbscan[mask]) if len(set(labels_dbscan[mask])) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X[mask], labels_dbscan[mask]) if len(set(labels_dbscan[mask])) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X[mask], labels_dbscan[mask]) if len(set(labels_dbscan[mask])) > 1 else np.nan
                }
        else:
            print("  警告: DBSCAN未能产生有效聚类")
        
        print("5. 高斯混合模型...")
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        gmm.fit(X)
        labels_gmm = gmm.predict(X)
        probs_gmm = gmm.predict_proba(X)
        
        clustering_results['GMM'] = {
            'labels': labels_gmm,
            'probabilities': probs_gmm,
            'model': gmm,
            'silhouette': silhouette_score(X, labels_gmm),
            'calinski_harabasz': calinski_harabasz_score(X, labels_gmm),
            'davies_bouldin': davies_bouldin_score(X, labels_gmm),
            'bic': gmm.bic(X),
            'aic': gmm.aic(X)
        }
        
        print("6. 谱聚类...")
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
        labels_spectral = spectral.fit_predict(X)
        clustering_results['Spectral'] = {
            'labels': labels_spectral,
            'model': spectral,
            'silhouette': silhouette_score(X, labels_spectral),
            'calinski_harabasz': calinski_harabasz_score(X, labels_spectral),
            'davies_bouldin': davies_bouldin_score(X, labels_spectral)
        }
        
        self.labels_dict = {name: result['labels'] for name, result in clustering_results.items()}
        
        comparison = []
        for method, result in clustering_results.items():
            comparison.append({
                'Method': method,
                'N_Clusters': n_clusters if method != 'DBSCAN' else result.get('n_clusters', n_clusters),
                'Silhouette': result['silhouette'],
                'Calinski_Harabasz': result['calinski_harabasz'],
                'Davies_Bouldin': result['davies_bouldin']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv('7.1_clustering_algorithms_comparison.csv', index=False, encoding='utf-8-sig')
        
        print("\n聚类算法对比:")
        print(comparison_df.round(4))
        
        self._visualize_clustering_comparison(X, clustering_results, city_year_id)
        
        self.results['clustering'] = clustering_results
        self.results['clustering_comparison'] = comparison_df
        
        return clustering_results, comparison_df
    
    def _visualize_clustering_comparison(self, X, clustering_results, city_year_id):
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        n_methods = len(clustering_results)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for idx, (method, result) in enumerate(clustering_results.items()):
            ax = axes[idx]
            labels = result['labels']
            
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=labels, cmap='viridis', 
                               s=50, alpha=0.6, edgecolors='black')
            
            ax.set_title(f'{method}\nSilhouette: {result["silhouette"]:.3f}')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('7.1_clustering_algorithms_visualization.png', dpi=300, bbox_inches='tight')
        print("聚类算法可视化对比图已保存: 7.1_clustering_algorithms_visualization.png")
        
        plt.close()
    
    def clustering_stability_analysis(self, n_bootstrap=100):
        print("\n聚类稳定性分析(Bootstrap重采样)")
        print("-" * 80)
        
        if self.clustering_data is None:
            self.prepare_clustering_data()
        
        X = self.clustering_data['X_scaled']
        n_samples = X.shape[0]
        
        if 'optimal_k' in self.results:
            n_clusters = self.results['optimal_k']['recommended']
        else:
            n_clusters = 4
        
        print(f"Bootstrap重采样次数: {n_bootstrap}")
        print(f"聚类数: {n_clusters}")
        
        kmeans_original = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        labels_original = kmeans_original.fit_predict(X)
        
        jaccard_scores = []
        ari_scores = []
        nmi_scores = []
        
        for i in range(n_bootstrap):
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{n_bootstrap}")
            
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            
            kmeans_bootstrap = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=i)
            labels_bootstrap = kmeans_bootstrap.fit_predict(X_bootstrap)
            
            labels_bootstrap_full = np.full(n_samples, -1)
            labels_bootstrap_full[indices] = labels_bootstrap
            
            mask = labels_bootstrap_full != -1
            if mask.sum() > 0:
                ari = adjusted_rand_score(labels_original[mask], labels_bootstrap_full[mask])
                nmi = normalized_mutual_info_score(labels_original[mask], labels_bootstrap_full[mask])
                
                ari_scores.append(ari)
                nmi_scores.append(nmi)
                
                jaccard = self._compute_jaccard_similarity(labels_original[mask], labels_bootstrap_full[mask])
                jaccard_scores.append(jaccard)
        
        stability_results = pd.DataFrame({
            'Jaccard_Mean': [np.mean(jaccard_scores)],
            'Jaccard_Std': [np.std(jaccard_scores)],
            'ARI_Mean': [np.mean(ari_scores)],
            'ARI_Std': [np.std(ari_scores)],
            'NMI_Mean': [np.mean(nmi_scores)],
            'NMI_Std': [np.std(nmi_scores)]
        })
        
        print("\n稳定性分析结果:")
        print(stability_results.round(4))
        
        stability_results.to_csv('7.1_clustering_stability_bootstrap.csv', index=False, encoding='utf-8-sig')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(jaccard_scores, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(jaccard_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(jaccard_scores):.3f}')
        axes[0].set_xlabel('Jaccard Coefficient')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Jaccard系数分布')
        axes[0].legend()
        
        axes[1].hist(ari_scores, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(np.mean(ari_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ari_scores):.3f}')
        axes[1].set_xlabel('Adjusted Rand Index')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('ARI分布')
        axes[1].legend()
        
        axes[2].hist(nmi_scores, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[2].axvline(np.mean(nmi_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(nmi_scores):.3f}')
        axes[2].set_xlabel('Normalized Mutual Information')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('NMI分布')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('7.1_clustering_stability_distribution.png', dpi=300, bbox_inches='tight')
        print("稳定性分析分布图已保存: 7.1_clustering_stability_distribution.png")
        
        self.results['stability'] = {
            'jaccard_scores': jaccard_scores,
            'ari_scores': ari_scores,
            'nmi_scores': nmi_scores,
            'summary': stability_results
        }
        
        return stability_results
    
    def _compute_jaccard_similarity(self, labels1, labels2):
        n = len(labels1)
        pairs1 = set()
        pairs2 = set()
        
        for i in range(n):
            for j in range(i+1, n):
                if labels1[i] == labels1[j]:
                    pairs1.add((i, j))
                if labels2[i] == labels2[j]:
                    pairs2.add((i, j))
        
        if len(pairs1) == 0 and len(pairs2) == 0:
            return 1.0
        
        intersection = len(pairs1 & pairs2)
        union = len(pairs1 | pairs2)
        
        return intersection / union if union > 0 else 0.0
    
    def dynamic_clustering_evolution(self):
        print("\n" + "=" * 80)
        print("动态聚类演化分析（按年份追踪城市类别变化）")
        print("=" * 80)
        
        if self.clustering_data is None:
            self.prepare_clustering_data()
        
        if 'clustering' not in self.results:
            print("警告: 请先执行聚类分析")
            return None
        
        labels = self.results['clustering']['KMeans']['labels']
        cities = self.clustering_data['cities']
        years = self.clustering_data['years']
        
        evolution_data = pd.DataFrame({
            '城市': cities,
            '年份': years,
            '聚类标签': labels
        })
        
        evolution_matrix = evolution_data.pivot(index='城市', columns='年份', values='聚类标签')
        evolution_matrix.to_csv('7.1_clustering_evolution_matrix.csv', encoding='utf-8-sig')
        
        print("\n聚类演化矩阵（城市×年份）:")
        print(evolution_matrix)
        
        yearly_distribution = evolution_data.groupby(['年份', '聚类标签']).size().unstack(fill_value=0)
        yearly_distribution.to_csv('7.1_yearly_cluster_distribution.csv', encoding='utf-8-sig')
        
        print("\n每年聚类分布:")
        print(yearly_distribution)
        
        self._visualize_cluster_evolution(evolution_matrix)
        
        self._plot_yearly_distribution(yearly_distribution)
        
        self._analyze_cluster_transitions(evolution_matrix)
        
        self.results['evolution'] = {
            'evolution_matrix': evolution_matrix,
            'yearly_distribution': yearly_distribution,
            'evolution_data': evolution_data
        }
        
        return evolution_matrix
    
    def _visualize_cluster_evolution(self, evolution_matrix):
        
        plt.figure(figsize=(12, max(8, len(evolution_matrix) * 0.5)))
        
        cmap = plt.cm.get_cmap('Set3')
        
        sns.heatmap(evolution_matrix, annot=True, fmt='.0f', cmap=cmap,
                   cbar_kws={'label': '簇标签'}, linewidths=0.5)
        plt.xlabel('年份')
        plt.ylabel('城市')
        plt.title('城市聚类演化热图（按年份）')
        plt.tight_layout()
        plt.savefig('7.1_clustering_evolution_heatmap.png', dpi=300, bbox_inches='tight')
        print("聚类演化热图已保存: 7.1_clustering_evolution_heatmap.png")
        
        plt.close()
    
    def _plot_yearly_distribution(self, yearly_distribution):
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        yearly_distribution.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
        ax.set_xlabel('年份')
        ax.set_ylabel('城市数量')
        ax.set_title('各年份聚类分布')
        ax.legend(title='聚类标签', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('7.1_yearly_cluster_distribution.png', dpi=300, bbox_inches='tight')
        print("年度聚类分布图已保存: 7.1_yearly_cluster_distribution.png")
        
        plt.close()
    
    def _analyze_cluster_transitions(self, evolution_matrix):
        
        years = evolution_matrix.columns.tolist()
        
        if len(years) < 2:
            print("警告: 年份数量不足，无法分析转移")
            return
        
        year_start = years[0]
        year_end = years[-1]
        
        labels_start = evolution_matrix[year_start].dropna()
        labels_end = evolution_matrix[year_end].dropna()
        
        common_cities = labels_start.index.intersection(labels_end.index)
        
        if len(common_cities) == 0:
            print("警告: 没有共同城市数据")
            return
        
        labels_start_common = labels_start.loc[common_cities].astype(int)
        labels_end_common = labels_end.loc[common_cities].astype(int)
        
        max_cluster = max(labels_start_common.max(), labels_end_common.max())
        transition_matrix = np.zeros((max_cluster + 1, max_cluster + 1))
        
        for city in common_cities:
            label_start = labels_start_common.loc[city]
            label_end = labels_end_common.loc[city]
            transition_matrix[label_start, label_end] += 1
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'C{i}' for i in range(max_cluster + 1)],
                   yticklabels=[f'C{i}' for i in range(max_cluster + 1)])
        plt.xlabel(f'{year_end}年簇标签')
        plt.ylabel(f'{year_start}年簇标签')
        plt.title(f'城市簇转移矩阵 ({year_start} → {year_end})')
        plt.tight_layout()
        plt.savefig('7.1_cluster_transition_matrix.png', dpi=300, bbox_inches='tight')
        print("簇转移矩阵图已保存: 7.1_cluster_transition_matrix.png")
        
        plt.close()
    
    def interpret_clustering_results(self):
        print("\n" + "=" * 80)
        print("聚类结果解释（基于城市-年份面板数据）")
        print("=" * 80)
        
        if 'clustering' not in self.results:
            print("警告: 请先执行聚类分析")
            return None
        
        kmeans_result = self.results['clustering']['KMeans']
        labels = kmeans_result['labels']
        
        X = self.clustering_data['X']
        features = self.clustering_data['features']
        cities = self.clustering_data['cities']
        years = self.clustering_data['years']
        
        cluster_profiles = []
        for cluster_id in range(len(set(labels))):
            mask = labels == cluster_id
            cluster_data = X[mask]
            
            profile = {'Cluster': f'C{cluster_id}', 'N_Samples': mask.sum()}
            for feat in features:
                profile[feat] = cluster_data[feat].mean()
            
            cluster_city_years = [(cities[i], years[i]) for i in range(len(labels)) if mask[i]]
            sample_city_years = cluster_city_years[:5]
            profile['Sample_City_Years'] = ', '.join([f'{c}_{y}' for c, y in sample_city_years])
            
            cluster_profiles.append(profile)
        
        profile_df = pd.DataFrame(cluster_profiles)
        profile_df.to_csv('7.1_cluster_profiles.csv', index=False, encoding='utf-8-sig')
        
        print("\n聚类特征剖面:")
        print(profile_df.round(2))
        
        self._plot_cluster_radar_chart(profile_df, features)
        
        feature_importance = self._compute_feature_importance(X, labels, features)
        
        self.results['cluster_interpretation'] = {
            'profiles': profile_df,
            'feature_importance': feature_importance
        }
        
        return profile_df
    
    def _plot_cluster_radar_chart(self, profile_df, features):
        
        n_clusters = len(profile_df)
        n_features = len(features)
        
        feature_data = profile_df[features].values
        feature_data_normalized = (feature_data - feature_data.min(axis=0)) / (feature_data.max(axis=0) - feature_data.min(axis=0) + 1e-10)
        
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            values = feature_data_normalized[i].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}', color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('聚类特征雷达图', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('7.1_cluster_radar_chart.png', dpi=300, bbox_inches='tight')
        print("聚类特征雷达图已保存: 7.1_cluster_radar_chart.png")
        
        plt.close()
    
    def _compute_feature_importance(self, X, labels, features):
        
        importance_scores = []
        
        for feat in features:
            groups = [X[labels == i][feat].values for i in range(len(set(labels)))]
            
            f_stat, p_val = stats.f_oneway(*groups)
            
            importance_scores.append({
                'Feature': feat,
                'F_Statistic': f_stat,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })
        
        importance_df = pd.DataFrame(importance_scores).sort_values('F_Statistic', ascending=False)
        importance_df.to_csv('7.1_feature_importance_clustering.csv', index=False, encoding='utf-8-sig')
        
        print("\n特征重要性(方差分析):")
        print(importance_df.round(4))
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['F_Statistic'])
        plt.xlabel('F-Statistic')
        plt.ylabel('Feature')
        plt.title('聚类特征重要性')
        plt.tight_layout()
        plt.savefig('7.1_feature_importance_bar.png', dpi=300, bbox_inches='tight')
        print("特征重要性柱状图已保存: 7.1_feature_importance_bar.png")
        
        plt.close()
        
        return importance_df
    
    
    def prepare_classification_data(self):
        print("\n" + "=" * 80)
        print("准备分类分析数据")
        print("=" * 80)
        
        X_full = self.clustering_data['X']
        features = self.clustering_data['features']
        
        print("\n步骤1: 先划分训练集和测试集（按城市分层）")
        
        cities = self.clustering_data['cities']
        years = self.clustering_data['years']
        
        unique_cities = np.unique(cities)
        
        indices = np.arange(len(X_full))
        
        train_indices, test_indices = train_test_split(
            indices, test_size=0.3, random_state=42, 
            stratify=cities  # 按城市分层
        )
        
        X_train_full = X_full.iloc[train_indices]
        X_test_full = X_full.iloc[test_indices]
        
        print(f"训练集大小: {len(train_indices)}")
        print(f"测试集大小: {len(test_indices)}")
        
        print("\n步骤2: 仅在训练集上进行聚类生成标签")
        
        scaler_train = StandardScaler()
        X_train_scaled = scaler_train.fit_transform(X_train_full)
        
        if 'optimal_k' in self.results:
            n_clusters = self.results['optimal_k']['recommended']
        else:
            n_clusters = 4
        
        if n_clusters < 3:
            print(f"警告: 推荐聚类数为{n_clusters}，但这会导致分类任务过于简单")
            n_clusters = min(4, len(train_indices) // 8)  # 至少3-4个簇
            print(f"调整聚类数为: {n_clusters}")
        
        print(f"使用聚类数: {n_clusters}")
        
        kmeans_train = KMeans(n_clusters=n_clusters, init='k-means++', 
                             n_init=10, random_state=42)
        y_train = kmeans_train.fit_predict(X_train_scaled)
        
        print("\n步骤3: 使用训练集聚类模型预测测试集")
        
        X_test_scaled = scaler_train.transform(X_test_full)
        y_test = kmeans_train.predict(X_test_scaled)
        
        print(f"\n训练集标签分布:")
        train_dist = pd.Series(y_train).value_counts().sort_index()
        print(train_dist)
        
        print(f"\n测试集标签分布:")
        test_dist = pd.Series(y_test).value_counts().sort_index()
        print(test_dist)
        
        print("\n步骤4: 诊断簇分离度")
        cluster_centers = kmeans_train.cluster_centers_
        from scipy.spatial.distance import pdist
        center_distances = pdist(cluster_centers, metric='euclidean')
        avg_distance = np.mean(center_distances)
        print(f"簇中心间平均距离: {avg_distance:.4f}")
        
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X_train_scaled, y_train)
        print(f"训练集轮廓系数: {silhouette:.4f}")
        
        if silhouette > 0.7:
            print("警告: 轮廓系数>0.7，簇分离度过高，分类任务可能过于简单")
        
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        valid_labels = unique_labels[label_counts >= 3]
        
        if len(valid_labels) < 2:
            raise ValueError("训练集有效类别数不足2个")
        
        train_mask = np.isin(y_train, valid_labels)
        X_train_filtered = X_train_full[train_mask]
        X_train_scaled_filtered = X_train_scaled[train_mask]
        y_train_filtered = y_train[train_mask]
        
        test_mask = np.isin(y_test, valid_labels)
        X_test_filtered = X_test_full[test_mask]
        X_test_scaled_filtered = X_test_scaled[test_mask]
        y_test_filtered = y_test[test_mask]
        
        print(f"\n过滤后:")
        print(f"训练集: {len(y_train_filtered)} 样本, {len(valid_labels)} 类别")
        print(f"测试集: {len(y_test_filtered)} 样本")
        
        print("\n步骤5: 评估分类任务难度")
        from sklearn.metrics import mutual_info_classif
        mi_scores = mutual_info_classif(X_train_scaled_filtered, y_train_filtered, 
                                       discrete_features=False, random_state=42)
        avg_mi = np.mean(mi_scores)
        print(f"特征与标签平均互信息: {avg_mi:.4f}")
        
        if avg_mi > 1.0:
            print("警告: 特征与标签互信息过高，可能导致过拟合")
        
        self.classification_data = {
            'X_train': X_train_filtered,
            'X_test': X_test_filtered,
            'X_train_scaled': X_train_scaled_filtered,
            'X_test_scaled': X_test_scaled_filtered,
            'y_train': y_train_filtered,
            'y_test': y_test_filtered,
            'features': features,
            'scaler': scaler_train,
            'kmeans_model': kmeans_train,
            'diagnostics': {
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'avg_cluster_distance': avg_distance,
                'avg_mutual_info': avg_mi
            }
        }
        
        return self.classification_data
    
    def perform_discriminant_analysis(self):
        print("\n判别分析")
        print("-" * 80)
        
        if not hasattr(self, 'classification_data'):
            self.prepare_classification_data()
        
        X_train = self.classification_data['X_train_scaled']
        X_test = self.classification_data['X_test_scaled']
        y_train = self.classification_data['y_train']
        y_test = self.classification_data['y_test']
        
        discriminant_results = {}
        
        print("\n1. 线性判别分析(LDA)...")
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, y_train)
            y_pred_lda = lda.predict(X_test)
            y_pred_proba_lda = lda.predict_proba(X_test)
            
            discriminant_results['LDA'] = {
                'model': lda,
                'y_pred': y_pred_lda,
                'y_pred_proba': y_pred_proba_lda,
                'accuracy': accuracy_score(y_test, y_pred_lda),
                'confusion_matrix': confusion_matrix(y_test, y_pred_lda)
            }
            print(f"   LDA准确率: {discriminant_results['LDA']['accuracy']:.4f}")
        except Exception as e:
            print(f"   LDA训练失败: {str(e)}")
        
        print("2. 二次判别分析(QDA)...")
        try:
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(X_train, y_train)
            y_pred_qda = qda.predict(X_test)
            y_pred_proba_qda = qda.predict_proba(X_test)
            
            discriminant_results['QDA'] = {
                'model': qda,
                'y_pred': y_pred_qda,
                'y_pred_proba': y_pred_proba_qda,
                'accuracy': accuracy_score(y_test, y_pred_qda),
                'confusion_matrix': confusion_matrix(y_test, y_pred_qda)
            }
            print(f"   QDA准确率: {discriminant_results['QDA']['accuracy']:.4f}")
        except Exception as e:
            print(f"   QDA训练失败: {str(e)}")
        
        if discriminant_results:
            self._plot_confusion_matrices(discriminant_results, y_test)
        
        self.results['discriminant_analysis'] = discriminant_results
        
        return discriminant_results
    
    def perform_classification(self):
        print("\n多分类算法对比")
        print("-" * 80)
        
        if not hasattr(self, 'classification_data'):
            self.prepare_classification_data()
        
        X_train = self.classification_data['X_train_scaled']
        X_test = self.classification_data['X_test_scaled']
        y_train = self.classification_data['y_train']
        y_test = self.classification_data['y_test']
        
        classification_results = {}
        
        if 'discriminant_analysis' in self.results:
            for method, result in self.results['discriminant_analysis'].items():
                classification_results[method] = result
        
        print("\n1. Logistic回归...")
        try:
            logistic = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
            logistic.fit(X_train, y_train)
            y_pred_logistic = logistic.predict(X_test)
            
            classification_results['Logistic'] = {
                'model': logistic,
                'y_pred': y_pred_logistic,
                'y_pred_proba': logistic.predict_proba(X_test)
            }
            print(f"   Logistic准确率: {accuracy_score(y_test, y_pred_logistic):.4f}")
        except Exception as e:
            print(f"   Logistic训练失败: {str(e)}")
        
        print("2. 随机森林...")
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            classification_results['RandomForest'] = {
                'model': rf,
                'y_pred': y_pred_rf,
                'y_pred_proba': rf.predict_proba(X_test),
                'feature_importance': rf.feature_importances_
            }
            print(f"   RandomForest准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
        except Exception as e:
            print(f"   RandomForest训练失败: {str(e)}")
        
        if not classification_results:
            print("\n错误: 没有成功训练任何分类模型")
            return None, None
        
        evaluation_results = []
        
        for method, result in classification_results.items():
            try:
                y_pred = result['y_pred']
                
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                
                y_pred_proba = result['y_pred_proba']
                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba, 
                                             multi_class='ovr', average='weighted')
                except:
                    auc_score = np.nan
                
                evaluation_results.append({
                    'Method': method,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'AUC_OvR': auc_score
                })
            except Exception as e:
                print(f"  警告: {method} 评估失败: {str(e)}")
                continue
        
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv('7.2_classification_evaluation.csv', index=False, encoding='utf-8-sig')
        
        print("\n分类算法评估结果:")
        print(evaluation_df.round(4))
        
        self._visualize_classification_comparison(evaluation_df)
        self._plot_multiclass_roc_curves(classification_results, y_test)
        
        if 'RandomForest' in classification_results:
            self._plot_feature_importance_rf(classification_results['RandomForest'], 
                                            self.classification_data['features'])
        
        self.results['classification'] = classification_results
        self.results['classification_evaluation'] = evaluation_df
        
        return classification_results, evaluation_df
    
    def _plot_confusion_matrices(self, discriminant_results, y_test):
        
        n_methods = len(discriminant_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(7*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, result) in enumerate(discriminant_results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=[f'C{i}' for i in range(cm.shape[0])],
                       yticklabels=[f'C{i}' for i in range(cm.shape[0])])
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(f'{method} 混淆矩阵\nAccuracy: {result["accuracy"]:.4f}')
        
        plt.tight_layout()
        plt.savefig('7.2_confusion_matrices_discriminant.png', dpi=300, bbox_inches='tight')
        print("判别分析混淆矩阵已保存: 7.2_confusion_matrices_discriminant.png")
        
        plt.close()
    
    def _visualize_classification_comparison(self, evaluation_df):
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            evaluation_df.plot(x='Method', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} 对比')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('7.2_classification_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("分类指标对比图已保存: 7.2_classification_metrics_comparison.png")
        
        plt.close()
    
    def _plot_multiclass_roc_curves(self, classification_results, y_test):
        
        from sklearn.preprocessing import label_binarize
        
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (method, result) in enumerate(classification_results.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            y_pred_proba = result['y_pred_proba']
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{method} - ROC Curves (OvR)')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(alpha=0.3)
        
        for idx in range(len(classification_results), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('7.2_multiclass_roc_curves.png', dpi=300, bbox_inches='tight')
        print("多分类ROC曲线已保存: 7.2_multiclass_roc_curves.png")
        
        plt.close()
    
    def _plot_feature_importance_rf(self, rf_result, features):
        
        importance = rf_result['feature_importance']
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv('7.2_feature_importance_rf.csv', index=False, encoding='utf-8-sig')
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('随机森林特征重要性')
        plt.tight_layout()
        plt.savefig('7.2_feature_importance_rf.png', dpi=300, bbox_inches='tight')
        print("随机森林特征重要性图已保存: 7.2_feature_importance_rf.png")
        
        plt.close()
    
    def cross_validation_analysis(self, k=5):
        print(f"\n{k}折交叉验证分析（每折独立聚类）")
        print("-" * 80)
        
        X = self.clustering_data['X']
        X_scaled_full = self.clustering_data['X_scaled']
        cities = self.clustering_data['cities']
        
        if 'optimal_k' in self.results:
            n_clusters = self.results['optimal_k']['recommended']
        else:
            n_clusters = 4
        
        print(f"数据: {len(X)} 个样本")
        print(f"聚类数: {n_clusters}")
        
        k_adjusted = min(k, len(X) // 10)
        k_adjusted = max(k_adjusted, 3)  # 至少3折
        
        if k_adjusted < k:
            print(f"注意: 将{k}折调整为{k_adjusted}折")
        
        models = {
            'LDA': LinearDiscriminantAnalysis(),
            'Logistic': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        kfold = KFold(n_splits=k_adjusted, shuffle=True, random_state=42)
        
        cv_results_all = {name: [] for name in models.keys()}
        
        print(f"\n开始{k_adjusted}折交叉验证...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"\n折 {fold_idx + 1}/{k_adjusted}")
            
            X_train_fold = X_scaled_full[train_idx]
            X_test_fold = X_scaled_full[test_idx]
            
            kmeans_fold = KMeans(n_clusters=n_clusters, init='k-means++', 
                                n_init=10, random_state=42)
            y_train_fold = kmeans_fold.fit_predict(X_train_fold)
            y_test_fold = kmeans_fold.predict(X_test_fold)
            
            unique_labels, label_counts = np.unique(y_train_fold, return_counts=True)
            valid_labels = unique_labels[label_counts >= 2]
            
            if len(valid_labels) < 2:
                print(f"  跳过（类别不足）")
                continue
            
            train_mask = np.isin(y_train_fold, valid_labels)
            test_mask = np.isin(y_test_fold, valid_labels)
            
            X_train_filtered = X_train_fold[train_mask]
            y_train_filtered = y_train_fold[train_mask]
            X_test_filtered = X_test_fold[test_mask]
            y_test_filtered = y_test_fold[test_mask]
            
            if len(np.unique(y_train_filtered)) < 2 or len(y_test_filtered) == 0:
                print(f"  跳过（数据不足）")
                continue
            
            for name, model in models.items():
                try:
                    model_fold = self.clone(model)
                    model_fold.fit(X_train_filtered, y_train_filtered)
                    score = model_fold.score(X_test_filtered, y_test_filtered)
                    cv_results_all[name].append(score)
                except Exception as e:
                    print(f"  {name} 失败: {str(e)}")
                    continue
        
        cv_results = []
        
        for name, scores in cv_results_all.items():
            if len(scores) > 0:
                cv_results.append({
                    'Model': name,
                    'Mean_Accuracy': np.mean(scores),
                    'Std_Accuracy': np.std(scores),
                    'Min_Accuracy': np.min(scores),
                    'Max_Accuracy': np.max(scores),
                    'N_Folds': len(scores)
                })
        
        if not cv_results:
            print("\n警告: 交叉验证未能成功")
            return None
        
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv('7.2_cross_validation_results.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n交叉验证结果:")
        print(cv_df.round(4))
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(cv_df))
        plt.bar(x_pos, cv_df['Mean_Accuracy'], yerr=cv_df['Std_Accuracy'],
               capsize=5, alpha=0.7, edgecolor='black')
        plt.xticks(x_pos, cv_df['Model'])
        plt.ylabel('Accuracy')
        plt.title(f'{k_adjusted}折交叉验证结果（每折独立聚类）')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('7.2_cross_validation_bar.png', dpi=300, bbox_inches='tight')
        print("交叉验证结果图已保存: 7.2_cross_validation_bar.png")
        
        plt.close()
        
        self.results['cross_validation'] = cv_df
        
        return cv_df
    
    def generate_chapter7_report(self):
        print("\n" + "=" * 80)
        print("生成综合报告")
        print("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("聚类与分类分析 - 综合报告（城市-年份面板数据）")
        report.append("=" * 80)
        report.append(f"\n生成时间: {pd.Timestamp.now()}")
        
        report.append("\n数据概况")
        report.append("-" * 80)
        if self.clustering_data is not None:
            report.append(f"样本数量（城市-年份对）: {len(self.clustering_data['X_scaled'])}")
            report.append(f"城市数量: {len(np.unique(self.clustering_data['cities']))}")
            report.append(f"年份数量: {len(np.unique(self.clustering_data['years']))}")
            report.append(f"特征数量: {len(self.clustering_data['features'])}")
        
        report.append("\n聚类分析")
        report.append("-" * 80)
        
        if 'optimal_k' in self.results:
            report.append(f"推荐聚类数: k = {self.results['optimal_k']['recommended']}")
        
        if 'clustering_comparison' in self.results:
            best_method = self.results['clustering_comparison'].loc[
                self.results['clustering_comparison']['Silhouette'].idxmax(), 'Method'
            ]
            best_silhouette = self.results['clustering_comparison']['Silhouette'].max()
            report.append(f"最优聚类算法: {best_method} (Silhouette: {best_silhouette:.4f})")
        
        if 'stability' in self.results:
            stability = self.results['stability']['summary']
            report.append(f"聚类稳定性:")
            report.append(f"  Jaccard系数: {stability['Jaccard_Mean'].values[0]:.4f} ± {stability['Jaccard_Std'].values[0]:.4f}")
            report.append(f"  ARI: {stability['ARI_Mean'].values[0]:.4f} ± {stability['ARI_Std'].values[0]:.4f}")
        
        if 'cluster_interpretation' in self.results:
            n_clusters = len(self.results['cluster_interpretation']['profiles'])
            report.append(f"识别出 {n_clusters} 个发展模式类别")
        
        report.append("\n判别与分类分析")
        report.append("-" * 80)
        
        if 'classification_evaluation' in self.results:
            eval_df = self.results['classification_evaluation']
            best_method = eval_df.loc[eval_df['Accuracy'].idxmax(), 'Method']
            best_accuracy = eval_df['Accuracy'].max()
            report.append(f"最优分类算法: {best_method} (准确率: {best_accuracy:.4f})")
            
            report.append("\n各算法性能:")
            for _, row in eval_df.iterrows():
                report.append(f"  {row['Method']}: Acc={row['Accuracy']:.4f}, F1={row['F1_Score']:.4f}")
        
        if 'cross_validation' in self.results:
            cv_df = self.results['cross_validation']
            best_cv = cv_df.loc[cv_df['Mean_Accuracy'].idxmax()]
            report.append(f"\n交叉验证最优模型: {best_cv['Model']}")
            report.append(f"  平均准确率: {best_cv['Mean_Accuracy']:.4f} ± {best_cv['Std_Accuracy']:.4f}")
        
        report.append("\n主要发现")
        report.append("-" * 80)
        report.append("1. 使用城市-年份面板数据有效扩充了样本量")
        report.append("2. 聚类分析识别出不同发展阶段和模式的城市群组")
        report.append("3. 分类模型可以准确预测城市发展类型")
        report.append("4. 特征重要性分析揭示了关键驱动因素")
        
        report.append("\n" + "=" * 80)
        report.append("报告结束")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open('chapter7_comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n综合报告已保存: chapter7_comprehensive_report.txt")
        
        return report_text
    
    def run_full_chapter7_analysis(self):
        print("\n" + "=" * 80)
        print("聚类与分类分析 - 完整流程（城市-年份面板数据）")
        print("=" * 80)
        
        print("\n聚类分析")
        self.prepare_clustering_data()
        self.determine_optimal_clusters()
        self.perform_multiple_clustering()
        self.clustering_stability_analysis(n_bootstrap=100)
        self.dynamic_clustering_evolution()
        self.interpret_clustering_results()
        
        print("\n 判别与分类分析")
        print("注意：分类分析独立进行，避免数据泄漏")
        self.prepare_classification_data() 
        self.perform_discriminant_analysis()
        self.perform_classification()
        self.cross_validation_analysis(k=10)
        
        self.generate_chapter7_report()
        
  
        if 'classification_evaluation' in self.results:
            print("\n" + "=" * 80)
            print("分类模型准确率说明")
            print("=" * 80)
            eval_df = self.results['classification_evaluation']
            print("\n训练-测试分离模型:")
            print(eval_df[['Method', 'Accuracy', 'F1_Score']].to_string(index=False))
            
        if 'cross_validation' in self.results:
            print("\n交叉验证结果（每折独立聚类）:")
            cv_df = self.results['cross_validation']
            print(cv_df[['Model', 'Mean_Accuracy', 'Std_Accuracy']].to_string(index=False))
            


    
if __name__ == "__main__":

    print("\n步骤: 聚类与分类分析（城市-年份面板数据）")
    

    try:
        clustering_classification = ClusteringClassificationAnalysis(
            main_data=preprocessor.main_data,
            od_matrix=preprocessor.od_matrix,
            od_yearly=preprocessor.od_yearly
        )
        clustering_classification.run_full_chapter7_analysis()
        
        print("\n" + "=" * 80)
        print("第七章分析完成！")
        print("=" * 80)
    except NameError:
        print("错误: 'preprocessor' 未定义")
        print("请先运行数据预处理步骤，创建 preprocessor 对象")
