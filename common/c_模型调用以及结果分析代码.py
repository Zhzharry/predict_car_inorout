"""
模型调用与结果分析模块
功能：模型预测、结果生成、评估指标计算（准确率、时延、计算效率、模型大小）
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from a_数据清洗代码 import DataCleaner
from b_模型训练代码 import SVDQRLLDATrainer


class ModelPredictor:
    """模型预测与结果分析类"""
    
    def __init__(self, model_dir='model/', test_dir='test/', output_dir='results/'):
        """
        初始化预测器
        
        Parameters:
        -----------
        model_dir : str
            模型目录
        test_dir : str
            测试数据目录
        output_dir : str
            结果输出目录
        """
        self.model_dir = model_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载组件
        self.trainer = SVDQRLLDATrainer(model_dir=model_dir)
        self.cleaner = DataCleaner(output_dir=model_dir)
        self.cleaner.load_preprocessor()
        
    def load_model(self, filename='improved_lda_model.pkl'):
        """
        加载训练好的模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        # 直接调用训练器的load_model方法
        self.trainer.load_model(filename)
        print(f"模型类型: SVD+QR+LU+LDA (改进版)")
    
    def predict_single_sample(self, X_sample):
        """
        预测单个样本（用于时延测试）
        
        Parameters:
        -----------
        X_sample : ndarray
            单个样本特征向量（1D或2D）
        
        Returns:
        --------
        prediction : int
            预测类别
        """
        if self.trainer.W is None:
            raise ValueError("模型尚未加载")
        
        # 确保是2D数组
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # 标准化
        X_scaled = self.cleaner.transform_features(X_sample, is_train=False)
        
        # SVD降维
        X_reduced = self.trainer.apply_svd_transform(X_scaled)
        
        # QR特征筛选
        X_selected = self.trainer.apply_qr_selection(X_reduced)
        
        # 预测（不使用阈值，用于时延测试）
        prediction = self.trainer.predict(X_selected, use_threshold=False)[0]
        return prediction
    
    def predict_batch(self, X_batch, use_threshold=True):
        """
        批量预测
        
        Parameters:
        -----------
        X_batch : ndarray
            批量特征矩阵
        use_threshold : bool
            是否使用阈值调整（默认True）
        
        Returns:
        --------
        predictions : ndarray
            预测标签数组
        """
        if self.trainer.W is None:
            raise ValueError("模型尚未加载")
        
        # 标准化
        X_scaled = self.cleaner.transform_features(X_batch, is_train=False)
        
        # SVD降维
        X_reduced = self.trainer.apply_svd_transform(X_scaled)
        
        # QR特征筛选
        X_selected = self.trainer.apply_qr_selection(X_reduced)
        
        # 预测（优化：使用类别权重以提升准确率，不使用阈值避免预测全为零）
        predictions = self.trainer.predict(X_selected, use_threshold=False, use_class_weight=True)
        return predictions
    
    def predict_test_file(self, cleaned_test_path='train/清洗测试数据.csv', 
                          output_path=None):
        """
        对清洗测试数据进行预测，并将结果填入result列
        
        Parameters:
        -----------
        cleaned_test_path : str
            清洗测试数据路径
        output_path : str or None
            输出文件路径，None则覆盖原文件（不创建新文件）
        """
        if not os.path.exists(cleaned_test_path):
            raise FileNotFoundError(f"文件不存在: {cleaned_test_path}")
        
        # 加载数据
        df_test = pd.read_csv(cleaned_test_path)
        print(f"测试数据形状: {df_test.shape}")
        
        # 准备特征（排除result和labelArea列）
        exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                       'labelMovement', 'result', 'labelArea']
        feature_cols = [col for col in df_test.columns if col not in exclude_cols]
        
        # 确保特征列顺序与训练时一致
        if self.cleaner.feature_columns is not None:
            feature_cols = [col for col in self.cleaner.feature_columns if col in feature_cols]
            X_test = df_test[feature_cols].copy()
            # 如果缺少某些特征列，用0填充
            missing_cols = set(self.cleaner.feature_columns) - set(feature_cols)
            if missing_cols:
                print(f"警告: 测试数据缺少特征列: {len(missing_cols)} 个，将用0填充")
                for col in missing_cols:
                    X_test[col] = 0
            X_test = X_test[self.cleaner.feature_columns]
        else:
            X_test = df_test[feature_cols].copy()
        
        # 预测（优化：使用类别权重，应用时序平滑）
        print("正在预测（优化版）...")
        start_time = time.time()
        predictions = self.predict_batch(X_test.values, use_threshold=False)
        
        # 应用适度的时序平滑以提升稳定性
        predictions = self.trainer.temporal_smoothing(predictions, window_size=9)
        
        predict_time = time.time() - start_time
        
        print(f"预测完成，耗时: {predict_time:.2f} 秒")
        print(f"预测类别分布:\n{pd.Series(predictions).value_counts().sort_index()}")
        
        # 回填到result列
        df_test['result'] = predictions
        
        # 保存结果（覆盖原文件，不创建新文件）
        if output_path is None:
            output_path = cleaned_test_path
        
        df_test.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"预测结果已填入并保存到: {output_path}")
        
        return df_test
    
    def predict_validation_files(self, validation_dir='test/', 
                                 output_results_dir='results/validation_predictions/'):
        """
        对验证集（原始测试目录中的文件）进行预测
        
        Parameters:
        -----------
        validation_dir : str
            验证集目录
        output_results_dir : str
            结果输出目录
        """
        print("\n" + "=" * 60)
        print("对验证集文件进行预测")
        print("=" * 60)
        
        if not os.path.exists(validation_dir):
            print(f"错误: 验证集目录 {validation_dir} 不存在。")
            return
        
        os.makedirs(output_results_dir, exist_ok=True)
        
        for filename in os.listdir(validation_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(validation_dir, filename)
                print(f"正在处理验证文件: {filename}")
                
                try:
                    # 1. 加载原始测试文件
                    original_test_df = pd.read_csv(filepath)
                    
                    # 2. 使用DataCleaner进行预处理
                    processed_test_df = self.cleaner.process_test_data(original_test_df.copy())
                    
                    # 3. 进行预测（优化：使用类别权重，应用时序平滑）
                    predictions = self.predict_batch(processed_test_df, use_threshold=False)
                    
                    # 应用适度的时序平滑以提升稳定性
                    predictions = self.trainer.temporal_smoothing(predictions, window_size=9)
                    
                    # 4. 将预测结果添加到原始DataFrame
                    original_test_df['labelArea'] = predictions
                    original_test_df['result'] = predictions
                    
                    # 5. 保存结果
                    output_filepath = os.path.join(output_results_dir, f"predicted_{filename}")
                    original_test_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
                    print(f"  预测结果已保存到: {output_filepath}")
                    
                except Exception as e:
                    print(f"  处理文件 {filename} 时出错: {e}")
                    continue
    
    def calculate_detection_latency(self, X_sample, n_iterations=1000):
        """
        计算检测时延（单样本预测平均耗时）
        
        Parameters:
        -----------
        X_sample : ndarray
            单个样本特征向量
        n_iterations : int
            重复次数（默认1000）
        
        Returns:
        --------
        latency_stats : dict
            时延统计信息（均值、中位数、P95、P99等）
        """
        print("\n" + "=" * 60)
        print("计算检测时延")
        print("=" * 60)
        
        latencies = []
        
        for i in range(n_iterations):
            start_time = time.perf_counter()
            _ = self.predict_single_sample(X_sample)
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
        
        latencies = np.array(latencies)
        avg_latency_ms = np.mean(latencies)
        median_latency_ms = np.median(latencies)
        p95_latency_ms = np.percentile(latencies, 95)
        p99_latency_ms = np.percentile(latencies, 99)
        std_latency_ms = np.std(latencies)
        min_latency_ms = np.min(latencies)
        max_latency_ms = np.max(latencies)
        
        print(f"重复 {n_iterations} 次预测")
        print(f"平均时延: {avg_latency_ms:.4f} 毫秒")
        print(f"中位数时延: {median_latency_ms:.4f} 毫秒")
        print(f"P95时延: {p95_latency_ms:.4f} 毫秒")
        print(f"P99时延: {p99_latency_ms:.4f} 毫秒")
        print(f"标准差: {std_latency_ms:.4f} 毫秒")
        print(f"最小时延: {min_latency_ms:.4f} 毫秒")
        print(f"最大时延: {max_latency_ms:.4f} 毫秒")
        
        latency_stats = {
            'mean': avg_latency_ms,
            'median': median_latency_ms,
            'p95': p95_latency_ms,
            'p99': p99_latency_ms,
            'std': std_latency_ms,
            'min': min_latency_ms,
            'max': max_latency_ms
        }
        
        return latency_stats
    
    def calculate_computational_efficiency(self, X_batch):
        """
        计算计算效率（样本/秒）
        
        Parameters:
        -----------
        X_batch : ndarray
            批量特征矩阵
        
        Returns:
        --------
        samples_per_second : float
            每秒处理的样本数
        """
        print("\n" + "=" * 60)
        print("计算计算效率")
        print("=" * 60)
        
        n_samples = len(X_batch)
        
        start_time = time.perf_counter()
        _ = self.predict_batch(X_batch)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        samples_per_second = n_samples / total_time
        
        print(f"样本数: {n_samples}")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"计算效率: {samples_per_second:.2f} 样本/秒")
        
        return samples_per_second
    
    def calculate_model_size(self):
        """
        计算模型大小（MB）和模型信息
        
        Returns:
        --------
        model_info : dict
            模型信息（文件大小、参数量、SVD/QR/LDA配置等）
        """
        print("\n" + "=" * 60)
        print("计算模型大小")
        print("=" * 60)
        
        if self.trainer.W is None:
            raise ValueError("模型尚未加载")
        
        # 获取模型文件大小
        model_file = os.path.join(self.model_dir, 'improved_lda_model.pkl')
        if os.path.exists(model_file):
            file_size_bytes = os.path.getsize(model_file)
            file_size_mb = file_size_bytes / (1024 * 1024)
        else:
            file_size_mb = 0
        
        # 统计核心参数量
        n_params_w = self.trainer.W.shape[0] * self.trainer.W.shape[1] if self.trainer.W is not None else 0
        n_components = self.trainer.W.shape[1] if self.trainer.W is not None else 0
        n_params_svd = self.trainer.k_svd if self.trainer.k_svd is not None else 0
        n_params_qr = len(self.trainer.selected_feature_indices) if self.trainer.selected_feature_indices is not None else 0
        
        # 计算总参数量（SVD的VT矩阵、QR选择的特征、LDA的W矩阵）
        total_params = 0
        if self.trainer.VT is not None and self.trainer.k_svd is not None:
            # VT矩阵的前k_svd行
            total_params += self.trainer.VT[:self.trainer.k_svd, :].size
        if self.trainer.selected_feature_indices is not None:
            total_params += len(self.trainer.selected_feature_indices)
        total_params += n_params_w
        
        print(f"模型文件大小: {file_size_mb:.2f} MB")
        print(f"SVD降维后维度: {n_params_svd}")
        print(f"QR筛选后特征数: {n_params_qr}")
        print(f"LDA判别向量数量: {n_components}")
        print(f"LDA判别向量总维度: {n_params_w}")
        print(f"总参数量: {total_params}")
        
        model_info = {
            'file_size_mb': file_size_mb,
            'n_params_svd': n_params_svd,
            'n_params_qr': n_params_qr,
            'n_components_lda': n_components,
            'n_params_lda': n_params_w,
            'total_params': total_params
        }
        
        return model_info
    
    def evaluate_performance(self, X_test, y_test):
        """
        评估模型性能（包括所有指标）
        
        Parameters:
        -----------
        X_test : ndarray
            测试特征矩阵
        y_test : ndarray
            测试标签
        
        Returns:
        --------
        performance_metrics : dict
            性能指标字典
        """
        print("\n" + "=" * 60)
        print("完整性能评估")
        print("=" * 60)
        
        # 1. 预测准确率
        y_pred = self.predict_batch(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n1. 预测准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 2. 检测时延（使用第一个样本）
        if len(X_test) > 0:
            latency_stats = self.calculate_detection_latency(X_test[0], n_iterations=1000)
        else:
            latency_stats = None
        
        # 3. 计算效率
        samples_per_second = self.calculate_computational_efficiency(X_test)
        
        # 4. 模型大小
        model_info = self.calculate_model_size()
        
        performance_metrics = {
            'accuracy': accuracy,
            'latency_stats': latency_stats,
            'computational_efficiency_samples_per_sec': samples_per_second,
            'model_info': model_info,
            'n_parameters': model_info.get('total_params', 0) if model_info else 0
        }
        
        return performance_metrics
    
    def generate_evaluation_report(self, csv_path='train/清洗测试数据.csv', output_path=None):
        """
        生成评估报告（类似random方案的格式）
        
        Parameters:
        -----------
        csv_path : str
            包含预测结果和真实标签的CSV文件路径
        output_path : str
            输出报告文件路径（如果为None，则自动生成）
        """
        print("\n" + "=" * 60)
        print("生成评估报告")
        print("=" * 60)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 检查必要的列
        if 'result' not in df.columns or 'labelArea' not in df.columns:
            raise ValueError("CSV文件必须包含 'result' 和 'labelArea' 列")
        
        # 准备数据
        y_true = df['labelArea'].copy()
        y_pred = df['result'].copy()
        
        # 处理缺失值
        valid_mask = ~(y_true.isna() | y_pred.isna())
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("没有有效样本进行评估")
        
        # 转换为数值类型
        y_true = pd.to_numeric(y_true, errors='coerce').astype(int)
        y_pred = pd.to_numeric(y_pred, errors='coerce').astype(int)
        
        # 1. 预测准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 2. 检测时延（使用第一个有效样本）
        exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                       'labelMovement', 'result', 'labelArea']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if self.cleaner.feature_columns is not None:
            feature_cols = [col for col in self.cleaner.feature_columns if col in feature_cols]
            X_sample_df = df[feature_cols].iloc[0:1].copy()
            # 如果缺少某些特征列，用0填充
            missing_cols = set(self.cleaner.feature_columns) - set(feature_cols)
            if missing_cols:
                for col in missing_cols:
                    X_sample_df[col] = 0
            X_sample_df = X_sample_df[self.cleaner.feature_columns]
            X_sample = X_sample_df.values
        else:
            X_sample = df[feature_cols].iloc[0:1].values
        
        # 标准化
        X_sample_scaled = self.cleaner.transform_features(X_sample, is_train=False)
        # SVD降维
        X_sample_reduced = self.trainer.apply_svd_transform(X_sample_scaled)
        # QR特征筛选
        X_sample_selected = self.trainer.apply_qr_selection(X_sample_reduced)
        
        latency_stats = self.calculate_detection_latency(X_sample_selected[0], n_iterations=1000)
        
        # 3. 计算效率
        if self.cleaner.feature_columns is not None:
            X_batch_df = df[[col for col in self.cleaner.feature_columns if col in df.columns]].copy()
            missing_cols = set(self.cleaner.feature_columns) - set(X_batch_df.columns)
            if missing_cols:
                for col in missing_cols:
                    X_batch_df[col] = 0
            X_batch_df = X_batch_df[self.cleaner.feature_columns]
            X_batch = X_batch_df.values
        else:
            X_batch = df[feature_cols].values
        
        # 标准化、SVD、QR
        X_batch_scaled = self.cleaner.transform_features(X_batch, is_train=False)
        X_batch_reduced = self.trainer.apply_svd_transform(X_batch_scaled)
        X_batch_selected = self.trainer.apply_qr_selection(X_batch_reduced)
        samples_per_second = self.calculate_computational_efficiency(X_batch_selected)
        avg_time_per_sample_ms = 1000 / samples_per_second if samples_per_second > 0 else 0
        
        # 4. 模型大小
        model_info = self.calculate_model_size()
        
        # 生成报告
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 60 + "\n")
            f.write("\n")
            
            f.write("[1] 预测准确率\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write("\n")
            
            f.write("[2] 检测时延\n")
            f.write(f"均值: {latency_stats['mean']:.4f} ms\n")
            f.write(f"中位数: {latency_stats['median']:.4f} ms\n")
            f.write(f"P95: {latency_stats['p95']:.4f} ms\n")
            f.write(f"P99: {latency_stats['p99']:.4f} ms\n")
            f.write("\n")
            
            f.write("[3] 计算效率\n")
            f.write(f"吞吐量: {samples_per_second:.2f} 样本/秒\n")
            f.write(f"平均单样本耗时: {avg_time_per_sample_ms:.4f} ms\n")
            f.write("\n")
            
            f.write("[4] 模型大小\n")
            f.write(f"文件大小: {model_info['file_size_mb']:.2f} MB\n")
            f.write(f"SVD降维维度: {model_info['n_params_svd']}\n")
            f.write(f"QR筛选特征数: {model_info['n_params_qr']}\n")
            f.write(f"LDA判别向量数量: {model_info['n_components_lda']}\n")
            f.write(f"LDA判别向量总维度: {model_info['n_params_lda']}\n")
            f.write(f"总参数量: {model_info['total_params']}\n")
        
        print(f"\n评估报告已保存到: {output_path}")
        
        return {
            'accuracy': accuracy,
            'latency_stats': latency_stats,
            'samples_per_second': samples_per_second,
            'avg_time_per_sample_ms': avg_time_per_sample_ms,
            'model_info': model_info
        }


def main():
    """主函数：演示模型预测和评估流程"""
    print("=" * 60)
    print("模型调用与结果分析")
    print("=" * 60)
    
    # 创建预测器
    predictor = ModelPredictor(
        model_dir='model/',
        test_dir='test/',
        output_dir='results/'
    )
    
    # 加载模型
    predictor.load_model('improved_lda_model.pkl')
    
    # 对清洗测试数据进行预测（直接覆盖原文件）
    cleaned_test_path = 'train/清洗测试数据.csv'
    if os.path.exists(cleaned_test_path):
        print("\n" + "=" * 60)
        print("对清洗测试数据进行预测并填入result列")
        print("=" * 60)
        predictor.predict_test_file(cleaned_test_path, output_path=cleaned_test_path)
        print("\n预测完成！结果已填入 train/清洗测试数据.csv 的 result 列")
        print("\n提示: 请按以下顺序执行：")
        print("  1. 运行 d_数据回填代码.py 来回填真实标签")
        print("  2. 运行 e_指标计算输出报告代码.py 来生成完整的评估报告（包括准确率、时延、计算效率、模型大小等）")
    else:
        print(f"错误: 文件不存在: {cleaned_test_path}")
        print("请先运行数据清洗代码生成清洗测试数据文件")
    
    print("\n" + "=" * 60)
    print("模型调用与结果分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

