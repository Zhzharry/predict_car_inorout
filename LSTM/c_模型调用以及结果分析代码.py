"""
LSTM模型调用与结果分析模块
功能：模型预测、结果生成、评估指标计算（准确率、时延、计算效率、模型大小）
"""

import pandas as pd
import numpy as np
import os
import time
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from a_数据清洗代码 import DataCleaner

# 导入 LSTMTrainer，如果 TensorFlow 导入失败会有友好提示
try:
    from b_模型训练代码 import LSTMTrainer
except ImportError as e:
    if "tensorflow" in str(e).lower() or "DLL" in str(e):
        print("\n" + "=" * 80)
        print("❌ TensorFlow 导入失败！")
        print("=" * 80)
        print("请查看 LSTM/安装指南.md 获取详细安装说明")
        print("=" * 80 + "\n")
    raise


class ModelPredictor:
    """LSTM模型预测与结果分析类"""
    
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
        self.trainer = LSTMTrainer(model_dir=model_dir)
        self.cleaner = DataCleaner(output_dir=model_dir)
        self.cleaner.load_preprocessor()
        
    def load_model(self, filename='lstm_model.h5'):
        """
        加载训练好的模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        self.trainer.load_model(filename)
        print(f"模型类型: LSTM")
    
    def create_sequences(self, X, sequence_length=None):
        """
        将数据转换为时序序列（用于预测）
        
        Parameters:
        -----------
        X : ndarray
            特征矩阵 (n_samples, n_features)
        sequence_length : int
            时间步长（滑动窗口大小）
        
        Returns:
        --------
        X_sequences : ndarray
            时序序列 (n_sequences, sequence_length, n_features)
        """
        if sequence_length is None:
            sequence_length = self.trainer.sequence_length
        
        n_samples = len(X)
        n_features = X.shape[1]
        n_sequences = n_samples - sequence_length + 1
        
        if n_sequences <= 0:
            # 如果样本数不足，使用填充
            X_padded = np.zeros((sequence_length, n_features))
            X_padded[-n_samples:] = X
            X_sequences = X_padded.reshape(1, sequence_length, n_features)
            return X_sequences, 0  # 返回填充偏移量
        
        X_sequences = np.zeros((n_sequences, sequence_length, n_features))
        for i in range(n_sequences):
            X_sequences[i] = X[i:i+sequence_length]
        
        return X_sequences, 0
    
    def predict_single_sample(self, X_sample):
        """
        预测单个样本（用于时延测试）
        注意：LSTM需要序列，所以这里传入的应该是序列
        
        Parameters:
        -----------
        X_sample : ndarray
            单个序列样本 (sequence_length, n_features) 或 (1, sequence_length, n_features)
        
        Returns:
        --------
        prediction : int
            预测类别
        """
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        # 确保是3D数组 (1, sequence_length, n_features)
        if X_sample.ndim == 2:
            X_sample = X_sample.reshape(1, -1, X_sample.shape[1])
        elif X_sample.ndim == 1:
            raise ValueError("LSTM需要序列输入，不能是1D数组")
        
        # 预测
        prediction_proba = self.trainer.model.predict(X_sample, verbose=0)
        prediction = np.argmax(prediction_proba, axis=1)[0]
        
        return prediction
    
    def predict_batch(self, X_batch, use_threshold=True):
        """
        批量预测
        
        Parameters:
        -----------
        X_batch : ndarray
            批量特征矩阵 (n_samples, n_features)
        use_threshold : bool
            是否使用阈值调整（LSTM不使用，保留接口兼容性）
        
        Returns:
        --------
        predictions : ndarray
            预测标签数组
        """
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        # 构造序列
        X_sequences, offset = self.create_sequences(X_batch)
        
        # 预测
        predictions_proba = self.trainer.model.predict(X_sequences, verbose=0)
        predictions_seq = np.argmax(predictions_proba, axis=1)
        
        # 将序列预测映射回原始样本
        # 对于前sequence_length-1个样本，使用第一个序列的预测
        # 对于后续样本，使用对应序列的预测
        final_predictions = np.zeros(len(X_batch), dtype=int)
        seq_len = self.trainer.sequence_length
        
        if len(X_sequences) > 0:
            # 前seq_len-1个样本使用第一个序列的预测
            final_predictions[:seq_len-1] = predictions_seq[0]
            
            # 后续样本使用对应序列的预测
            for i in range(seq_len-1, len(X_batch)):
                seq_idx = i - (seq_len - 1)
                if seq_idx < len(predictions_seq):
                    final_predictions[i] = predictions_seq[seq_idx]
                else:
                    final_predictions[i] = predictions_seq[-1]
        else:
            # 如果只有一个序列（填充的情况），所有样本使用该序列的预测
            final_predictions[:] = predictions_seq[0]
        
        return final_predictions
    
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
        
        # 标准化
        X_test_scaled = self.cleaner.transform_features(X_test.values, is_train=False)
        
        # 预测
        print("正在预测...")
        start_time = time.time()
        predictions = self.predict_batch(X_test_scaled)
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
    
    def calculate_detection_latency(self, X_sample, n_iterations=1000):
        """
        计算检测时延（单样本预测平均耗时）
        
        Parameters:
        -----------
        X_sample : ndarray
            单个样本特征向量（需要转换为序列）
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
        
        # 将单个样本转换为序列（使用历史数据填充）
        # 为了测试时延，我们使用该样本重复构造序列
        seq_len = self.trainer.sequence_length
        X_seq = np.tile(X_sample, (seq_len, 1)).reshape(1, seq_len, -1)
        
        latencies = []
        
        for i in range(n_iterations):
            start_time = time.perf_counter()
            _ = self.predict_single_sample(X_seq)
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
            模型信息（文件大小、参数量、LSTM配置等）
        """
        print("\n" + "=" * 60)
        print("计算模型大小")
        print("=" * 60)
        
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        # 获取模型文件大小
        model_file = os.path.join(self.model_dir, 'lstm_model.h5')
        if os.path.exists(model_file):
            file_size_bytes = os.path.getsize(model_file)
            file_size_mb = file_size_bytes / (1024 * 1024)
        else:
            file_size_mb = 0
        
        # 统计参数量
        total_params = self.trainer.model.count_params()
        
        print(f"模型文件大小: {file_size_mb:.2f} MB")
        print(f"序列长度: {self.trainer.sequence_length}")
        print(f"LSTM单元数: {self.trainer.lstm_units}")
        print(f"特征维度: {self.trainer.n_features}")
        print(f"类别数: {self.trainer.n_classes}")
        print(f"总参数量: {total_params:,}")
        
        model_info = {
            'file_size_mb': file_size_mb,
            'sequence_length': self.trainer.sequence_length,
            'lstm_units': self.trainer.lstm_units,
            'n_features': self.trainer.n_features,
            'n_classes': self.trainer.n_classes,
            'total_params': total_params
        }
        
        return model_info


def main():
    """主函数：演示模型预测和评估流程"""
    print("=" * 60)
    print("LSTM模型调用与结果分析")
    print("=" * 60)
    
    # 创建预测器
    predictor = ModelPredictor(
        model_dir='model/',
        test_dir='test/',
        output_dir='results/'
    )
    
    # 加载模型
    predictor.load_model('lstm_model.h5')
    
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

