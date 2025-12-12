"""
GNN模型调用与结果分析模块
功能：模型预测、结果生成、评估指标计算（准确率、时延、计算效率、模型大小）
"""

import pandas as pd
import numpy as np
import os
import time
import json
import torch
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from a_数据清洗代码 import DataCleaner
from b_模型训练代码 import GNNTrainer, GraphBuilder


class ModelPredictor:
    """GNN模型预测与结果分析类"""
    
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
        self.trainer = GNNTrainer(model_dir=model_dir)
        self.cleaner = DataCleaner(output_dir=model_dir)
        self.cleaner.load_preprocessor()
        
    def load_model(self, filename='gnn_model.pth'):
        """
        加载训练好的模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        self.trainer.load_model(filename)
        print(f"模型类型: GNN ({self.trainer.model_type})")
        if self.trainer.classes is not None:
            print(f"类别数: {len(self.trainer.classes)}")
    
    def predict_single_sample(self, X_sample):
        """
        预测单个样本（用于时延测试）
        
        Parameters:
        -----------
        X_sample : ndarray
            单个样本特征向量（原始特征，未标准化）
        
        Returns:
        --------
        prediction : int
            预测类别
        """
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        # 标准化
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        X_scaled = self.cleaner.transform_features(X_sample, is_train=False)
        
        # 构建图（需要窗口数据，这里使用单个样本重复构建窗口）
        window_size = self.trainer.graph_builder.window_size
        if len(X_scaled) < window_size:
            # 如果样本数不足，重复最后一个样本
            X_window = np.vstack([X_scaled] * window_size)
        else:
            # 使用最后window_size个样本
            X_window = X_scaled[-window_size:]
        
        # 构建图
        graph = self.trainer.graph_builder.build_graph(X_window)
        
        # 预测
        self.trainer.model.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([graph])
            out = self.trainer.model(batch.x, batch.edge_index, batch.batch)
            prediction = out.argmax(dim=1).item()
        
        return prediction
    
    def predict_batch(self, X_batch):
        """
        批量预测
        
        Parameters:
        -----------
        X_batch : ndarray
            批量特征矩阵（原始特征，未标准化）
        
        Returns:
        --------
        predictions : ndarray
            预测类别数组
        """
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        # 标准化
        X_scaled = self.cleaner.transform_features(X_batch, is_train=False)
        
        # 构建图
        graphs = self.trainer.build_graphs_from_data(X_scaled, 
                                                       np.zeros(len(X_scaled)))
        
        # 批量预测
        self.trainer.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(graphs), self.trainer.batch_size):
                batch_graphs = graphs[i:i+self.trainer.batch_size]
                batch = Batch.from_data_list(batch_graphs)
                out = self.trainer.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1).cpu().numpy()
                predictions.extend(pred)
        
        # 将预测结果映射回原始样本
        final_predictions = np.zeros(len(X_batch), dtype=int)
        window_size = self.trainer.graph_builder.window_size
        
        if len(predictions) > 0:
            # 前window_size-1个样本使用第一个预测
            final_predictions[:window_size-1] = predictions[0]
            
            # 后续样本使用对应预测
            for i in range(window_size-1, len(X_batch)):
                seq_idx = i - (window_size - 1)
                if seq_idx < len(predictions):
                    final_predictions[i] = predictions[seq_idx]
                else:
                    final_predictions[i] = predictions[-1]
        
        return final_predictions
    
    def calculate_detection_latency(self, X_sample, n_iterations=1000):
        """
        计算检测时延
        
        Parameters:
        -----------
        X_sample : ndarray
            单个样本特征向量
        n_iterations : int
            迭代次数
        
        Returns:
        --------
        latency_stats : dict
            时延统计信息
        """
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        print(f"\n计算检测时延（迭代 {n_iterations} 次）...")
        
        latencies = []
        
        for i in range(n_iterations):
            start_time = time.perf_counter()
            
            # 预测
            _ = self.predict_single_sample(X_sample)
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        latency_stats = {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
        
        print(f"平均时延: {latency_stats['mean_ms']:.4f} ms")
        print(f"中位数时延: {latency_stats['median_ms']:.4f} ms")
        print(f"P95时延: {latency_stats['p95_ms']:.4f} ms")
        print(f"P99时延: {latency_stats['p99_ms']:.4f} ms")
        
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
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        print(f"\n计算计算效率（批量大小: {len(X_batch)}）...")
        
        # 预热
        _ = self.predict_batch(X_batch[:min(10, len(X_batch))])
        
        # 正式测试
        start_time = time.perf_counter()
        _ = self.predict_batch(X_batch)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        samples_per_second = len(X_batch) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"处理 {len(X_batch)} 个样本耗时: {elapsed_time:.4f} 秒")
        print(f"计算效率: {samples_per_second:.2f} 样本/秒")
        
        return samples_per_second
    
    def calculate_model_size(self):
        """
        计算模型大小
        
        Returns:
        --------
        model_info : dict
            模型信息
        """
        if self.trainer.model is None:
            raise ValueError("模型尚未加载")
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.trainer.model.parameters())
        trainable_params = sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad)
        
        # 计算模型文件大小
        model_file = os.path.join(self.model_dir, 'gnn_model.pth')
        if os.path.exists(model_file):
            file_size_bytes = os.path.getsize(model_file)
            file_size_kb = file_size_bytes / 1024
            file_size_mb = file_size_bytes / (1024 * 1024)
        else:
            file_size_bytes = 0
            file_size_kb = 0
            file_size_mb = 0
        
        # 估算内存占用（参数 + 激活值）
        param_memory_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        estimated_memory_mb = param_memory_mb * 2  # 估算（参数 + 激活值）
        
        model_info = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'file_size_bytes': int(file_size_bytes),
            'file_size_kb': float(file_size_kb),
            'file_size_mb': float(file_size_mb),
            'estimated_memory_mb': float(estimated_memory_mb),
            'model_type': self.trainer.model_type,
            'input_dim': int(self.trainer.input_dim),
            'hidden_dim': int(self.trainer.hidden_dim),
            'output_dim': int(self.trainer.output_dim)
        }
        
        print(f"\n模型信息:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  模型文件大小: {file_size_mb:.2f} MB ({file_size_kb:.2f} KB)")
        print(f"  估算内存占用: {estimated_memory_mb:.2f} MB")
        
        return model_info
    
    def predict_test_file(self, test_file_path):
        """
        预测测试文件并更新result列
        
        Parameters:
        -----------
        test_file_path : str
            测试文件路径
        """
        print(f"\n预测测试文件: {test_file_path}")
        
        # 加载测试数据
        df_test = pd.read_csv(test_file_path)
        print(f"测试数据形状: {df_test.shape}")
        
        # 准备特征
        exclude_cols = ['time', 'group_name', 'sufficient_window_size',
                        'labelMovement', 'result', 'labelArea']
        feature_cols = [col for col in df_test.columns if col not in exclude_cols]
        
        if self.cleaner.feature_columns is not None:
            feature_cols = [col for col in self.cleaner.feature_columns if col in feature_cols]
            X_test = df_test[feature_cols].copy()
            missing_cols = set(self.cleaner.feature_columns) - set(feature_cols)
            if missing_cols:
                print(f"警告: 测试数据缺少特征列: {len(missing_cols)} 个")
                for col in missing_cols:
                    X_test[col] = 0
            X_test = X_test[self.cleaner.feature_columns]
        else:
            X_test = df_test[feature_cols].copy()
        
        # 预测
        predictions = self.predict_batch(X_test.values)
        
        # 更新result列
        df_test['result'] = predictions
        
        # 保存
        df_test.to_csv(test_file_path, index=False)
        print(f"预测结果已保存到: {test_file_path}")
        
        return df_test


def main():
    """主函数：预测测试数据"""
    print("=" * 60)
    print("GNN模型调用与结果分析")
    print("=" * 60)
    
    # 创建预测器
    predictor = ModelPredictor(
        model_dir='model/',
        test_dir='test/',
        output_dir='results/'
    )
    
    # 加载模型
    predictor.load_model('gnn_model.pth')
    
    # 预测测试数据
    test_file_path = 'train/清洗测试数据.csv'
    if os.path.exists(test_file_path):
        predictor.predict_test_file(test_file_path)
    else:
        print(f"测试文件不存在: {test_file_path}")
        print("提示：请先运行数据清洗代码生成清洗测试数据")
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

