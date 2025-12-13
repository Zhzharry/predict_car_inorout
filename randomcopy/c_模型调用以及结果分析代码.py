"""
模型调用与结果分析模块
功能：模型预测、结果生成、评估指标计算（准确率、时延、计算效率、模型大小）
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from a_数据清洗代码 import DataCleaner
from b_模型训练代码 import RandomForestTrainer


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
        self.model = None
        self.selected_features = None
        self.cleaner = DataCleaner(output_dir=model_dir)
        self.cleaner.load_preprocessor()
        
    def load_model(self, filename='random_forest_model.pkl'):
        """
        加载训练好的模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.selected_features = model_data.get('selected_features')
        
        print(f"模型已加载: {filepath}")
        print(f"模型类型: {type(self.model).__name__}")
        
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
        if self.model is None:
            raise ValueError("模型尚未加载")
        
        # 确保是2D数组
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # 特征筛选
        if self.selected_features is not None:
            X_sample = X_sample[:, self.selected_features]
        
        # 预测
        prediction = self.model.predict(X_sample)[0]
        return prediction
    
    def predict_batch(self, X_batch):
        """
        批量预测
        
        Parameters:
        -----------
        X_batch : ndarray
            批量特征矩阵
        
        Returns:
        --------
        predictions : ndarray
            预测结果数组
        """
        if self.model is None:
            raise ValueError("模型尚未加载")
        
        # 特征筛选
        if self.selected_features is not None:
            X_batch = X_batch[:, self.selected_features]
        
        # 预测
        predictions = self.model.predict(X_batch)
        return predictions
    
    def predict_test_file(self, test_filepath, output_filepath=None):
        """
        预测单个测试文件（清洗测试数据CSV）
        注意：不读取labelArea列，避免标签泄露
        
        Parameters:
        -----------
        test_filepath : str
            测试文件路径（清洗测试数据.csv）
        output_filepath : str or None
            输出文件路径，None则覆盖原文件
        
        Returns:
        --------
        df_result : DataFrame
            包含预测结果的DataFrame
        """
        print(f"正在处理测试文件: {test_filepath}")
        
        # 加载测试数据（不读取labelArea列，避免标签泄露）
        df_test = pd.read_csv(test_filepath)
        original_shape = df_test.shape
        
        # 确保不包含labelArea列
        if 'labelArea' in df_test.columns:
            print("警告: 测试数据包含labelArea列，已自动删除以避免标签泄露")
            df_test = df_test.drop(columns=['labelArea'])
        
        # 检查是否有result列
        has_result_col = 'result' in df_test.columns
        
        # 数据清洗和预处理（使用不包含labelArea的数据）
        X_test = self.cleaner.process_test_data(df_test)
        
        # 预测
        predictions = self.predict_batch(X_test)
        
        # 将预测结果写入result列
        df_result = df_test.copy()
        if has_result_col:
            df_result['result'] = predictions
        else:
            # 如果没有result列，在sufficient_window_size之后插入
            cols = df_result.columns.tolist()
            if 'sufficient_window_size' in cols:
                sufficient_idx = cols.index('sufficient_window_size')
                cols.insert(sufficient_idx + 1, 'result')
                df_result = df_result.reindex(columns=cols)
            else:
                df_result['result'] = predictions
        
        df_result['result'] = predictions
        
        # 保存结果
        if output_filepath is None:
            output_filepath = test_filepath
        
        df_result.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        
        print(f"预测完成: {original_shape[0]} 个样本")
        print(f"预测类别分布:\n{pd.Series(predictions).value_counts().sort_index()}")
        print(f"结果已保存到: {output_filepath}")
        
        return df_result
    
    def predict_all_test_files(self, test_file_path=None):
        """
        预测所有测试文件或指定的清洗测试数据文件
        
        Parameters:
        -----------
        test_file_path : str or None
            清洗测试数据文件路径，None则查找test_dir下的所有CSV文件
        
        Returns:
        --------
        results : dict or DataFrame
            如果指定单个文件，返回DataFrame；否则返回字典
        """
        if self.model is None:
            self.load_model()
        
        # 如果指定了清洗测试数据文件路径
        if test_file_path and os.path.exists(test_file_path):
            print(f"处理清洗测试数据文件: {test_file_path}")
            df_result = self.predict_test_file(test_file_path)
            return df_result
        
        # 否则处理test_dir下的所有文件
        results = {}
        
        if os.path.exists(self.test_dir):
            test_files = [f for f in os.listdir(self.test_dir) if f.endswith('.csv')]
            
            print(f"找到 {len(test_files)} 个测试文件")
            
            for filename in test_files:
                filepath = os.path.join(self.test_dir, filename)
                df_result = self.predict_test_file(filepath)
                results[filename] = df_result
                
                # 保存结果
                output_filename = filename.replace('.csv', '_predicted.csv')
                output_path = os.path.join(self.output_dir, output_filename)
                df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"结果已保存: {output_path}\n")
        
        return results
    
    def evaluate_local_test(self, X_test, y_test):
        """
        评估本地测试集
        
        Parameters:
        -----------
        X_test : ndarray
            测试特征矩阵
        y_test : ndarray
            真实标签
        
        Returns:
        --------
        results : dict
            评估结果
        """
        if self.model is None:
            raise ValueError("模型尚未加载")
        
        print("=" * 60)
        print("本地测试集评估")
        print("=" * 60)
        
        # 预测
        start_time = time.time()
        y_pred = self.predict_batch(X_test)
        prediction_time = time.time() - start_time
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 分类报告
        report = classification_report(
            y_test, y_pred,
            target_names=['其他场景', '进入地库', '出地库'],
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n准确率: {accuracy:.4f}")
        print(f"\n预测耗时: {prediction_time:.4f} 秒")
        print(f"平均单样本耗时: {prediction_time / len(y_test) * 1000:.4f} 毫秒")
        print(f"\n分类报告:")
        print(classification_report(
            y_test, y_pred,
            target_names=['其他场景', '进入地库', '出地库']
        ))
        print(f"\n混淆矩阵:")
        print(cm)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': prediction_time,
            'avg_time_per_sample_ms': prediction_time / len(y_test) * 1000,
            'y_pred': y_pred,
            'y_true': y_test,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return results
    
    def measure_latency(self, X_sample, n_iterations=1000):
        """
        测量单样本预测时延（重复多次取平均）
        
        Parameters:
        -----------
        X_sample : ndarray
            单个样本特征向量
        n_iterations : int
            重复次数
        
        Returns:
        --------
        latency_stats : dict
            时延统计信息
        """
        if self.model is None:
            raise ValueError("模型尚未加载")
        
        print("=" * 60)
        print(f"测量单样本预测时延 (重复 {n_iterations} 次)")
        print("=" * 60)
        
        # 准备样本
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # 特征筛选
        if self.selected_features is not None:
            X_sample = X_sample[:, self.selected_features]
        
        # 预热（避免首次调用开销）
        _ = self.model.predict(X_sample)
        
        # 测量时延
        latencies = []
        for i in range(n_iterations):
            start = time.perf_counter()
            _ = self.model.predict(X_sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为毫秒
        
        latencies = np.array(latencies)
        
        latency_stats = {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }
        
        print(f"\n时延统计 (毫秒):")
        print(f"  均值: {latency_stats['mean_ms']:.4f} ms")
        print(f"  中位数: {latency_stats['median_ms']:.4f} ms")
        print(f"  标准差: {latency_stats['std_ms']:.4f} ms")
        print(f"  最小值: {latency_stats['min_ms']:.4f} ms")
        print(f"  最大值: {latency_stats['max_ms']:.4f} ms")
        print(f"  P95: {latency_stats['p95_ms']:.4f} ms")
        print(f"  P99: {latency_stats['p99_ms']:.4f} ms")
        
        return latency_stats
    
    def measure_computational_efficiency(self, X_batch):
        """
        测量计算效率（批量处理吞吐量）
        
        Parameters:
        -----------
        X_batch : ndarray
            批量特征矩阵
        
        Returns:
        --------
        efficiency_stats : dict
            效率统计信息
        """
        if self.model is None:
            raise ValueError("模型尚未加载")
        
        print("=" * 60)
        print("测量计算效率（批量处理）")
        print("=" * 60)
        
        n_samples = len(X_batch)
        
        # 特征筛选
        if self.selected_features is not None:
            X_batch = X_batch[:, self.selected_features]
        
        # 预热
        _ = self.model.predict(X_batch[:100])
        
        # 测量批量预测时间
        start_time = time.time()
        _ = self.model.predict(X_batch)
        total_time = time.time() - start_time
        
        # 计算吞吐量
        throughput = n_samples / total_time  # 样本/秒
        avg_time_per_sample = total_time / n_samples * 1000  # 毫秒
        
        efficiency_stats = {
            'total_samples': n_samples,
            'total_time_sec': total_time,
            'avg_time_per_sample_ms': avg_time_per_sample,
            'throughput_samples_per_sec': throughput
        }
        
        print(f"\n批量处理统计:")
        print(f"  总样本数: {n_samples}")
        print(f"  总耗时: {total_time:.4f} 秒")
        print(f"  平均单样本耗时: {avg_time_per_sample:.4f} 毫秒")
        print(f"  吞吐量: {throughput:.2f} 样本/秒")
        
        return efficiency_stats
    
    def measure_model_size(self, model_filename='random_forest_model.pkl'):
        """
        测量模型大小
        
        Parameters:
        -----------
        model_filename : str
            模型文件名
        
        Returns:
        --------
        size_info : dict
            模型大小信息
        """
        filepath = os.path.join(self.model_dir, model_filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 文件大小
        file_size_bytes = os.path.getsize(filepath)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # 加载模型以计算参数量
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        
        # 估算参数量（随机森林的节点数）
        n_trees = len(model.estimators_)
        total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        
        # 每个节点大约需要存储：特征索引(4字节) + 阈值(8字节) + 左右子节点索引(8字节) + 其他(约20字节)
        # 简化估算：每个节点约40字节
        estimated_params = total_nodes
        estimated_size_bytes = total_nodes * 40  # 粗略估算
        
        size_info = {
            'file_size_bytes': file_size_bytes,
            'file_size_mb': file_size_mb,
            'n_trees': n_trees,
            'total_nodes': total_nodes,
            'avg_nodes_per_tree': total_nodes / n_trees if n_trees > 0 else 0,
            'estimated_params': estimated_params,
            'estimated_size_bytes': estimated_size_bytes
        }
        
        print("=" * 60)
        print("模型大小统计")
        print("=" * 60)
        print(f"文件大小: {file_size_mb:.2f} MB ({file_size_bytes:,} 字节)")
        print(f"决策树数量: {n_trees}")
        print(f"总节点数: {total_nodes:,}")
        print(f"平均每棵树节点数: {size_info['avg_nodes_per_tree']:.1f}")
        print(f"估算参数量: {estimated_params:,}")
        
        return size_info
    
    def generate_evaluation_report(self, X_test, y_test, X_sample=None):
        """
        生成完整的评估报告
        
        Parameters:
        -----------
        X_test : ndarray
            测试特征矩阵
        y_test : ndarray
            测试标签
        X_sample : ndarray or None
            用于时延测试的样本
        
        Returns:
        --------
        report : dict
            完整评估报告
        """
        print("=" * 60)
        print("生成完整评估报告")
        print("=" * 60)
        
        if self.model is None:
            self.load_model()
        
        report = {}
        
        # 1. 预测准确率
        print("\n[1] 预测准确率评估")
        test_results = self.evaluate_local_test(X_test, y_test)
        report['accuracy'] = test_results['accuracy']
        report['classification_report'] = test_results['classification_report']
        report['confusion_matrix'] = test_results['confusion_matrix']
        
        # 2. 检测时延
        print("\n[2] 检测时延测量")
        if X_sample is None:
            X_sample = X_test[0]  # 使用第一个样本
        latency_stats = self.measure_latency(X_sample, n_iterations=1000)
        report['latency'] = latency_stats
        
        # 3. 计算效率
        print("\n[3] 计算效率测量")
        efficiency_stats = self.measure_computational_efficiency(X_test)
        report['efficiency'] = efficiency_stats
        
        # 4. 模型大小
        print("\n[4] 模型大小统计")
        size_info = self.measure_model_size()
        report['model_size'] = size_info
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        self._save_text_report(report, report_path)
        
        return report
    
    def _save_text_report(self, report, filepath):
        """保存文本格式的评估报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 准确率
            f.write("[1] 预测准确率\n")
            f.write(f"准确率: {report['accuracy']:.4f}\n\n")
            
            # 时延
            f.write("[2] 检测时延\n")
            latency = report['latency']
            f.write(f"均值: {latency['mean_ms']:.4f} ms\n")
            f.write(f"中位数: {latency['median_ms']:.4f} ms\n")
            f.write(f"P95: {latency['p95_ms']:.4f} ms\n")
            f.write(f"P99: {latency['p99_ms']:.4f} ms\n\n")
            
            # 计算效率
            f.write("[3] 计算效率\n")
            efficiency = report['efficiency']
            f.write(f"吞吐量: {efficiency['throughput_samples_per_sec']:.2f} 样本/秒\n")
            f.write(f"平均单样本耗时: {efficiency['avg_time_per_sample_ms']:.4f} ms\n\n")
            
            # 模型大小
            f.write("[4] 模型大小\n")
            size_info = report['model_size']
            f.write(f"文件大小: {size_info['file_size_mb']:.2f} MB\n")
            f.write(f"决策树数量: {size_info['n_trees']}\n")
            f.write(f"总节点数: {size_info['total_nodes']:,}\n")
        
        print(f"\n评估报告已保存: {filepath}")


def main():
    """主函数：演示模型预测和评估流程"""
    print("=" * 60)
    print("模型调用与结果分析")
    print("=" * 60)
    
    # 1. 加载数据（用于评估）
    cleaner = DataCleaner(
        train_path='train/训练数据.csv',
        test_dir='test/',
        output_dir='model/'
    )
    
    # 加载预处理对象
    cleaner.load_preprocessor()
    
    # 处理训练数据以获取测试集
    X_train, X_test, y_train, y_test = cleaner.process_train_data(
        remove_outliers_flag=True,
        fill_missing_flag=True,
        standardize=True,
        test_size=0.2
    )
    
    # 2. 创建预测器
    predictor = ModelPredictor(
        model_dir='model/',
        test_dir='test/',
        output_dir='results/'
    )
    
    # 3. 加载模型
    predictor.load_model('random_forest_model.pkl')
    
    # 4. 评估本地测试集
    test_results = predictor.evaluate_local_test(X_test, y_test)
    
    # 5. 测量时延
    latency_stats = predictor.measure_latency(X_test[0], n_iterations=1000)
    
    # 6. 测量计算效率
    efficiency_stats = predictor.measure_computational_efficiency(X_test)
    
    # 7. 测量模型大小
    size_info = predictor.measure_model_size()
    
    # 8. 生成完整报告
    report = predictor.generate_evaluation_report(X_test, y_test, X_test[0])
    
    # 9. 预测清洗测试数据文件
    print("\n" + "=" * 60)
    print("预测清洗测试数据文件")
    print("=" * 60)
    cleaned_test_path = 'train/清洗测试数据.csv'
    if os.path.exists(cleaned_test_path):
        test_file_results = predictor.predict_all_test_files(test_file_path=cleaned_test_path)
    else:
        print(f"警告: 清洗测试数据文件不存在: {cleaned_test_path}")
        print("尝试预测test目录下的所有文件...")
        test_file_results = predictor.predict_all_test_files()
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

