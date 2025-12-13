"""
改进版 SVD+QR+LU+LDA 模型训练模块
主要改进：
1. SMOTE过采样处理类别不平衡
2. 改进的LDA求解方法（使用广义特征值分解）
3. 基于概率的分类决策（带阈值调整）
4. 集成多个判别向量
5. 添加后处理逻辑（时序平滑）
"""

import pandas as pd
import numpy as np
from scipy.linalg import svd, qr, lu, eigh
from scipy.linalg import LinAlgError
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import os
import pickle
import json
import time
from datetime import datetime
from a_数据清洗代码 import DataCleaner


class SVDQRLLDATrainer:
    """改进版 SVD+QR+LU+LDA 模型训练类"""
    
    def __init__(self, model_dir='model/', random_state=42):
        self.model_dir = model_dir
        self.random_state = random_state
        os.makedirs(model_dir, exist_ok=True)
        np.random.seed(random_state)
        
        # SVD相关
        self.U = None
        self.S = None
        self.VT = None
        self.k_svd = None
        
        # QR相关
        self.selected_feature_indices = None
        self.feature_importance = None
        
        # LDA相关（支持多判别向量）
        self.W = None  # 判别向量矩阵（多个向量）
        self.class_means = None
        self.class_covs = None  # 各类别协方差
        self.classes = None
        self.class_priors = None
        
        # 分类阈值（用于调整决策边界）
        self.thresholds = None
        
    def smote_oversample(self, X, y, k_neighbors=5):
        """
        SMOTE过采样处理类别不平衡
        """
        print("\n" + "=" * 60)
        print("SMOTE过采样")
        print("=" * 60)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # 统计各类别样本数
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = max(class_counts)
        
        print(f"原始类别分布:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  类别 {cls}: {count} ({count/len(y)*100:.2f}%)")
        
        X_resampled = []
        y_resampled = []
        
        for cls in unique_classes:
            X_cls = X[y == cls]
            n_samples = len(X_cls)
            
            # 保留原始样本
            X_resampled.append(X_cls)
            y_resampled.extend([cls] * n_samples)
            
            # 如果是少数类，进行过采样
            if n_samples < max_count * 0.8:  # 少于最大类80%的样本数
                n_synthetic = int(max_count * 0.8) - n_samples
                
                # SMOTE合成
                synthetic_samples = []
                for _ in range(n_synthetic):
                    # 随机选择一个样本
                    idx = np.random.randint(0, n_samples)
                    sample = X_cls[idx]
                    
                    # 找k个最近邻
                    distances = np.sum((X_cls - sample) ** 2, axis=1)
                    k_nearest_idx = np.argsort(distances)[1:k_neighbors+1]
                    
                    # 随机选择一个近邻
                    neighbor_idx = np.random.choice(k_nearest_idx)
                    neighbor = X_cls[neighbor_idx]
                    
                    # 在样本和近邻之间线性插值
                    alpha = np.random.random()
                    synthetic = sample + alpha * (neighbor - sample)
                    synthetic_samples.append(synthetic)
                
                if synthetic_samples:
                    X_resampled.append(np.array(synthetic_samples))
                    y_resampled.extend([cls] * len(synthetic_samples))
        
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.array(y_resampled)
        
        print(f"\n过采样后类别分布:")
        unique_classes_new, class_counts_new = np.unique(y_resampled, return_counts=True)
        for cls, count in zip(unique_classes_new, class_counts_new):
            print(f"  类别 {cls}: {count} ({count/len(y_resampled)*100:.2f}%)")
        
        return X_resampled, y_resampled
    
    def svd_decomposition(self, X_train, cumulative_ratio=0.99):
        """SVD降噪与降维（提高保留率到99%）"""
        print("=" * 60)
        print("SVD降噪与降维")
        print("=" * 60)
        
        print(f"原始特征矩阵形状: {X_train.shape}")
        
        start_time = time.time()
        U, S, VT = svd(X_train, full_matrices=False)
        svd_time = time.time() - start_time
        print(f"SVD分解完成，耗时: {svd_time:.2f} 秒")
        
        self.U = U
        self.S = S
        self.VT = VT
        
        # 计算累计贡献率
        S_squared = S ** 2
        cumulative_contribution = np.cumsum(S_squared) / np.sum(S_squared)
        
        k = np.where(cumulative_contribution >= cumulative_ratio)[0]
        self.k_svd = k[0] + 1 if len(k) > 0 else len(S)
        
        print(f"保留的奇异值数量: {self.k_svd}/{len(S)}")
        print(f"累计贡献率: {cumulative_contribution[self.k_svd-1]:.4f}")
        
        # 投影到低维空间
        U_k = U[:, :self.k_svd]
        S_k = S[:self.k_svd]
        X_train_reduced = U_k * S_k
        
        print(f"降维后特征矩阵形状: {X_train_reduced.shape}")
        
        return X_train_reduced
    
    def apply_svd_transform(self, X):
        """应用SVD降维"""
        if self.VT is None or self.k_svd is None:
            raise ValueError("请先对训练数据执行SVD分解")
        
        VT_k = self.VT[:self.k_svd, :]
        X_reduced = X @ VT_k.T
        return X_reduced
    
    def qr_feature_selection(self, X_train_reduced, y_train, feature_ratio=0.90):
        """QR分解筛选关键特征（提高到90%）"""
        print("\n" + "=" * 60)
        print("QR分解筛选关键特征")
        print("=" * 60)
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        y_col = y_train.reshape(-1, 1)
        augmented_matrix = np.hstack([X_train_reduced, y_col])
        
        Q, R = qr(augmented_matrix, mode='economic')
        
        n_features = X_train_reduced.shape[1]
        R_features = R[:n_features, :n_features]
        
        feature_importance = np.abs(np.diag(R_features))
        self.feature_importance = feature_importance
        
        sorted_indices = np.argsort(feature_importance)[::-1]
        n_selected = int(n_features * feature_ratio)
        self.selected_feature_indices = sorted_indices[:n_selected]
        
        print(f"筛选后特征数: {n_selected}/{n_features} ({feature_ratio*100:.1f}%)")
        
        X_train_selected = X_train_reduced[:, self.selected_feature_indices]
        return X_train_selected
    
    def apply_qr_selection(self, X_reduced):
        """应用QR特征筛选"""
        if self.selected_feature_indices is None:
            raise ValueError("请先对训练数据执行QR特征筛选")
        return X_reduced[:, self.selected_feature_indices]
    
    def improved_lda_solve(self, X_train_selected, y_train, n_components=2, regularization=1e-4):
        """
        改进的LDA求解（使用广义特征值分解，获取多个判别向量）
        """
        print("\n" + "=" * 60)
        print("改进的LDA求解（多判别向量）")
        print("=" * 60)
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        n_features = X_train_selected.shape[1]
        
        print(f"类别数: {n_classes}, 特征数: {n_features}")
        
        # 计算总体均值
        overall_mean = np.mean(X_train_selected, axis=0)
        
        # 计算类内和类间散度矩阵
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        
        for cls in self.classes:
            X_cls = X_train_selected[y_train == cls]
            n_cls = len(X_cls)
            
            if n_cls > 0:
                class_mean = np.mean(X_cls, axis=0)
                
                # 类内散度
                X_cls_centered = X_cls - class_mean
                Sw += X_cls_centered.T @ X_cls_centered
                
                # 类间散度
                mean_diff = (class_mean - overall_mean).reshape(-1, 1)
                Sb += n_cls * (mean_diff @ mean_diff.T)
        
        # 正则化
        Sw += regularization * np.eye(n_features)
        
        # 求解广义特征值问题: Sb * w = lambda * Sw * w
        try:
            eigenvalues, eigenvectors = eigh(Sb, Sw)
            
            # 选择最大的n_components个特征值对应的特征向量
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 最多n_classes-1个有效判别向量
            n_components = min(n_components, n_classes - 1, n_features)
            
            self.W = eigenvectors[:, :n_components]
            
            print(f"保留 {n_components} 个判别向量")
            print(f"前{n_components}个特征值: {eigenvalues[:n_components]}")
            
        except Exception as e:
            print(f"广义特征值分解失败: {e}")
            print("使用备用方法...")
            
            # 备用方法：使用伪逆
            Sw_inv = np.linalg.pinv(Sw)
            A = Sw_inv @ Sb
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            n_components = min(n_components, n_classes - 1)
            self.W = np.real(eigenvectors[:, :n_components])
        
        # 归一化判别向量
        for i in range(self.W.shape[1]):
            self.W[:, i] = self.W[:, i] / np.linalg.norm(self.W[:, i])
        
        return self.W
    
    def train_lda_model(self, X_train_selected, y_train):
        """训练LDA模型（计算各类别统计量）"""
        print("\n" + "=" * 60)
        print("训练LDA模型")
        print("=" * 60)
        
        if self.W is None:
            raise ValueError("请先求解LDA判别向量")
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # 投影到判别空间
        X_projected = X_train_selected @ self.W
        
        # 计算各类别的统计量
        self.class_means = {}
        self.class_covs = {}
        self.class_priors = {}
        total_samples = len(y_train)
        
        for cls in self.classes:
            cls_mask = y_train == cls
            X_cls_proj = X_projected[cls_mask]
            
            self.class_means[cls] = np.mean(X_cls_proj, axis=0)
            self.class_covs[cls] = np.cov(X_cls_proj.T) + 1e-6 * np.eye(self.W.shape[1])
            self.class_priors[cls] = np.sum(cls_mask) / total_samples
            
            print(f"类别 {cls}: 先验概率={self.class_priors[cls]:.4f}, 样本数={np.sum(cls_mask)}")
        
        # 优化分类阈值
        self.optimize_thresholds(X_train_selected, y_train)
    
    def optimize_thresholds(self, X_train, y_train):
        """
        优化分类阈值（针对类别不平衡问题）
        """
        print("\n优化分类阈值...")
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # 计算训练集上的后验概率
        X_proj = X_train @ self.W
        posteriors = self.compute_posteriors(X_proj)
        
        # 对于少数类，降低阈值
        self.thresholds = {}
        for cls in self.classes:
            cls_samples = np.sum(y_train == cls)
            total_samples = len(y_train)
            
            # 根据类别比例调整阈值
            if cls_samples / total_samples < 0.1:  # 少数类
                self.thresholds[cls] = 0.3  # 降低阈值
            else:
                self.thresholds[cls] = 0.5  # 默认阈值
            
            print(f"类别 {cls} 阈值: {self.thresholds[cls]:.2f}")
    
    def compute_posteriors(self, X_projected):
        """
        计算后验概率（使用多元高斯模型）
        """
        n_samples = X_projected.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            mean = self.class_means[cls]
            cov = self.class_covs[cls]
            prior = self.class_priors[cls]
            
            # 多元高斯概率密度
            diff = X_projected - mean
            cov_inv = np.linalg.inv(cov)
            
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
            log_likelihood = -0.5 * mahalanobis
            
            posteriors[:, i] = np.log(prior) + log_likelihood
        
        # Softmax归一化
        posteriors = np.exp(posteriors - np.max(posteriors, axis=1, keepdims=True))
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        
        return posteriors
    
    def predict(self, X_selected, use_threshold=True):
        """
        预测（带阈值调整）
        """
        if self.W is None or self.class_means is None:
            raise ValueError("模型尚未训练")
        
        # 投影
        X_proj = X_selected @ self.W
        
        # 计算后验概率
        posteriors = self.compute_posteriors(X_proj)
        
        # 应用阈值
        predictions = np.zeros(len(posteriors), dtype=int)
        
        if use_threshold and self.thresholds is not None:
            for i in range(len(posteriors)):
                # 检查是否有类别超过阈值
                valid_classes = []
                for j, cls in enumerate(self.classes):
                    if posteriors[i, j] >= self.thresholds[cls]:
                        valid_classes.append((cls, posteriors[i, j]))
                
                if valid_classes:
                    # 选择概率最高的
                    predictions[i] = max(valid_classes, key=lambda x: x[1])[0]
                else:
                    # 都不满足，选择概率最高的
                    predictions[i] = self.classes[np.argmax(posteriors[i])]
        else:
            predictions = self.classes[np.argmax(posteriors, axis=1)]
        
        return predictions
    
    def temporal_smoothing(self, predictions, window_size=5):
        """
        时序平滑（避免抖动）
        """
        smoothed = np.copy(predictions)
        
        for i in range(len(predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start:end]
            
            # 使用众数
            counter = Counter(window)
            smoothed[i] = counter.most_common(1)[0][0]
        
        return smoothed
    
    def evaluate_model(self, X_test_selected, y_test, use_smoothing=False):
        """评估模型"""
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        y_pred = self.predict(X_test_selected)
        
        if use_smoothing:
            y_pred = self.temporal_smoothing(y_pred)
        
        accuracy = np.mean(y_pred == y_test)
        print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n各类别准确率:")
        for cls in self.classes:
            cls_mask = y_test == cls
            if np.any(cls_mask):
                cls_accuracy = np.mean(y_pred[cls_mask] == y_test[cls_mask])
                print(f"  类别 {cls}: {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")
        
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        print("\n混淆矩阵:")
        print(cm)
        
        class_names = {0: '其他场景', 1: '进入地库', 2: '出地库'}
        target_names = [class_names.get(cls, f'类别{cls}') for cls in self.classes]
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, labels=self.classes, 
                                  target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_true': y_test
        }
    
    def save_model(self, filename='improved_lda_model.pkl'):
        """保存模型"""
        filepath = os.path.join(self.model_dir, filename)
        
        model_data = {
            'U': self.U,
            'S': self.S,
            'VT': self.VT,
            'k_svd': self.k_svd,
            'selected_feature_indices': self.selected_feature_indices,
            'W': self.W,
            'class_means': self.class_means,
            'class_covs': self.class_covs,
            'class_priors': self.class_priors,
            'classes': self.classes,
            'thresholds': self.thresholds
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n模型已保存到: {filepath}")
        model_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"模型文件大小: {model_size:.2f} MB")
    
    def load_model(self, filename='improved_lda_model.pkl'):
        """加载模型"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 加载所有模型参数
        self.U = model_data.get('U')
        self.S = model_data.get('S')
        self.VT = model_data.get('VT')
        self.k_svd = model_data.get('k_svd')
        self.selected_feature_indices = model_data.get('selected_feature_indices')
        self.W = model_data.get('W')
        self.class_means = model_data.get('class_means')
        self.class_covs = model_data.get('class_covs')
        self.class_priors = model_data.get('class_priors')
        self.classes = model_data.get('classes')
        self.thresholds = model_data.get('thresholds')
        
        print(f"模型已加载: {filepath}")
        if self.W is not None:
            print(f"判别向量数量: {self.W.shape[1]}")
        if self.classes is not None:
            print(f"类别数: {len(self.classes)}")


def main():
    """主函数"""
    print("=" * 60)
    print("改进版 SVD+QR+LU+LDA 模型训练")
    print("=" * 60)
    
    # 1. 数据预处理
    cleaner = DataCleaner(
        train_path='train/训练数据.csv',
        test_dir='test/',
        output_dir='model/'
    )
    
    X_train, X_test, y_train, y_test = cleaner.process_train_data(
        remove_outliers_flag=True,
        fill_missing_flag=True,
        standardize=True,
        test_size=0.3
    )
    
    # 2. 创建改进的训练器
    trainer = SVDQRLLDATrainer(model_dir='model/', random_state=42)
    
    # 3. SMOTE过采样
    X_train_balanced, y_train_balanced = trainer.smote_oversample(X_train, y_train)
    
    # 4. SVD降维（提高到99%）
    X_train_reduced = trainer.svd_decomposition(X_train_balanced, cumulative_ratio=0.99)
    X_test_reduced = trainer.apply_svd_transform(X_test)
    
    # 5. QR特征筛选（提高到90%）
    X_train_selected = trainer.qr_feature_selection(X_train_reduced, y_train_balanced, feature_ratio=0.90)
    X_test_selected = trainer.apply_qr_selection(X_test_reduced)
    
    # 6. 改进的LDA求解（多判别向量）
    trainer.improved_lda_solve(X_train_selected, y_train_balanced, n_components=2)
    
    # 7. 训练模型
    trainer.train_lda_model(X_train_selected, y_train_balanced)
    
    # 8. 评估（带时序平滑）
    results = trainer.evaluate_model(X_test_selected, y_test, use_smoothing=True)
    
    # 9. 保存模型
    trainer.save_model('improved_lda_model.pkl')
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()