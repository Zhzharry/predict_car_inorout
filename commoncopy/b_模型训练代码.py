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
from sklearn.metrics import confusion_matrix, classification_report, f1_score
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
        
    def smote_oversample(self, X, y, k_neighbors=5, target_ratio=1.0):
        """
        SMOTE过采样处理类别不平衡（优化版：完全平衡）
        
        Parameters:
        -----------
        target_ratio : float
            目标比例，1.0表示完全平衡到最大类
        """
        print("\n" + "=" * 60)
        print("SMOTE过采样（优化版：完全平衡）")
        print("=" * 60)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # 统计各类别样本数
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = max(class_counts)
        target_count = int(max_count * target_ratio)
        
        print(f"原始类别分布:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  类别 {cls}: {count} ({count/len(y)*100:.2f}%)")
        print(f"目标样本数: {target_count} (最大类的 {target_ratio*100:.0f}%)")
        
        X_resampled = []
        y_resampled = []
        
        for cls in unique_classes:
            X_cls = X[y == cls]
            n_samples = len(X_cls)
            
            # 保留原始样本
            X_resampled.append(X_cls)
            y_resampled.extend([cls] * n_samples)
            
            # 如果样本数少于目标数，进行过采样
            if n_samples < target_count:
                n_synthetic = target_count - n_samples
                
                # 改进的SMOTE合成（使用更多样化的合成策略）
                synthetic_samples = []
                for _ in range(n_synthetic):
                    # 随机选择一个样本
                    idx = np.random.randint(0, n_samples)
                    sample = X_cls[idx]
                    
                    # 找k个最近邻（使用更多邻居以提高多样性）
                    k_available = min(k_neighbors, n_samples - 1)
                    if k_available > 0:
                        distances = np.sum((X_cls - sample) ** 2, axis=1)
                        k_nearest_idx = np.argsort(distances)[1:k_available+1]
                        
                        # 随机选择一个近邻
                        neighbor_idx = np.random.choice(k_nearest_idx)
                        neighbor = X_cls[neighbor_idx]
                    else:
                        # 如果样本太少，使用添加小噪声的方式
                        neighbor = sample + np.random.normal(0, 0.01, size=sample.shape)
                    
                    # 在样本和近邻之间线性插值（使用更均匀的分布）
                    alpha = np.random.beta(2, 2)  # Beta分布，更集中在中间
                    synthetic = sample + alpha * (neighbor - sample)
                    
                    # 添加少量噪声以提高多样性
                    noise = np.random.normal(0, 0.001, size=synthetic.shape)
                    synthetic = synthetic + noise
                    
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
    
    def svd_decomposition(self, X_train, cumulative_ratio=0.995):
        """SVD降噪与降维（优化版：提高保留率到99.5%以保留更多信息）"""
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
        
        # 检查特征数量是否匹配
        expected_features = self.VT.shape[1]  # VT的列数应该等于原始特征数
        actual_features = X.shape[1]
        
        if actual_features != expected_features:
            raise ValueError(
                f"SVD变换维度不匹配: 训练时特征数={expected_features}, "
                f"当前输入特征数={actual_features}。"
                f"请确保测试数据使用与训练数据相同的特征列。"
            )
        
        VT_k = self.VT[:self.k_svd, :]
        X_reduced = X @ VT_k.T
        return X_reduced
    
    def qr_feature_selection(self, X_train_reduced, y_train, feature_ratio=0.95):
        """QR分解筛选关键特征（优化版：提高到95%以保留更多特征）"""
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
    
    def improved_lda_solve(self, X_train_selected, y_train, n_components=None, regularization=1e-5):
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
        
        # 自动确定判别向量数量（使用所有可用的）
        if n_components is None:
            n_components = min(n_classes - 1, n_features, 4)  # 最多4个，但不超过类别数-1和特征数
        
        print(f"类别数: {n_classes}, 特征数: {n_features}")
        print(f"将使用 {n_components} 个判别向量")
        
        # 计算总体均值
        overall_mean = np.mean(X_train_selected, axis=0)
        
        # 计算类内和类间散度矩阵（使用加权方式，给少数类更高权重）
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        
        # 计算类别权重（给少数类更高权重）
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        class_weights = {}
        for cls, count in zip(unique_classes, class_counts):
            # 权重与样本数成反比
            class_weights[cls] = total_samples / (len(unique_classes) * count)
        
        for cls in self.classes:
            X_cls = X_train_selected[y_train == cls]
            n_cls = len(X_cls)
            weight = class_weights.get(cls, 1.0)
            
            if n_cls > 0:
                class_mean = np.mean(X_cls, axis=0)
                
                # 类内散度（加权）
                X_cls_centered = X_cls - class_mean
                Sw += weight * (X_cls_centered.T @ X_cls_centered)
                
                # 类间散度（加权）
                mean_diff = (class_mean - overall_mean).reshape(-1, 1)
                Sb += weight * n_cls * (mean_diff @ mean_diff.T)
        
        # 正则化（减小正则化强度以保留更多信息）
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
            
            # 只保留正特征值对应的特征向量
            positive_eigenvalues = eigenvalues[:n_components] > 1e-10
            n_valid = np.sum(positive_eigenvalues)
            if n_valid < n_components:
                print(f"警告: 只有 {n_valid} 个正特征值，将使用 {n_valid} 个判别向量")
                n_components = n_valid
            
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
    
    def train_lda_model(self, X_train_selected, y_train, original_y_train=None):
        """
        训练LDA模型（计算各类别统计量）
        
        Parameters:
        -----------
        original_y_train : array-like, optional
            原始训练标签（未过采样前），用于计算真实的先验概率
        """
        print("\n" + "=" * 60)
        print("训练LDA模型（优化版）")
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
        
        # 使用原始标签计算先验概率（如果提供），否则使用过采样后的
        if original_y_train is not None:
            if isinstance(original_y_train, pd.Series):
                original_y_train = original_y_train.values
            total_samples_original = len(original_y_train)
            print("使用原始训练标签计算先验概率（更准确）")
        else:
            total_samples_original = len(y_train)
            print("使用过采样后的标签计算先验概率")
        
        for cls in self.classes:
            cls_mask = y_train == cls
            X_cls_proj = X_projected[cls_mask]
            
            # 计算均值
            self.class_means[cls] = np.mean(X_cls_proj, axis=0)
            
            # 计算协方差矩阵（使用更稳定的方法）
            if len(X_cls_proj) > 1:
                # 使用样本协方差
                cov_matrix = np.cov(X_cls_proj.T)
                # 添加正则化以提高数值稳定性
                self.class_covs[cls] = cov_matrix + 1e-5 * np.eye(self.W.shape[1])
            else:
                # 如果样本太少，使用单位矩阵
                self.class_covs[cls] = np.eye(self.W.shape[1])
            
            # 使用原始标签计算先验概率
            if original_y_train is not None:
                self.class_priors[cls] = np.sum(original_y_train == cls) / total_samples_original
            else:
                self.class_priors[cls] = np.sum(cls_mask) / total_samples_original
            
            print(f"类别 {cls}: 先验概率={self.class_priors[cls]:.4f}, "
                  f"训练样本数={np.sum(cls_mask)}, "
                  f"原始样本数={np.sum(original_y_train == cls) if original_y_train is not None else np.sum(cls_mask)}")
        
        # 优化分类阈值
        self.optimize_thresholds(X_train_selected, y_train)
    
    def optimize_thresholds(self, X_train, y_train):
        """
        优化分类阈值（优化版：使用网格搜索找到最优阈值）
        """
        print("\n优化分类阈值（网格搜索）...")
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # 计算训练集上的后验概率
        X_proj = X_train @ self.W
        posteriors = self.compute_posteriors(X_proj)
        
        # 计算各类别的样本比例
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        class_ratios = {cls: count / total_samples for cls, count in zip(unique_classes, class_counts)}
        
        # 网格搜索最优阈值
        self.thresholds = {}
        best_thresholds = {}
        best_f1_scores = {}
        
        for cls in self.classes:
            cls_ratio = class_ratios[cls]
            
            # 根据类别比例确定搜索范围
            if cls_ratio < 0.05:  # 极少数类
                threshold_range = np.arange(0.15, 0.45, 0.05)
            elif cls_ratio < 0.1:  # 少数类
                threshold_range = np.arange(0.25, 0.50, 0.05)
            elif cls_ratio < 0.3:  # 中等类
                threshold_range = np.arange(0.35, 0.60, 0.05)
            else:  # 多数类
                threshold_range = np.arange(0.40, 0.65, 0.05)
            
            best_f1 = 0
            best_threshold = 0.5
            
            # 网格搜索
            for threshold in threshold_range:
                # 应用阈值进行预测
                predictions = np.zeros(len(posteriors), dtype=int)
                for i in range(len(posteriors)):
                    cls_idx = np.where(self.classes == cls)[0][0]
                    if posteriors[i, cls_idx] >= threshold:
                        # 检查是否是最可能的类别
                        if cls_idx == np.argmax(posteriors[i]):
                            predictions[i] = cls
                        else:
                            # 选择概率最高的
                            predictions[i] = self.classes[np.argmax(posteriors[i])]
                    else:
                        predictions[i] = self.classes[np.argmax(posteriors[i])]
                
                # 计算F1分数（针对当前类别）
                # 将多分类问题转换为二分类（当前类别 vs 其他类别）
                y_binary = (y_train == cls).astype(int)
                pred_binary = (predictions == cls).astype(int)
                
                # 计算二分类F1分数
                tp = np.sum((y_binary == 1) & (pred_binary == 1))
                fp = np.sum((y_binary == 0) & (pred_binary == 1))
                fn = np.sum((y_binary == 1) & (pred_binary == 0))
                
                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)
                
                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)
                
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            best_thresholds[cls] = best_threshold
            best_f1_scores[cls] = best_f1
        
        self.thresholds = best_thresholds
        
        print(f"优化后的阈值:")
        for cls in self.classes:
            print(f"  类别 {cls}: {self.thresholds[cls]:.3f} (F1={best_f1_scores[cls]:.4f})")
    
    def compute_posteriors(self, X_projected):
        """
        计算后验概率（优化版：使用更稳定的多元高斯模型）
        """
        n_samples = X_projected.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            mean = self.class_means[cls]
            cov = self.class_covs[cls]
            prior = self.class_priors[cls]
            
            # 多元高斯概率密度（使用更稳定的计算方法）
            diff = X_projected - mean
            
            # 使用伪逆而不是直接求逆，提高数值稳定性
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # 如果求逆失败，使用伪逆
                cov_inv = np.linalg.pinv(cov)
            
            # 计算马氏距离（更稳定的方式）
            mahalanobis = np.sum((diff @ cov_inv) * diff, axis=1)
            
            # 添加数值裁剪，避免溢出
            mahalanobis = np.clip(mahalanobis, 0, 1e6)
            
            # 计算对数似然
            log_likelihood = -0.5 * mahalanobis
            
            # 添加先验概率（使用原始先验，而不是SMOTE后的先验）
            # 这里使用训练时的原始先验概率
            posteriors[:, i] = np.log(max(prior, 1e-10)) + log_likelihood
        
        # Softmax归一化（数值稳定版本）
        # 减去最大值以避免溢出
        max_log_posterior = np.max(posteriors, axis=1, keepdims=True)
        posteriors = posteriors - max_log_posterior
        
        # 计算exp并归一化
        posteriors = np.exp(np.clip(posteriors, -500, 500))  # 裁剪避免溢出
        posteriors = posteriors / (np.sum(posteriors, axis=1, keepdims=True) + 1e-10)
        
        return posteriors
    
    def predict(self, X_selected, use_threshold=True, use_ensemble=False, n_ensemble=5):
        """
        预测（带阈值调整和集成预测）
        
        Parameters:
        -----------
        use_threshold : bool
            是否使用优化后的阈值
        use_ensemble : bool
            是否使用集成预测（通过添加小噪声创建多个预测变体）
        n_ensemble : int
            集成预测的变体数量
        """
        if self.W is None or self.class_means is None:
            raise ValueError("模型尚未训练")
        
        if use_ensemble and n_ensemble > 1:
            # 集成预测：创建多个预测变体并投票
            all_predictions = []
            
            for _ in range(n_ensemble):
                # 添加小噪声（仅用于集成预测）
                if _ > 0:  # 第一次使用原始数据
                    noise_scale = 1e-6  # 非常小的噪声，不影响主要特征
                    X_noisy = X_selected + np.random.normal(0, noise_scale, X_selected.shape)
                else:
                    X_noisy = X_selected
                
                # 投影
                X_proj = X_noisy @ self.W
                
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
                
                all_predictions.append(predictions)
            
            # 投票：选择每个样本的众数
            final_predictions = np.zeros(len(X_selected), dtype=int)
            for i in range(len(X_selected)):
                votes = [pred[i] for pred in all_predictions]
                final_predictions[i] = Counter(votes).most_common(1)[0][0]
            
            return final_predictions
        else:
            # 标准预测（不使用集成）
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
    
    def temporal_smoothing(self, predictions, window_size=7):
        """
        时序平滑（优化版：增大窗口以提高稳定性）
        """
        smoothed = np.copy(predictions)
        
        for i in range(len(predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start:end]
            
            # 使用众数（加权：中心位置权重更高）
            if len(window) > 0:
                # 计算加权投票
                weights = np.exp(-np.abs(np.arange(len(window)) - len(window)//2) / (len(window)/4))
                counter = Counter()
                for j, pred in enumerate(window):
                    counter[pred] = counter.get(pred, 0) + weights[j]
                smoothed[i] = counter.most_common(1)[0][0]
            else:
                smoothed[i] = predictions[i]
        
        return smoothed
    
    def evaluate_model(self, X_test_selected, y_test, use_smoothing=True, use_ensemble=False):
        """
        评估模型（优化版）
        
        Parameters:
        -----------
        use_smoothing : bool
            是否使用时序平滑
        use_ensemble : bool
            是否使用集成预测
        """
        print("\n" + "=" * 60)
        print("模型评估（优化版）")
        print("=" * 60)
        
        # 使用集成预测（如果启用）
        y_pred = self.predict(X_test_selected, use_threshold=True, use_ensemble=use_ensemble, n_ensemble=5)
        
        if use_smoothing:
            y_pred = self.temporal_smoothing(y_pred, window_size=7)
        
        accuracy = np.mean(y_pred == y_test)
        print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n各类别准确率:")
        for cls in self.classes:
            cls_mask = y_test == cls
            if np.any(cls_mask):
                cls_accuracy = np.mean(y_pred[cls_mask] == y_test[cls_mask])
                cls_precision = np.sum((y_pred == cls) & (y_test == cls)) / max(np.sum(y_pred == cls), 1)
                cls_recall = np.sum((y_pred == cls) & (y_test == cls)) / max(np.sum(y_test == cls), 1)
                print(f"  类别 {cls}: 准确率={cls_accuracy:.4f}, 精确率={cls_precision:.4f}, 召回率={cls_recall:.4f}")
        
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
    # 优先使用已清洗的训练数据（如果存在），否则从原始数据加载
    cleaned_train_path = 'train/清洗训练数据.csv'
    if os.path.exists(cleaned_train_path):
        print("=" * 60)
        print("使用已清洗的训练数据（包含采样后的数据）")
        print("=" * 60)
        cleaner = DataCleaner(
            train_path='train/训练数据.csv',  # 用于加载预处理器
            test_dir='test/',
            output_dir='model/'
        )
        cleaner.load_preprocessor()  # 加载预处理器（标准化器等）
        
        # 从清洗训练数据加载（这是已经采样后的数据）
        df_train_cleaned = pd.read_csv(cleaned_train_path)
        print(f"加载清洗训练数据: {df_train_cleaned.shape}")
        print(f"训练数据类别分布:\n{df_train_cleaned['labelArea'].value_counts().sort_index()}")
        
        # 准备特征和标签（使用已清洗的数据）
        X_train, y_train = cleaner.prepare_features(df_train_cleaned, is_train=True)
        
        # 保存训练数据的特征列（确保后续测试数据使用相同的特征）
        train_feature_columns = cleaner.feature_columns.copy() if cleaner.feature_columns is not None else None
        if train_feature_columns is None and hasattr(X_train, 'columns'):
            train_feature_columns = list(X_train.columns)
        
        print(f"训练数据特征列数: {len(train_feature_columns) if train_feature_columns else X_train.shape[1]}")
        
        # 标准化（使用已加载的预处理器）
        X_train = cleaner.transform_features(X_train, is_train=False)
        
        # 对于测试集，需要确保使用与训练数据相同的特征列
        # 重要：测试集必须使用与训练数据相同的特征列，否则SVD会失败
        print("\n从原始数据生成测试集（用于模型评估）...")
        print("注意：确保测试集使用与训练数据相同的特征列")
        
        # 从原始数据加载并划分测试集
        df_full = cleaner.load_train_data()
        
        # 使用split_data获取测试集（这会使用cleaner.feature_columns，应该与训练数据一致）
        # 但为了确保一致性，我们显式设置feature_columns
        if train_feature_columns is not None:
            cleaner.feature_columns = train_feature_columns
        
        # 划分数据
        _, X_test_df, _, y_test = cleaner.split_data(df_full, test_size=0.3)
        
        # 确保测试数据使用相同的特征列
        if train_feature_columns is not None and hasattr(X_test_df, 'columns'):
            # 如果X_test_df是DataFrame，确保列顺序和数量一致
            available_cols = [col for col in train_feature_columns if col in X_test_df.columns]
            X_test_df = X_test_df[available_cols]
            
            # 如果缺少某些列，用0填充
            missing_cols = set(train_feature_columns) - set(available_cols)
            if missing_cols:
                print(f"  警告: 测试数据缺少 {len(missing_cols)} 个特征列，将用0填充")
                for col in missing_cols:
                    X_test_df[col] = 0
            
            # 确保列顺序一致
            X_test_df = X_test_df[train_feature_columns]
        
        # 转换为numpy数组（如果需要）
        if isinstance(X_test_df, pd.DataFrame):
            X_test = X_test_df.values
        else:
            X_test = X_test_df
        
        # 标准化（使用已加载的预处理器）
        X_test = cleaner.transform_features(X_test, is_train=False)
        
        # 检查特征数量是否一致
        print(f"\n训练集特征数: {X_train.shape[1]}")
        print(f"测试集特征数: {X_test.shape[1]}")
        
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"特征数量不匹配: 训练集={X_train.shape[1]}, 测试集={X_test.shape[1]}。"
                f"请确保训练数据和测试数据使用相同的特征选择。"
            )
        
        print(f"\n训练集形状: {X_train.shape} (来自清洗训练数据)")
        print(f"测试集形状: {X_test.shape} (来自原始数据划分)")
    else:
        print("=" * 60)
        print("使用原始训练数据（未找到清洗训练数据）")
        print("=" * 60)
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
    
    # 保存原始训练标签（用于计算真实的先验概率）
    y_train_original = y_train.copy()
    
    # 3. SMOTE过采样（完全平衡）
    X_train_balanced, y_train_balanced = trainer.smote_oversample(
        X_train, y_train, 
        k_neighbors=5, 
        target_ratio=1.0  # 完全平衡到最大类
    )
    
    # 4. SVD降维（提高到99.5%以保留更多信息）
    X_train_reduced = trainer.svd_decomposition(X_train_balanced, cumulative_ratio=0.995)
    X_test_reduced = trainer.apply_svd_transform(X_test)
    
    # 5. QR特征筛选（提高到95%以保留更多特征）
    X_train_selected = trainer.qr_feature_selection(X_train_reduced, y_train_balanced, feature_ratio=0.95)
    X_test_selected = trainer.apply_qr_selection(X_test_reduced)
    
    # 6. 改进的LDA求解（使用更多判别向量）
    trainer.improved_lda_solve(X_train_selected, y_train_balanced, n_components=None)  # 自动选择最大数量
    
    # 7. 训练模型（使用原始标签计算先验概率）
    trainer.train_lda_model(X_train_selected, y_train_balanced, original_y_train=y_train_original)
    
    # 8. 评估（带时序平滑和集成预测）
    results = trainer.evaluate_model(X_test_selected, y_test, use_smoothing=True, use_ensemble=False)
    
    # 9. 保存模型
    trainer.save_model('improved_lda_model.pkl')
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()