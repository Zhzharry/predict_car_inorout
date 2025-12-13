"""
GNN模型训练模块
功能：图构建、SVD/QR/LU预处理、轻量级GCN/GAT模型训练
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import time
from datetime import datetime
from scipy.linalg import svd, qr, lu_factor, lu_solve
# from scipy.stats import pearsonr  # 不再使用，避免警告
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv
from a_数据清洗代码 import DataCleaner


def convert_to_python_type(obj):
    """递归将 numpy 类型转换为 Python 原生类型（用于 JSON 序列化）"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_python_type(k): convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_type(item) for item in obj]
    else:
        return obj


class GraphBuilder:
    """图构建器：将传感器数据转换为图结构"""
    
    def __init__(self, window_size=25, svd_ratio=0.95, qr_threshold=0.1):
        """
        初始化图构建器
        
        Parameters:
        -----------
        window_size : int
            滑动窗口大小（时间步数）
        svd_ratio : float
            SVD保留的方差比例
        qr_threshold : float
            QR分解边剪枝阈值
        """
        self.window_size = window_size
        self.svd_ratio = svd_ratio
        self.qr_threshold = qr_threshold
        
    def svd_denoise(self, X):
        """
        SVD降噪：对单维度传感器数据降噪（优化版）
        
        Parameters:
        -----------
        X : ndarray
            输入数据 (n_samples, n_features) 或 (n_samples,)
        
        Returns:
        --------
        X_denoised : ndarray
            降噪后的数据
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 对于小窗口，简化处理：只对特征维度做降噪
        window_size, n_features = X.shape
        
        # 如果窗口很小或特征很少，跳过SVD或简化处理
        if window_size < 5 or n_features < 3:
            # 简单平滑：使用移动平均
            if window_size > 1:
                X_denoised = (X[:-1, :] + X[1:, :]) / 2
                X_denoised = np.vstack([X_denoised, X[-1:, :]])
            else:
                X_denoised = X.copy()
            return X_denoised
        
        # 对于较大的窗口，进行SVD降噪
        try:
            # SVD分解
            U, S, VT = svd(X, full_matrices=False)
            
            # 计算累积贡献率
            if len(S) > 0 and np.sum(S) > 0:
                cumsum_ratio = np.cumsum(S) / np.sum(S)
                k = np.argmax(cumsum_ratio >= self.svd_ratio) + 1
                k = min(k, len(S), min(window_size, n_features))
            else:
                k = min(window_size, n_features)
            
            # 重构降噪数据
            if k > 0:
                X_denoised = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
            else:
                X_denoised = X.copy()
        except:
            # 如果SVD失败，返回原始数据
            X_denoised = X.copy()
        
        return X_denoised
    
    def compute_temporal_edges(self, X_window):
        """
        计算时序依赖边权重（简化版，避免复杂计算）
        
        Parameters:
        -----------
        X_window : ndarray
            窗口数据 (window_size, n_features)
        
        Returns:
        --------
        edge_index : ndarray
            边索引 (2, n_edges)
        edge_attr : ndarray
            边权重 (n_edges,)
        """
        window_size, n_features = X_window.shape
        edge_list = []
        edge_weights = []
        
        # 对每个特征维度计算时序相关性（简化：使用差值作为权重）
        for feat_idx in range(n_features):
            feat_series = X_window[:, feat_idx]
            
            # 计算相邻时间步的权重（使用归一化差值）
            for t in range(window_size - 1):
                # 使用简单的相似度计算，避免相关系数的警告
                val_t = feat_series[t]
                val_t1 = feat_series[t + 1]
                
                # 计算权重：1 - 归一化差值
                if abs(val_t) > 1e-6:
                    weight = 1.0 - min(abs(val_t - val_t1) / (abs(val_t) + 1e-6), 1.0)
                else:
                    # 如果值接近0，使用绝对值相似度
                    weight = 1.0 - min(abs(val_t - val_t1), 1.0)
                
                weight = max(0.0, weight)  # 确保权重非负
                
                # 添加时序边：t时刻的feat_idx节点 -> t+1时刻的feat_idx节点
                node_from = t * n_features + feat_idx
                node_to = (t + 1) * n_features + feat_idx
                edge_list.append([node_from, node_to])
                edge_weights.append(weight)
        
        if len(edge_list) == 0:
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_attr = np.array(edge_weights, dtype=np.float32)
        
        return edge_index, edge_attr
    
    def compute_feature_edges(self, X_window):
        """
        计算特征关联边（简化版，避免复杂计算）
        
        Parameters:
        -----------
        X_window : ndarray
            窗口数据 (window_size, n_features)
        
        Returns:
        --------
        edge_index : ndarray
            边索引 (2, n_edges)
        edge_attr : ndarray
            边权重 (n_edges,)
        """
        window_size, n_features = X_window.shape
        edge_list = []
        edge_weights = []
        
        # 对每个时间步计算特征关联（简化：只计算部分特征对，避免O(n^2)复杂度）
        max_feature_pairs = min(50, n_features * (n_features - 1) // 2)  # 限制特征对数量
        
        for t in range(window_size):
            feat_vector = X_window[t, :]
            
            if n_features > 1:
                # 简化：只计算部分特征对，使用随机采样或选择重要特征
                # 使用QR分解的R矩阵对角线元素作为特征重要性
                try:
                    Q, R = qr(feat_vector.reshape(-1, 1), mode='economic')
                    # 使用R的对角线元素（如果有）或特征值的绝对值作为重要性
                    feature_importance = np.abs(feat_vector)
                except:
                    feature_importance = np.abs(feat_vector)
                
                # 选择重要性较高的特征进行关联计算
                top_k = min(20, n_features)  # 只计算前k个重要特征之间的关联
                top_indices = np.argsort(feature_importance)[-top_k:]
                
                # 计算选中的特征对之间的关联
                for idx_i, i in enumerate(top_indices):
                    for j in top_indices[idx_i + 1:]:
                        # 使用简单的相似度计算
                        val_i = feat_vector[i]
                        val_j = feat_vector[j]
                        
                        # 计算权重：归一化差值
                        max_val = max(abs(val_i), abs(val_j), 1e-6)
                        weight = 1.0 - min(abs(val_i - val_j) / max_val, 1.0)
                        weight = max(0.0, weight)
                        
                        # QR剪枝：只保留权重>阈值的边
                        if weight > self.qr_threshold:
                            node_i = t * n_features + i
                            node_j = t * n_features + j
                            edge_list.append([node_i, node_j])
                            edge_list.append([node_j, node_i])  # 无向边
                            edge_weights.append(weight)
                            edge_weights.append(weight)
        
        if len(edge_list) == 0:
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_attr = np.array(edge_weights, dtype=np.float32)
        
        return edge_index, edge_attr
    
    def build_graph(self, X_window, y_label=None):
        """
        构建图结构
        
        Parameters:
        -----------
        X_window : ndarray
            窗口数据 (window_size, n_features)
        y_label : int or None
            图标签（整图对应一个分类标签）
        
        Returns:
        --------
        graph_data : torch_geometric.data.Data
            图数据对象
        """
        window_size, n_features = X_window.shape
        
        # SVD降噪
        X_denoised = self.svd_denoise(X_window)
        
        # 构建节点特征：每个节点对应"时间步+传感器维度"
        node_features = []
        for t in range(window_size):
            for feat_idx in range(n_features):
                # 节点特征：该时间步该维度的降噪值（单维度）
                node_feat = X_denoised[t, feat_idx] if X_denoised.ndim == 2 else X_denoised[t]
                node_features.append([float(node_feat)])
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # 计算边
        temporal_edges, temporal_weights = self.compute_temporal_edges(X_denoised)
        feature_edges, feature_weights = self.compute_feature_edges(X_denoised)
        
        # 合并边
        if temporal_edges.shape[1] > 0 and feature_edges.shape[1] > 0:
            edge_index = np.concatenate([temporal_edges, feature_edges], axis=1)
            edge_attr = np.concatenate([temporal_weights, feature_weights])
        elif temporal_edges.shape[1] > 0:
            edge_index = temporal_edges
            edge_attr = temporal_weights
        elif feature_edges.shape[1] > 0:
            edge_index = feature_edges
            edge_attr = feature_weights
        else:
            # 如果没有边，创建自环
            n_nodes = len(node_features)
            edge_index = np.array([[i, i] for i in range(n_nodes)], dtype=np.int64).T
            edge_attr = np.ones(n_nodes, dtype=np.float32)
        
        # 检查并清理 NaN/Inf
        if np.isnan(node_features).any() or np.isinf(node_features).any():
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.isnan(edge_attr).any() or np.isinf(edge_attr).any():
            edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 转换为torch tensor
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr).float()
        x = torch.from_numpy(node_features).float()
        
        # 再次检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 创建图数据
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # 添加图标签
        if y_label is not None:
            graph_data.y = torch.tensor([y_label], dtype=torch.long)
        
        return graph_data


class LightweightGCN(nn.Module):
    """轻量级GCN模型（2层，无隐藏层堆叠）"""
    
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=3):
        """
        初始化轻量级GCN
        
        Parameters:
        -----------
        input_dim : int
            输入特征维度（节点特征维度）
        hidden_dim : int
            隐藏层维度
        output_dim : int
            输出维度（类别数）
        """
        super(LightweightGCN, self).__init__()
        
        # 2层GCN：输入层 + 输出层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        # 初始化权重（Xavier初始化，提高数值稳定性）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, GCNConv):
                # 使用Xavier初始化
                if hasattr(m, 'lin'):
                    torch.nn.init.xavier_uniform_(m.lin.weight)
                    if m.lin.bias is not None:
                        torch.nn.init.zeros_(m.lin.bias)
        
    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        
        Parameters:
        -----------
        x : Tensor
            节点特征 (n_nodes, input_dim)
        edge_index : Tensor
            边索引 (2, n_edges)
        batch : Tensor or None
            批次索引（用于图级别分类）
        
        Returns:
        --------
        out : Tensor
            图级别输出 (n_graphs, output_dim)
        """
        # 检查输入是否有 NaN 或 Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告：输入包含 NaN 或 Inf，进行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 第一层GCN + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 检查中间结果
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告：第一层GCN输出包含 NaN 或 Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        
        # 检查输出
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告：第二层GCN输出包含 NaN 或 Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 图级别聚合：均值池化
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            # 单图情况：直接均值
            x = x.mean(dim=0, keepdim=True)
        
        # 最终检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告：最终输出包含 NaN 或 Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x


class GNNTrainer:
    """GNN模型训练类"""
    
    def __init__(self, model_dir='model/', random_state=42, model_type='GCN'):
        """
        初始化GNN训练器
        
        Parameters:
        -----------
        model_dir : str
            模型保存目录
        random_state : int
            随机种子
        model_type : str
            模型类型：'GCN' 或 'GAT'
        """
        self.model_dir = model_dir
        self.random_state = random_state
        self.model_type = model_type
        os.makedirs(model_dir, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        # 图构建器
        self.graph_builder = GraphBuilder(window_size=25, svd_ratio=0.95, qr_threshold=0.1)
        
        # 模型参数
        self.model = None
        self.input_dim = 1  # 节点特征维度（单维度）
        self.hidden_dim = 32
        self.output_dim = 3  # 三分类
        self.learning_rate = 5e-4  # 适中的学习率（避免梯度爆炸）
        self.batch_size = 8
        self.epochs = 50
        
        # 数据信息
        self.classes = None
        self.class_weights = None
        
    def build_graphs_from_data(self, X, y):
        """
        从数据构建图列表
        
        Parameters:
        -----------
        X : ndarray
            特征矩阵 (n_samples, n_features)
        y : ndarray
            标签数组 (n_samples,)
        
        Returns:
        --------
        graphs : list
            图数据列表
        """
        print(f"\n正在构建图结构（窗口大小: {self.graph_builder.window_size}）...")
        
        # 确保y是numpy数组
        if hasattr(y, 'values'):
            y = y.values
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        window_size = self.graph_builder.window_size
        n_graphs = n_samples - window_size + 1
        
        graphs = []
        print(f"预计构建 {n_graphs} 个图，请稍候...")
        
        # 创建类别到索引的映射（用于将标签映射到模型输出索引）
        if self.classes is not None:
            class_to_idx = {int(cls): idx for idx, cls in enumerate(sorted(self.classes))}
        else:
            # 如果还没有类别信息，从y中获取
            unique_classes = np.unique(y)
            class_to_idx = {int(cls): idx for idx, cls in enumerate(sorted(unique_classes))}
        
        for i in range(n_graphs):
            # 提取窗口数据
            X_window = X[i:i+window_size, :]
            y_label_raw = y[i + window_size // 2]  # 使用窗口中心标签
            
            # 将原始标签映射到模型输出索引
            y_label_idx = class_to_idx[int(y_label_raw)]
            
            # 构建图（传入映射后的索引）
            graph = self.graph_builder.build_graph(X_window, y_label_idx)
            graphs.append(graph)
            
            # 显示进度
            if (i + 1) % max(1, n_graphs // 20) == 0 or i == n_graphs - 1:
                progress = (i + 1) / n_graphs * 100
                print(f"  进度: {progress:.1f}% ({i+1}/{n_graphs})")
        
        print(f"构建完成：共 {len(graphs)} 个图")
        return graphs
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        训练GNN模型
        
        Parameters:
        -----------
        X_train : ndarray
            训练特征
        y_train : ndarray
            训练标签
        X_val : ndarray
            验证特征
        y_val : ndarray
            验证标签
        
        Returns:
        --------
        history : dict
            训练历史
        """
        print("\n" + "=" * 60)
        print("GNN模型训练")
        print("=" * 60)
        
        # 先获取类别信息（在构建图之前）
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        y_train = np.asarray(y_train)
        self.classes = np.unique(y_train)
        self.output_dim = len(self.classes)
        
        # 创建类别到索引的映射（用于将原始标签映射到模型输出索引）
        sorted_classes = sorted(self.classes)
        self.class_to_idx = {int(cls): idx for idx, cls in enumerate(sorted_classes)}
        print(f"类别映射: {self.class_to_idx}")
        
        # 构建图（标签会被映射到索引）
        train_graphs = self.build_graphs_from_data(X_train, y_train)
        val_graphs = self.build_graphs_from_data(X_val, y_val)
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=self.classes, y=y_train)
        self.class_weights = {int(cls): float(weight) for cls, weight in zip(self.classes, class_weights)}
        print(f"类别权重: {self.class_weights}")
        
        # 将类别权重转换为tensor（按类别顺序）
        sorted_classes = sorted(self.classes)
        weight_tensor = torch.FloatTensor([self.class_weights[int(cls)] for cls in sorted_classes])
        print(f"权重tensor: {weight_tensor}")
        
        # 创建模型
        self.model = LightweightGCN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # 优化器和损失函数（添加梯度裁剪和学习率调整，使用类别权重）
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)  # 使用类别权重！
        
        # 梯度裁剪阈值
        max_grad_norm = 1.0
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        print(f"\n开始训练（epochs={self.epochs}, batch_size={self.batch_size}）...")
        print(f"训练图数量: {len(train_graphs)}, 验证图数量: {len(val_graphs)}")
        
        try:
            for epoch in range(self.epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                n_batches = (len(train_graphs) + self.batch_size - 1) // self.batch_size
                print(f"\nEpoch {epoch+1}/{self.epochs}: 开始训练（{n_batches} 个批次）...")
                
                # 批量训练
                batch_count = 0
                for i in range(0, len(train_graphs), self.batch_size):
                    try:
                        batch_graphs = train_graphs[i:i+self.batch_size]
                        batch = Batch.from_data_list(batch_graphs)
                        
                        # 前向传播
                        out = self.model(batch.x, batch.edge_index, batch.batch)
                        labels = batch.y
                        
                        # 计算损失
                        loss = criterion(out, labels)
                        
                        # 检查损失是否为 NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"  警告：批次 {batch_count+1} 损失为 NaN/Inf，跳过该批次")
                            print(f"    输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
                            print(f"    输出包含NaN: {torch.isnan(out).any().item()}")
                            print(f"    标签: {labels}")
                            continue
                        
                        # 检查输出是否正常
                        if torch.isnan(out).any() or torch.isinf(out).any():
                            print(f"  警告：批次 {batch_count+1} 输出包含 NaN/Inf，跳过该批次")
                            continue
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪（防止梯度爆炸）
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        pred = out.argmax(dim=1)
                        train_correct += (pred == labels).sum().item()
                        train_total += len(labels)
                        
                        batch_count += 1
                        # 每10个batch打印一次进度
                        if batch_count % 10 == 0:
                            print(f"  批次 {batch_count}/{n_batches}, 当前损失: {loss.item():.4f}")
                    except Exception as e:
                        print(f"  错误：处理批次 {batch_count+1} 时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
            
                avg_train_loss = train_loss / max(batch_count, 1)
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                
                print(f"  训练完成: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
                
                # 验证阶段
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                val_n_batches = (len(val_graphs) + self.batch_size - 1) // self.batch_size
                print(f"  开始验证（{val_n_batches} 个批次）...")
                
                with torch.no_grad():
                    val_batch_count = 0
                    for i in range(0, len(val_graphs), self.batch_size):
                        try:
                            batch_graphs = val_graphs[i:i+self.batch_size]
                            batch = Batch.from_data_list(batch_graphs)
                            
                            out = self.model(batch.x, batch.edge_index, batch.batch)
                            labels = batch.y
                            
                            loss = criterion(out, labels)
                            val_loss += loss.item()
                            
                            pred = out.argmax(dim=1)
                            val_correct += (pred == labels).sum().item()
                            val_total += len(labels)
                            
                            val_batch_count += 1
                        except Exception as e:
                            print(f"  错误：验证批次 {val_batch_count+1} 时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            raise
            
                avg_val_loss = val_loss / max(val_batch_count, 1)
                val_acc = val_correct / val_total if val_total > 0 else 0.0
                
                # 记录历史
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                # 打印进度（每个epoch都打印）
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    print(f"  ✓ 验证准确率提升，保存最佳模型")
                    # 保存最佳模型
                    self.save_model('gnn_model_best.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                    else:
                        print(f"  验证准确率未提升 ({patience_counter}/{patience})")
        
        except Exception as e:
            print(f"\n训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型
        
        Parameters:
        -----------
        X_test : ndarray
            测试特征
        y_test : ndarray
            测试标签
        
        Returns:
        --------
        results : dict
            评估结果
        """
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        # 构建图
        test_graphs = self.build_graphs_from_data(X_test, y_test)
        
        # 确保y是numpy数组
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        y_test = np.asarray(y_test)
        
        # 预测
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(test_graphs), self.batch_size):
                batch_graphs = test_graphs[i:i+self.batch_size]
                batch = Batch.from_data_list(batch_graphs)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()
                
                all_preds.extend(pred)
                all_labels.extend(labels)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        print(f"\n准确率: {accuracy:.4f}")
        print(f"\n混淆矩阵:\n{cm}")
        print(f"\n分类报告:\n{classification_report(all_labels, all_preds)}")
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'true_labels': all_labels
        }
        
        return results
    
    def save_model(self, filename='gnn_model.pth'):
        """
        保存模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        filepath = os.path.join(self.model_dir, filename)
        
        # 保存PyTorch模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'model_type': self.model_type,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'class_weights': self.class_weights,
            'graph_builder_config': {
                'window_size': self.graph_builder.window_size,
                'svd_ratio': self.graph_builder.svd_ratio,
                'qr_threshold': self.graph_builder.qr_threshold
            }
        }, filepath)
        
        print(f"\n模型已保存到: {filepath}")
        model_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"模型文件大小: {model_size:.2f} MB")
        
        # 保存配置
        config = {
            'model_type': 'GNN',
            'gnn_type': self.model_type,
            'input_dim': int(self.input_dim),
            'hidden_dim': int(self.hidden_dim),
            'output_dim': int(self.output_dim),
            'learning_rate': float(self.learning_rate),
            'batch_size': int(self.batch_size),
            'window_size': int(self.graph_builder.window_size),
            'svd_ratio': float(self.graph_builder.svd_ratio),
            'qr_threshold': float(self.graph_builder.qr_threshold),
            'classes': self.classes.tolist() if self.classes is not None else None,
            'class_weights': self.class_weights,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'training_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        config = convert_to_python_type(config)
        config_path = os.path.join(self.model_dir, 'gnn_model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"模型配置已保存到: {config_path}")
    
    def load_model(self, filename='gnn_model.pth'):
        """
        加载模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.input_dim = checkpoint.get('input_dim', 1)
        self.hidden_dim = checkpoint.get('hidden_dim', 32)
        self.output_dim = checkpoint.get('output_dim', 3)
        self.model_type = checkpoint.get('model_type', 'GCN')
        
        # 重建模型
        self.model = LightweightGCN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载其他信息
        self.classes = np.array(checkpoint.get('classes')) if checkpoint.get('classes') else None
        self.class_weights = checkpoint.get('class_weights')
        
        # 重建类别到索引的映射
        if self.classes is not None:
            sorted_classes = sorted(self.classes)
            self.class_to_idx = {int(cls): idx for idx, cls in enumerate(sorted_classes)}
        
        # 加载图构建器配置
        graph_config = checkpoint.get('graph_builder_config', {})
        self.graph_builder.window_size = graph_config.get('window_size', 25)
        self.graph_builder.svd_ratio = graph_config.get('svd_ratio', 0.95)
        self.graph_builder.qr_threshold = graph_config.get('qr_threshold', 0.1)
        
        print(f"模型已加载: {filepath}")
        if self.classes is not None:
            print(f"类别数: {len(self.classes)}")
            print(f"类别映射: {self.class_to_idx}")


def main():
    """主函数：完整的模型训练流程"""
    print("=" * 60)
    print("GNN模型训练流程")
    print("=" * 60)
    
    # 1. 数据清洗和预处理
    cleaner = DataCleaner(
        train_path='train/清洗训练数据.csv',
        test_dir='test/',
        output_dir='model/'
    )
    
    X_train, X_test, y_train, y_test = cleaner.process_train_data(
        remove_outliers_flag=True,
        fill_missing_flag=True,
        standardize=True,
        test_size=0.3
    )
    
    # 2. 划分验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )
    
    # 3. 创建训练器
    trainer = GNNTrainer(model_dir='model/', random_state=42, model_type='GCN')
    
    # 4. 训练模型
    history = trainer.train_model(X_train, y_train, X_val, y_val)
    
    # 5. 评估模型
    results = trainer.evaluate_model(X_test, y_test)
    
    # 6. 保存模型
    trainer.save_model('gnn_model.pth')
    
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)
    print("\n提示：")
    print("  - 模型已保存到: model/gnn_model.pth")
    print("  - 最佳模型已保存到: model/gnn_model_best.pth")
    print("  - 请运行 c_模型调用以及结果分析代码.py 进行预测")
    print("=" * 60)


if __name__ == '__main__':
    main()

