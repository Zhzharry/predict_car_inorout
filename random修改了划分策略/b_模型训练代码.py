"""
随机森林模型训练模块（改进版：处理类别不平衡）
功能：特征重要性评估、特征筛选、模型训练、超参数优化
主要改进：
1. 使用class_weight='balanced'平衡类别权重
2. 确保预测分布更接近真实分布
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import time
import json
from datetime import datetime
from a_数据清洗代码 import DataCleaner


class RandomForestTrainer:
    """随机森林模型训练类"""
    
    def __init__(self, model_dir='model/', random_state=42):
        """
        初始化训练器
        
        Parameters:
        -----------
        model_dir : str
            模型保存目录
        random_state : int
            随机种子
        """
        self.model_dir = model_dir
        self.random_state = random_state
        os.makedirs(model_dir, exist_ok=True)
        
        # 模型和特征选择器
        self.model = None
        self.best_model = None
        self.selected_features = None
        self.feature_importance = None
        
    def train_initial_model(self, X_train, y_train, n_estimators=100, 
                           max_depth=None, max_features='sqrt', 
                           min_samples_leaf=1, n_jobs=-1):
        """
        训练初始随机森林模型（用于特征重要性评估）
        
        Parameters:
        -----------
        X_train : ndarray
            训练特征矩阵
        y_train : ndarray
            训练标签
        n_estimators : int
            决策树数量
        max_depth : int or None
            树的最大深度
        max_features : str or int
            每棵树的最大特征数
        min_samples_leaf : int
            叶节点最小样本数
        n_jobs : int
            并行任务数
        
        Returns:
        --------
        model : RandomForestClassifier
            训练好的模型
        """
        print("=" * 60)
        print("训练初始随机森林模型（用于特征重要性评估）")
        print("=" * 60)
        
        # 计算自定义类别权重：给类别1和类别2更高的权重
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        # 创建类别到样本数的映射
        class_to_count = dict(zip(unique_classes, class_counts))
        
        # 计算基础权重（平衡权重）
        class_weights_balanced = {}
        for cls in unique_classes:
            class_weights_balanced[cls] = n_samples / (n_classes * class_to_count[cls])
        
        # 针对性地提高类别1和类别2的权重（相对于类别0）
        # 类别1和类别2的权重是类别0的2.5倍
        class_weights_custom = class_weights_balanced.copy()
        if 0 in class_weights_custom:
            base_weight = class_weights_custom[0]
            if 1 in class_weights_custom:
                class_weights_custom[1] = base_weight * 2.5  # 进入地库权重提高2.5倍
            if 2 in class_weights_custom:
                class_weights_custom[2] = base_weight * 2.5  # 出地库权重提高2.5倍
        
        print(f"类别权重设置:")
        for cls in sorted(unique_classes):
            print(f"  类别 {cls}: {class_weights_custom[cls]:.4f} (样本数: {class_to_count[cls]})")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=0,
            class_weight=class_weights_custom  # 使用自定义类别权重，提高类别1和类别2的权重
        )
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"模型训练完成，耗时: {training_time:.2f} 秒")
        
        # 评估初始模型
        train_score = self.model.score(X_train, y_train)
        print(f"训练集准确率: {train_score:.4f}")
        
        return self.model
    
    def evaluate_feature_importance(self, feature_names=None, top_n=None):
        """
        评估特征重要性
        
        Parameters:
        -----------
        feature_names : list or None
            特征名称列表
        top_n : int or None
            显示前N个重要特征
        
        Returns:
        --------
        importance_df : DataFrame
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train_initial_model")
        
        print("=" * 60)
        print("评估特征重要性")
        print("=" * 60)
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        self.feature_importance = importances
        
        # 创建特征重要性DataFrame
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            # 确保 feature_names 和 importances 长度一致
            if len(feature_names) != len(importances):
                print(f"警告: feature_names 长度 ({len(feature_names)}) 与 importances 长度 ({len(importances)}) 不一致")
                print(f"将使用前 {len(importances)} 个特征名称")
                # 如果 feature_names 更长，截取前面的部分
                if len(feature_names) > len(importances):
                    feature_names = feature_names[:len(importances)]
                # 如果 feature_names 更短，补充默认名称
                else:
                    feature_names = list(feature_names) + [f'feature_{i}' for i in range(len(feature_names), len(importances))]
        
        # 再次确保长度一致
        assert len(feature_names) == len(importances), f"特征名称数量 ({len(feature_names)}) 必须等于重要性数量 ({len(importances)})"
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 计算累积重要性
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        print(f"\n总特征数: {len(importance_df)}")
        print(f"\n前10个最重要特征:")
        print(importance_df.head(10).to_string(index=False))
        
        if top_n:
            print(f"\n前{top_n}个特征的重要性:")
            print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def select_features(self, X_train, y_train, feature_names=None, 
                        importance_threshold=None, top_ratio=0.7):
        """
        基于特征重要性筛选特征
        
        Parameters:
        -----------
        X_train : ndarray
            训练特征矩阵
        y_train : ndarray
            训练标签
        feature_names : list or None
            特征名称列表
        importance_threshold : float or None
            重要性阈值（保留重要性大于阈值的特征）
        top_ratio : float
            保留前top_ratio比例的特征（0-1之间）
        
        Returns:
        --------
        X_selected : ndarray
            筛选后的特征矩阵
        selected_indices : ndarray
            选中的特征索引
        """
        if self.feature_importance is None:
            raise ValueError("请先评估特征重要性")
        
        print("=" * 60)
        print("特征筛选")
        print("=" * 60)
        
        # 获取特征重要性
        importances = self.feature_importance
        
        # 创建特征重要性DataFrame并排序
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'index': range(len(importances))
        }).sort_values('importance', ascending=False)
        
        # 根据阈值或比例筛选特征
        if importance_threshold is not None:
            selected_mask = importances >= importance_threshold
            selected_indices = np.where(selected_mask)[0]
            print(f"使用重要性阈值筛选: {importance_threshold}")
        else:
            # 保留前top_ratio比例的特征
            n_selected = int(len(importances) * top_ratio)
            selected_indices = importance_df.head(n_selected)['index'].values
            print(f"保留前 {top_ratio*100:.1f}% 的特征: {n_selected}/{len(importances)}")
        
        # 筛选特征
        X_selected = X_train[:, selected_indices]
        self.selected_features = selected_indices
        
        print(f"筛选后特征数: {X_selected.shape[1]}")
        print(f"筛选后特征索引: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"筛选后特征索引: {selected_indices}")
        
        return X_selected, selected_indices
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5, 
                                 search_method='grid', n_iter=50):
        """
        超参数优化
        
        Parameters:
        -----------
        X_train : ndarray
            训练特征矩阵（已筛选）
        y_train : ndarray
            训练标签
        cv : int
            交叉验证折数
        search_method : str
            搜索方法：'grid' 或 'random'
        n_iter : int
            随机搜索迭代次数
        
        Returns:
        --------
        best_model : RandomForestClassifier
            最优模型
        best_params : dict
            最优超参数
        """
        print("=" * 60)
        print(f"超参数优化 ({search_method} search)")
        print("=" * 60)
        
        # 定义超参数搜索空间
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'max_features': ['sqrt', 'log2', 0.5],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10]
        }
        
        # 计算自定义类别权重：给类别1和类别2更高的权重
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        # 创建类别到样本数的映射
        class_to_count = dict(zip(unique_classes, class_counts))
        
        # 计算基础权重（平衡权重）
        class_weights_balanced = {}
        for cls in unique_classes:
            class_weights_balanced[cls] = n_samples / (n_classes * class_to_count[cls])
        
        # 针对性地提高类别1和类别2的权重（相对于类别0）
        class_weights_custom = class_weights_balanced.copy()
        if 0 in class_weights_custom:
            base_weight = class_weights_custom[0]
            if 1 in class_weights_custom:
                class_weights_custom[1] = base_weight * 2.5  # 进入地库权重提高2.5倍
            if 2 in class_weights_custom:
                class_weights_custom[2] = base_weight * 2.5  # 出地库权重提高2.5倍
        
        print(f"超参数优化使用类别权重:")
        for cls in sorted(unique_classes):
            print(f"  类别 {cls}: {class_weights_custom[cls]:.4f} (样本数: {class_to_count[cls]})")
        
        # 创建基础模型（使用自定义类别权重）
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
            class_weight=class_weights_custom  # 使用自定义类别权重，提高类别1和类别2的权重
        )
        
        # 选择搜索方法
        # 使用f1_macro作为评分指标，更关注少数类别的性能
        if search_method == 'grid':
            print("使用网格搜索...")
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1_macro',  # 使用f1_macro而不是accuracy，更关注类别1和类别2
                n_jobs=-1,
                verbose=1
            )
        else:
            print("使用随机搜索...")
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='f1_macro',  # 使用f1_macro而不是accuracy，更关注类别1和类别2
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        
        # 执行搜索
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        print(f"\n超参数优化完成，耗时: {search_time:.2f} 秒")
        print(f"最优交叉验证得分 (f1_macro): {search.best_score_:.4f}")
        print(f"最优超参数:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        
        self.best_model = search.best_estimator_
        return self.best_model, search.best_params_
    
    def train_final_model(self, X_train, y_train, best_params=None):
        """
        使用最优超参数训练最终模型
        
        Parameters:
        -----------
        X_train : ndarray
            训练特征矩阵（已筛选）
        y_train : ndarray
            训练标签
        best_params : dict or None
            最优超参数，None则使用默认参数
        
        Returns:
        --------
        model : RandomForestClassifier
            训练好的最终模型
        """
        print("=" * 60)
        print("训练最终模型")
        print("=" * 60)
        
        if best_params is None:
            # 使用默认参数
            best_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'max_features': 'sqrt',
                'min_samples_leaf': 2,
                'min_samples_split': 5
            }
        
        # 确保best_params中不包含class_weight，避免冲突
        best_params_clean = {k: v for k, v in best_params.items() if k != 'class_weight'}
        
        # 计算自定义类别权重：给类别1和类别2更高的权重
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        # 创建类别到样本数的映射
        class_to_count = dict(zip(unique_classes, class_counts))
        
        # 计算基础权重（平衡权重）
        class_weights_balanced = {}
        for cls in unique_classes:
            class_weights_balanced[cls] = n_samples / (n_classes * class_to_count[cls])
        
        # 针对性地提高类别1和类别2的权重（相对于类别0）
        class_weights_custom = class_weights_balanced.copy()
        if 0 in class_weights_custom:
            base_weight = class_weights_custom[0]
            if 1 in class_weights_custom:
                class_weights_custom[1] = base_weight * 2.5  # 进入地库权重提高2.5倍
            if 2 in class_weights_custom:
                class_weights_custom[2] = base_weight * 2.5  # 出地库权重提高2.5倍
        
        print(f"最终模型使用类别权重:")
        for cls in sorted(unique_classes):
            print(f"  类别 {cls}: {class_weights_custom[cls]:.4f} (样本数: {class_to_count[cls]})")
        
        self.best_model = RandomForestClassifier(
            **best_params_clean,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
            class_weight=class_weights_custom  # 使用自定义类别权重，提高类别1和类别2的权重
        )
        
        start_time = time.time()
        self.best_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"最终模型训练完成，耗时: {training_time:.2f} 秒")
        
        # 评估模型
        train_score = self.best_model.score(X_train, y_train)
        print(f"训练集准确率: {train_score:.4f}")
        
        # 调试信息：检查训练集预测结果
        train_pred = self.best_model.predict(X_train)
        unique_train_preds = np.unique(train_pred)
        print(f"训练集预测类别: {unique_train_preds}")
        print(f"训练集预测分布: {dict(zip(*np.unique(train_pred, return_counts=True)))}")
        print(f"训练标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # 检查是否所有预测都是0
        if len(unique_train_preds) == 1 and unique_train_preds[0] == 0:
            print("警告: 训练集所有预测结果都是0，模型可能存在问题！")
        
        return self.best_model
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        
        Parameters:
        -----------
        X_test : ndarray
            测试特征矩阵
        y_test : ndarray
            测试标签
        
        Returns:
        --------
        results : dict
            评估结果字典
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，请先训练模型")
        
        print("=" * 60)
        print("模型评估")
        print("=" * 60)
        
        # 预测
        y_pred = self.best_model.predict(X_test)
        
        # 调试信息：检查预测结果
        unique_preds = np.unique(y_pred)
        print(f"\n预测结果类别: {unique_preds}")
        print(f"预测结果分布: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
        print(f"真实标签分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n测试集准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['其他场景', '进入地库', '出地库']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)
        
        results = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_true': y_test,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=['其他场景', '进入地库', '出地库'],
                output_dict=True
            )
        }
        
        return results
    
    def save_model(self, filename='random_forest_model.pkl', best_params=None):
        """
        保存模型和相关信息
        
        Parameters:
        -----------
        filename : str
            保存文件名
        best_params : dict or None
            最优超参数
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        filepath = os.path.join(self.model_dir, filename)
        
        model_data = {
            'model': self.best_model,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
        
        # 计算模型大小
        model_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"模型文件大小: {model_size:.2f} MB")
        
        # 保存模型配置和超参数到JSON文件
        self.save_model_config(best_params=best_params)
    
    def save_model_config(self, best_params=None, config_filename='model_config.json'):
        """
        保存模型配置和超参数到JSON文件
        
        Parameters:
        -----------
        best_params : dict or None
            最优超参数
        config_filename : str
            配置文件名称
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，无法保存配置")
        
        # 获取模型参数
        model_params = self.best_model.get_params()
        
        # 构建配置字典
        config = {
            'model_type': 'RandomForestClassifier',
            'model_config': {
                'n_estimators': model_params.get('n_estimators'),
                'max_depth': model_params.get('max_depth'),
                'max_features': str(model_params.get('max_features')),
                'min_samples_leaf': model_params.get('min_samples_leaf'),
                'min_samples_split': model_params.get('min_samples_split'),
                'class_weight': str(model_params.get('class_weight', 'balanced')),
                'random_state': model_params.get('random_state'),
                'n_jobs': model_params.get('n_jobs')
            },
            'hyperparameters': best_params if best_params else model_params,
            'feature_selection': {
                'n_selected_features': len(self.selected_features) if self.selected_features is not None else None,
                'selected_feature_indices': self.selected_features.tolist() if self.selected_features is not None else None
            },
            'training_info': {
                'n_trees': len(self.best_model.estimators_),
                'total_nodes': sum(tree.tree_.node_count for tree in self.best_model.estimators_),
                'feature_importance_available': self.feature_importance is not None
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到JSON文件
        config_path = os.path.join(self.model_dir, config_filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"模型配置已保存到: {config_path}")
    
    def load_model(self, filename='random_forest_model.pkl'):
        """
        加载模型
        
        Parameters:
        -----------
        filename : str
            文件名
        """
        filepath = os.path.join(self.model_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['model']
            self.selected_features = model_data.get('selected_features')
            self.feature_importance = model_data.get('feature_importance')
            
            print(f"模型已加载: {filepath}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {filepath}")


def main():
    """主函数：完整的模型训练流程"""
    print("=" * 60)
    print("随机森林模型训练流程")
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
        
        # 标准化（使用已加载的预处理器）
        X_train = cleaner.transform_features(X_train, is_train=False)
        
        # 对于测试集，使用process_train_data中的测试集（用于评估）
        # 注意：测试集仍然从原始数据划分，用于评估模型性能
        print("\n从原始数据生成测试集（用于模型评估）...")
        _, X_test, _, y_test = cleaner.process_train_data(
            remove_outliers_flag=True,
            fill_missing_flag=True,
            standardize=True,
            test_size=0.2
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
            test_size=0.2
        )
    
    # 获取特征名称
    feature_names = cleaner.feature_columns
    
    # 2. 创建训练器
    trainer = RandomForestTrainer(model_dir='model/', random_state=42)
    
    # 3. 训练初始模型（用于特征重要性评估）
    trainer.train_initial_model(
        X_train, y_train,
        n_estimators=100,
        max_depth=20,
        max_features='sqrt',
        min_samples_leaf=2
    )
    
    # 4. 评估特征重要性
    importance_df = trainer.evaluate_feature_importance(
        feature_names=feature_names,
        top_n=20
    )
    
    # 5. 特征筛选（保留前70%的特征）
    X_train_selected, selected_indices = trainer.select_features(
        X_train, y_train,
        feature_names=feature_names,
        top_ratio=1
    )
    
    # 对测试集也进行特征筛选
    X_test_selected = X_test[:, selected_indices]
    
    # 6. 超参数优化（使用随机搜索以节省时间）
    best_model, best_params = trainer.optimize_hyperparameters(
        X_train_selected, y_train,
        cv=5,
        search_method='random',  # 可以改为 'grid' 进行网格搜索
        n_iter=30  # 随机搜索迭代次数
    )
    
    # 7. 训练最终模型
    final_model = trainer.train_final_model(
        X_train_selected, y_train,
        best_params=best_params
    )
    
    # 8. 模型评估
    results = trainer.evaluate_model(X_test_selected, y_test)
    
    # 9. 保存模型（包含配置）
    trainer.save_model('random_forest_model.pkl', best_params=best_params)
    
    # 10. 对清洗训练数据进行预测并回填结果
    print("\n" + "=" * 60)
    print("对清洗训练数据进行预测并回填结果")
    print("=" * 60)
    cleaned_train_path = 'train/清洗训练数据.csv'
    if os.path.exists(cleaned_train_path):
        # 加载清洗训练数据
        df_train_cleaned = pd.read_csv(cleaned_train_path)
        print(f"加载清洗训练数据: {df_train_cleaned.shape}")
        
        # 准备特征（不包含result和labelArea列，但需要包含labelArea用于计算准确率）
        # 先保存labelArea用于后续计算
        if 'labelArea' in df_train_cleaned.columns:
            y_train_true = df_train_cleaned['labelArea'].copy()
        else:
            y_train_true = None
        
        # 准备特征（排除result和labelArea列）
        exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                       'labelMovement', 'result', 'labelArea']
        feature_cols = [col for col in df_train_cleaned.columns if col not in exclude_cols]
        
        # 确保特征列顺序与训练时一致
        if cleaner.feature_columns is not None:
            feature_cols = [col for col in cleaner.feature_columns if col in feature_cols]
            X_train_cleaned = df_train_cleaned[feature_cols].copy()
            # 如果缺少某些特征列，用0填充
            missing_cols = set(cleaner.feature_columns) - set(feature_cols)
            if missing_cols:
                print(f"警告: 清洗训练数据缺少特征列: {len(missing_cols)} 个")
                for col in missing_cols:
                    X_train_cleaned[col] = 0
            # 确保列顺序一致
            X_train_cleaned = X_train_cleaned[cleaner.feature_columns]
        else:
            X_train_cleaned = df_train_cleaned[feature_cols].copy()
        
        # 标准化
        X_train_cleaned_scaled = cleaner.transform_features(X_train_cleaned, is_train=False)
        
        # 特征筛选
        if trainer.selected_features is not None:
            X_train_cleaned_scaled = X_train_cleaned_scaled[:, trainer.selected_features]
        
        # 预测
        predictions_train = trainer.best_model.predict(X_train_cleaned_scaled)
        
        # 回填到result列
        df_train_cleaned['result'] = predictions_train
        
        # 保存
        df_train_cleaned.to_csv(cleaned_train_path, index=False, encoding='utf-8-sig')
        print(f"预测结果已回填到: {cleaned_train_path}")
        print(f"预测类别分布:\n{pd.Series(predictions_train).value_counts().sort_index()}")
        
        # 计算训练集准确率
        if y_train_true is not None:
            train_accuracy = (predictions_train == y_train_true.values).mean()
            print(f"清洗训练数据准确率: {train_accuracy:.4f}")
            
            # 显示各类别的准确率
            from sklearn.metrics import classification_report
            print("\n训练集分类报告:")
            print(classification_report(y_train_true, predictions_train, 
                                       target_names=['其他场景', '进入地库', '出地库']))
    else:
        print(f"警告: 清洗训练数据文件不存在: {cleaned_train_path}")
    
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

