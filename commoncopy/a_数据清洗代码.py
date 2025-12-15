"""
数据清洗与预处理模块
功能：数据加载、异常值处理、缺失值填充、特征标准化、数据拆分
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle


class DataCleaner:
    """数据清洗与预处理类"""
    
    def __init__(self, train_path='train/训练数据.csv', test_dir='test/', 
                 output_dir='./model/', random_state=42):
        """
        初始化数据清洗器
        
        Parameters:
        -----------
        train_path : str
            训练数据文件路径
        test_dir : str
            测试数据目录路径
        output_dir : str
            输出目录路径（用于保存预处理对象）
        random_state : int
            随机种子
        """
        self.train_path = train_path
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.random_state = random_state
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        
        # 存储特征列名和标签列名
        self.feature_columns = None
        self.label_column = 'labelArea'
        
    def load_train_data(self):
        """
        加载训练数据
        
        Returns:
        --------
        df : DataFrame
            训练数据
        """
        print(f"正在加载训练数据: {self.train_path}")
        df = pd.read_csv(self.train_path)
        print(f"训练数据形状: {df.shape}")
        print(f"类别分布:\n{df[self.label_column].value_counts().sort_index()}")
        return df
    
    def load_test_data(self):
        """
        加载所有测试数据文件
        
        Returns:
        --------
        test_files : dict
            字典，键为文件名，值为DataFrame
        """
        test_files = {}
        if os.path.exists(self.test_dir):
            for filename in os.listdir(self.test_dir):
                if filename.endswith('.csv'):
                    filepath = os.path.join(self.test_dir, filename)
                    print(f"正在加载测试文件: {filename}")
                    df = pd.read_csv(filepath)
                    test_files[filename] = df
                    print(f"  数据形状: {df.shape}")
        return test_files
    
    def remove_outliers(self, df, columns=None, threshold=3):
        """
        移除异常值（使用3倍标准差方法）
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list or None
            要处理的列，None表示处理所有数值列
        threshold : float
            标准差倍数阈值
        
        Returns:
        --------
        df_cleaned : DataFrame
            清洗后的数据
        """
        df_cleaned = df.copy()
        
        if columns is None:
            # 选择数值列（排除标识列和标签列）
            exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                           self.label_column, 'labelMovement']
            columns = [col for col in df_cleaned.columns 
                      if col not in exclude_cols and 
                      df_cleaned[col].dtype in [np.float64, np.int64]]
        
        print(f"正在处理异常值，阈值: {threshold}倍标准差")
        initial_count = len(df_cleaned)
        
        for col in columns:
            if col in df_cleaned.columns:
                mean = df_cleaned[col].mean()
                std = df_cleaned[col].std()
                
                if std > 0:  # 避免除零
                    # 标记异常值
                    mask = np.abs(df_cleaned[col] - mean) <= threshold * std
                    # 将异常值替换为边界值
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    df_cleaned.loc[~mask, col] = np.clip(
                        df_cleaned.loc[~mask, col], 
                        lower_bound, 
                        upper_bound
                    )
        
        print(f"异常值处理完成，样本数: {initial_count} -> {len(df_cleaned)}")
        return df_cleaned
    
    def fill_missing_values(self, df, strategy='median'):
        """
        填充缺失值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        strategy : str
            填充策略：'mean', 'median', 'mode', 'zero'
        
        Returns:
        --------
        df_filled : DataFrame
            填充后的数据
        """
        df_filled = df.copy()
        
        print(f"正在填充缺失值，策略: {strategy}")
        
        # 选择数值列
        exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                       self.label_column, 'labelMovement']
        numeric_cols = [col for col in df_filled.columns 
                       if col not in exclude_cols and 
                       df_filled[col].dtype in [np.float64, np.int64]]
        
        missing_count = df_filled[numeric_cols].isnull().sum().sum()
        if missing_count > 0:
            print(f"发现缺失值: {missing_count} 个")
            
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = df_filled[col].mean()
                    elif strategy == 'median':
                        fill_value = df_filled[col].median()
                    elif strategy == 'mode':
                        fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
                    elif strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = df_filled[col].median()
                    
                    df_filled[col].fillna(fill_value, inplace=True)
        else:
            print("未发现缺失值")
        
        return df_filled
    
    def prepare_features(self, df, is_train=True):
        """
        准备特征矩阵和标签向量
        确保只使用有效的特征列（根据feature_columns或特征列表文件）
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        is_train : bool
            是否为训练数据
        
        Returns:
        --------
        X : DataFrame or ndarray
            特征矩阵
        y : Series or None
            标签向量（测试数据为None）
        """
        # 排除非特征列（这些列永远不应该作为特征）
        exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                       'labelMovement', 'labelArea', 'result']
        
        if is_train:
            # 训练数据：使用feature_columns（如果已设置）或排除后的所有列
            if self.feature_columns is not None:
                # 使用已设置的特征列（确保顺序一致）
                feature_cols = [col for col in self.feature_columns if col in df.columns]
                X = df[feature_cols].copy()
                
                # 如果缺少某些特征列，用0填充
                missing_cols = set(self.feature_columns) - set(feature_cols)
                if missing_cols:
                    print(f"警告: 训练数据缺少特征列: {len(missing_cols)} 个，将用0填充")
                    for col in missing_cols:
                        X[col] = 0
                # 确保列顺序一致
                X = X[self.feature_columns]
            else:
                # 如果feature_columns未设置，排除标识列和标签列
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                X = df[feature_cols].copy()
                # 保存特征列名
                self.feature_columns = feature_cols
            
            y = df[self.label_column].copy()
            
            print(f"训练特征列数: {len(X.columns) if hasattr(X, 'columns') else X.shape[1]}")
            return X, y
        else:
            # 测试数据：必须使用训练时设置的特征列（确保一致性）
            if self.feature_columns is None:
                raise ValueError("特征列未设置，请先处理训练数据或加载预处理器")
            
            # 只保留训练时使用的特征列
            feature_cols = [col for col in self.feature_columns if col in df.columns]
            X = df[feature_cols].copy()
            
            # 如果缺少某些特征列，用0填充
            missing_cols = set(self.feature_columns) - set(feature_cols)
            if missing_cols:
                print(f"警告: 测试数据缺少特征列: {len(missing_cols)} 个，将用0填充")
                for col in missing_cols:
                    X[col] = 0
            
            # 确保列顺序与训练时一致
            X = X[self.feature_columns]
            
            print(f"测试特征列数: {len(X.columns) if hasattr(X, 'columns') else X.shape[1]}")
            return X, None
    
    def split_data(self, df, test_size=0.2):
        """
        按批次划分训练集和本地测试集（避免数据泄露）
        
        为什么按批次划分：
        - 同一批次内的样本有时序相关性
        - 随机划分会导致数据泄露（模型可能看到"未来"数据）
        - 按批次划分更接近真实的实时检测场景
        
        Parameters:
        -----------
        df : DataFrame
            包含特征、标签和group_name的完整数据
        test_size : float
            测试集比例
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : 
            划分后的数据（numpy数组）
        """
        print(f"正在按批次划分数据，测试集比例: {test_size}")
        print("注意：按批次划分可以避免同一批次内的样本同时出现在训练集和测试集中，防止数据泄露")
        
        # 强制要求必须有group_name列
        if 'group_name' not in df.columns:
            raise ValueError(
                "错误: 数据中缺少必需的 'group_name' 列！\n"
                "必须按照 group_name 进行批次划分，不能使用随机划分（会导致数据泄露）。\n"
                "请确保数据中包含 'group_name' 列。"
            )
        
        # 获取所有唯一的批次名称
        unique_groups = df['group_name'].unique()
        n_groups = len(unique_groups)
        print(f"总批次数量: {n_groups}")
        
        # 对批次进行排序以确保可重复性，然后打乱
        unique_groups_sorted = sorted(unique_groups)
        np.random.seed(self.random_state)
        np.random.shuffle(unique_groups_sorted)
        
        # 计算训练集和测试集的批次数量
        n_test_groups = int(n_groups * test_size)
        train_groups = set(unique_groups_sorted[:-n_test_groups] if n_test_groups > 0 else unique_groups_sorted)
        test_groups = set(unique_groups_sorted[-n_test_groups:] if n_test_groups > 0 else [])
        
        print(f"训练集批次数量: {len(train_groups)}, 测试集批次数量: {len(test_groups)}")
        
        # 根据批次划分数据
        train_df = df[df['group_name'].isin(train_groups)].copy()
        test_df = df[df['group_name'].isin(test_groups)].copy()
        
        print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")
        print(f"训练集类别分布:\n{train_df[self.label_column].value_counts().sort_index()}")
        print(f"测试集类别分布:\n{test_df[self.label_column].value_counts().sort_index()}")
        
        # 准备特征和标签
        X_train_df, y_train = self.prepare_features(train_df, is_train=True)
        X_test_df, y_test = self.prepare_features(test_df, is_train=True)
        
        # 转换为numpy数组
        if isinstance(X_train_df, pd.DataFrame):
            X_train = X_train_df.values
        else:
            X_train = X_train_df
            
        if isinstance(X_test_df, pd.DataFrame):
            X_test = X_test_df.values
        else:
            X_test = X_test_df
            
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        print(f"训练集特征形状: {X_train.shape}, 测试集特征形状: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train):
        """
        拟合标准化器
        
        Parameters:
        -----------
        X_train : DataFrame or ndarray
            训练特征矩阵
        """
        print("正在拟合标准化器...")
        self.scaler.fit(X_train)
        print("标准化器拟合完成")
    
    def transform_features(self, X, is_train=False):
        """
        标准化特征
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            特征矩阵
        is_train : bool
            是否为训练数据（训练数据需要fit，测试数据只需要transform）
        
        Returns:
        --------
        X_scaled : ndarray
            标准化后的特征矩阵
        """
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def save_preprocessor(self, filename='preprocessor.pkl'):
        """
        保存预处理对象（标准化器等）
        
        Parameters:
        -----------
        filename : str
            保存文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        preprocessor = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"预处理对象已保存到: {filepath}")
    
    def load_preprocessor(self, filename='preprocessor.pkl'):
        """
        加载预处理对象
        
        Parameters:
        -----------
        filename : str
            文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                preprocessor = pickle.load(f)
            self.scaler = preprocessor['scaler']
            self.feature_columns = preprocessor['feature_columns']
            print(f"预处理对象已加载: {filepath}")
        else:
            print(f"警告: 预处理对象文件不存在: {filepath}")
    
    def get_feature_columns(self, df, feature_list_path='top_150_features.csv'):
        """
        根据特征列表文件获取特征列
        只保留top_150_features.csv中列出的特征（排除标识列、标签列和结果列）
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        feature_list_path : str
            特征列表文件路径（CSV格式，包含feature_name列）
        
        Returns:
        --------
        feature_columns : list
            特征列名列表（只包含特征列表文件中列出的特征）
        """
        # 排除的列：标识列、标签列、结果列（这些不应该在特征列表中）
        exclude_cols = [
            'time',                    # 时间标识列
            'group_name',              # 批次标识列
            'sufficient_window_size',  # 窗口大小标识列
            'labelArea',               # 标签列（避免标签泄露）
            'labelMovement',           # 其他标签列
            'result'                   # 预测结果列（不应作为特征）
        ]
        
        # 尝试从特征列表文件读取
        valid_feature_names = None
        if feature_list_path is not None and os.path.exists(feature_list_path):
            try:
                feature_df = pd.read_csv(feature_list_path)
                if 'feature_name' in feature_df.columns:
                    # 获取特征名称列表，排除标识列和结果列
                    valid_feature_names = feature_df['feature_name'].tolist()
                    # 过滤掉标识列和结果列
                    valid_feature_names = [name for name in valid_feature_names 
                                         if name not in exclude_cols]
                    print(f"从特征列表文件读取: {feature_list_path}")
                    print(f"特征列表中的特征数: {len(valid_feature_names)}")
            except Exception as e:
                print(f"警告: 读取特征列表文件失败: {e}")
                print("将使用数据中的所有特征列（排除标识列和标签列）")
                valid_feature_names = None
        
        # 获取数据中的所有列
        all_cols = df.columns.tolist()
        
        if valid_feature_names is not None:
            # 只保留特征列表中存在的特征（同时确保在数据中存在）
            feature_columns = [col for col in valid_feature_names if col in all_cols]
            
            # 检查是否有特征列表中的特征在数据中不存在
            missing_features = [col for col in valid_feature_names if col not in all_cols]
            if missing_features:
                print(f"警告: 特征列表中有 {len(missing_features)} 个特征在数据中不存在")
                print(f"缺失的特征示例: {missing_features[:5]}...")
        else:
            # 如果没有特征列表文件，使用所有列（排除标识列、标签列和结果列）
            feature_columns = [col for col in all_cols if col not in exclude_cols]
        
        print(f"总列数: {len(all_cols)}")
        print(f"最终特征列数: {len(feature_columns)}")
        if valid_feature_names:
            print(f"特征覆盖率: {len(feature_columns)}/{len(valid_feature_names)} ({len(feature_columns)/len(valid_feature_names)*100:.1f}%)")
        
        return feature_columns
    
    def clean_and_split_data(self, train_ratio=0.8, remove_outliers_flag=True, 
                            fill_missing_flag=True, random_state=None, 
                            label_zero_keep_ratio=0.5):
        """
        清洗数据并划分训练集和测试集
        按批次（group_name）划分：将80%的批次作为训练数据，20%的批次作为测试数据
        这样可以避免同一批次内的样本同时出现在训练集和测试集中，防止数据泄露
        
        为什么按批次划分：
        - 同一批次内的样本有时序相关性
        - 随机划分会导致数据泄露（模型可能看到"未来"数据）
        - 按批次划分更接近真实的实时检测场景
        
        Parameters:
        -----------
        train_ratio : float
            训练集比例（默认0.8）
        remove_outliers_flag : bool
            是否移除异常值
        fill_missing_flag : bool
            是否填充缺失值
        random_state : int or None
            随机种子（用于批次划分的随机性）
        label_zero_keep_ratio : float
            标签为0的样本保留比例（0-1之间，默认0.5，即50%）
            在保存测试数据时，会对标签为0的样本进行随机采样
        
        Returns:
        --------
        train_df : DataFrame
            清洗后的训练数据（包含result列）
        test_df : DataFrame
            清洗后的测试数据
        """
        if random_state is None:
            random_state = self.random_state
        
        # 1. 加载数据
        df = self.load_train_data()
        
        # 2. 获取特征列（用于特征工程，根据top_150_features.csv）
        # 尝试从多个位置查找特征列表文件
        feature_list_paths = [
            'top_150_features.csv',
            '../top_150_features.csv',
            '../../top_150_features.csv',
            os.path.join(os.path.dirname(os.path.dirname(self.train_path)), 'top_150_features.csv')
        ]
        feature_list_path = None
        for path in feature_list_paths:
            if os.path.exists(path):
                feature_list_path = path
                break
        
        if feature_list_path:
            print(f"找到特征列表文件: {feature_list_path}")
            feature_columns = self.get_feature_columns(df, feature_list_path=feature_list_path)
        else:
            print("警告: 未找到top_150_features.csv文件，将使用数据中的所有特征列")
            feature_columns = self.get_feature_columns(df, feature_list_path=None)
        
        # 3. 数据清洗（在完整数据上进行，保留所有列以便后续处理）
        if remove_outliers_flag:
            df = self.remove_outliers(df)
        
        if fill_missing_flag:
            df = self.fill_missing_values(df)
        
        # 4. 清洗后只保留需要的列：特征列 + 必要的标识列和标签列
        # 训练数据需要保留：特征列 + 标识列 + labelArea（用于训练）
        # 测试数据需要保留：特征列 + 标识列（不包含labelArea，避免标签泄露）
        required_cols_train = feature_columns + ['time', 'group_name', 'sufficient_window_size', 'labelArea', 'labelMovement']
        required_cols_train = [col for col in required_cols_train if col in df.columns]
        df = df[required_cols_train].copy()
        
        print(f"清洗后保留列数: {len(required_cols_train)} (特征列: {len(feature_columns)}, 标识列: 3, 标签列: 2)")
        print(f"清洗后数据形状: {df.shape}")
        
        # 5. 按批次划分训练集和测试集
        print(f"正在按批次划分数据，训练集比例: {train_ratio}")
        print("注意：按批次划分可以避免同一批次内的样本同时出现在训练集和测试集中，防止数据泄露")
        
        # 强制要求必须有group_name列
        if 'group_name' not in df.columns:
            raise ValueError(
                "错误: 数据中缺少必需的 'group_name' 列！\n"
                "必须按照 group_name 进行批次划分，不能使用随机划分（会导致数据泄露）。\n"
                "请确保数据中包含 'group_name' 列。"
            )
        
        # 获取所有唯一的批次名称
        unique_groups = df['group_name'].unique()
        n_groups = len(unique_groups)
        print(f"总批次数量: {n_groups}")
        
        # 对批次进行排序以确保可重复性，然后打乱
        unique_groups_sorted = sorted(unique_groups)
        np.random.seed(random_state)
        np.random.shuffle(unique_groups_sorted)
        
        # 计算训练集和测试集的批次数量
        n_train_groups = int(n_groups * train_ratio)
        train_groups = set(unique_groups_sorted[:n_train_groups])
        test_groups = set(unique_groups_sorted[n_train_groups:])
        
        print(f"训练集批次数量: {len(train_groups)}, 测试集批次数量: {len(test_groups)}")
        
        # 根据批次划分数据
        train_df = df[df['group_name'].isin(train_groups)].copy()
        test_df = df[df['group_name'].isin(test_groups)].copy()
        
        print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")
        print(f"训练集批次: {sorted(train_groups)[:5]}..." if len(train_groups) > 5 else f"训练集批次: {sorted(train_groups)}")
        print(f"测试集批次: {sorted(test_groups)[:5]}..." if len(test_groups) > 5 else f"测试集批次: {sorted(test_groups)}")
        
        # 6. 对训练数据中标签为0的样本进行随机采样（保留指定比例）
        # 重要：必须先保留labelArea列，进行采样，然后再删除labelArea列
        print(f"\n对训练数据中标签为0的样本进行随机采样，保留比例: {label_zero_keep_ratio*100:.1f}%")
        original_train_count = len(train_df)
        
        # 确保train_df包含labelArea列（用于采样判断）
        if self.label_column not in train_df.columns:
            raise ValueError(f"训练数据中缺少标签列: {self.label_column}，无法进行采样")
        
        # 分离标签为0和其他标签的样本（此时labelArea列还在）
        label_zero_mask = train_df[self.label_column] == 0
        label_zero_count = label_zero_mask.sum()
        label_nonzero_count = (~label_zero_mask).sum()
        
        print(f"训练数据原始行数: {original_train_count}")
        print(f"  标签为0的样本数: {label_zero_count}")
        print(f"  标签非0的样本数: {label_nonzero_count}")
        
        # 对标签为0的样本进行随机采样（此时labelArea列还在，可以用于判断）
        train_df_sampled = train_df.copy()
        if label_zero_count > 0 and label_zero_keep_ratio < 1.0:
            # 设置随机种子以确保可重复性
            if random_state is not None:
                np.random.seed(random_state)
            
            # 获取标签为0的样本索引（使用位置索引，不是标签索引）
            label_zero_indices = train_df_sampled.index[label_zero_mask].tolist()
            
            # 计算需要保留的数量
            n_keep = int(label_zero_count * label_zero_keep_ratio)
            if n_keep < 1 and label_zero_count > 0:
                n_keep = 1  # 至少保留1个
            
            if n_keep >= label_zero_count:
                print(f"  保留比例导致保留所有标签为0的样本: {label_zero_count}")
            else:
                # 随机选择要保留的索引
                keep_indices = set(np.random.choice(label_zero_indices, size=n_keep, replace=False))
                
                # 获取要删除的索引（标签为0但未被选中的）
                drop_indices = [idx for idx in label_zero_indices if idx not in keep_indices]
                
                # 删除未被选中的标签为0的样本
                if drop_indices:
                    train_df_sampled = train_df_sampled.drop(index=drop_indices)
                    print(f"  保留标签为0的样本数: {n_keep} (从 {label_zero_count} 个中采样)")
                    print(f"  删除标签为0的样本数: {len(drop_indices)}")
                    
                    # 验证采样结果
                    label_zero_after = (train_df_sampled[self.label_column] == 0).sum()
                    print(f"  验证: 采样后标签为0的样本数: {label_zero_after} (预期: {n_keep})")
                    if label_zero_after != n_keep:
                        print(f"  警告: 采样后标签为0的样本数与预期不符！")
        else:
            print(f"  保留所有标签为0的样本: {label_zero_count}")
        
        final_train_count = len(train_df_sampled)
        print(f"训练数据采样后行数: {final_train_count} (减少了 {original_train_count - final_train_count} 行)")
        
        # 更新train_df为采样后的数据
        train_df = train_df_sampled
        
        # 7. 在训练数据中，在sufficient_window_size与labelArea之间插入result列（值为空）
        # 找到sufficient_window_size和labelArea的位置
        cols = train_df.columns.tolist()
        sufficient_idx = cols.index('sufficient_window_size')
        label_idx = cols.index('labelArea')
        
        # 在labelArea之前插入result列
        new_cols = cols[:label_idx] + ['result'] + cols[label_idx:]
        train_df = train_df.reindex(columns=new_cols)
        train_df['result'] = ''  # 设置为空字符串
        
        # 8. 保存清洗后的训练数据
        train_output_path = os.path.join(os.path.dirname(self.train_path), '清洗训练数据.csv')
        train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后的训练数据已保存: {train_output_path}")
        print(f"最终训练数据行数: {len(train_df)}")
        
        # 9. 保存测试数据（不包含labelArea列，避免标签泄露）
        test_output_path = os.path.join(os.path.dirname(self.train_path), '清洗测试数据.csv')
        
        # 测试数据只保留：特征列 + 标识列 + result列（用于存放预测结果）
        # 不包含labelArea列和labelMovement列（避免标签泄露）
        test_cols = feature_columns + ['time', 'group_name', 'sufficient_window_size']
        test_cols = [col for col in test_cols if col in test_df.columns]
        
        # 确保所有测试数据都被保存（不进行任何过滤）
        test_df_output = test_df[test_cols].copy()
        
        # 检查数据完整性
        print(f"测试数据原始行数: {len(test_df)}")
        print(f"测试数据输出行数: {len(test_df_output)}")
        if len(test_df_output) != len(test_df):
            print(f"警告: 测试数据行数不匹配！原始: {len(test_df)}, 输出: {len(test_df_output)}")
        
        # 在sufficient_window_size之后插入result列
        test_cols_list = test_df_output.columns.tolist()
        if 'sufficient_window_size' in test_cols_list:
            sufficient_idx_test = test_cols_list.index('sufficient_window_size')
            test_cols_list.insert(sufficient_idx_test + 1, 'result')
        else:
            test_cols_list.append('result')
        test_df_output = test_df_output.reindex(columns=test_cols_list)
        test_df_output['result'] = ''  # 设置为空字符串
        
        # 保存测试数据（确保所有行都被保存）
        test_df_output.to_csv(test_output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后的测试数据已保存: {test_output_path}")
        print(f"测试数据行数: {len(test_df_output)}")
        print(f"测试数据列: {len(test_cols_list)} 列（特征列: {len(feature_columns)}, 标识列: 3, result列: 1）")
        print(f"测试数据不包含labelArea和labelMovement列（避免标签泄露）")
        
        # 验证保存的数据
        if os.path.exists(test_output_path):
            saved_df = pd.read_csv(test_output_path)
            print(f"验证: 保存的文件行数: {len(saved_df)}")
            if len(saved_df) != len(test_df_output):
                print(f"错误: 保存的文件行数不匹配！预期: {len(test_df_output)}, 实际: {len(saved_df)}")
        
        return train_df, test_df
    
    def process_train_data(self, remove_outliers_flag=True, fill_missing_flag=True, 
                           standardize=True, test_size=0.2):
        """
        完整处理训练数据流程
        
        Parameters:
        -----------
        remove_outliers_flag : bool
            是否移除异常值
        fill_missing_flag : bool
            是否填充缺失值
        standardize : bool
            是否标准化
        test_size : float
            测试集比例
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : 
            处理后的数据
        """
        # 1. 加载数据
        df = self.load_train_data()
        
        # 2. 数据清洗
        if remove_outliers_flag:
            df = self.remove_outliers(df)
        
        if fill_missing_flag:
            df = self.fill_missing_values(df)
        
        # 3. 获取特征列（根据top_150_features.csv或所有特征列）
        # 尝试从多个位置查找特征列表文件
        feature_list_paths = [
            'top_150_features.csv',
            '../top_150_features.csv',
            '../../top_150_features.csv',
            os.path.join(os.path.dirname(os.path.dirname(self.train_path)), 'top_150_features.csv')
        ]
        feature_list_path = None
        for path in feature_list_paths:
            if os.path.exists(path):
                feature_list_path = path
                break
        
        if feature_list_path:
            print(f"找到特征列表文件: {feature_list_path}")
            feature_columns = self.get_feature_columns(df, feature_list_path=feature_list_path)
        else:
            print("警告: 未找到top_150_features.csv文件，将使用数据中的所有特征列")
            feature_columns = self.get_feature_columns(df, feature_list_path=None)
        
        # 保存特征列（确保后续使用一致）
        self.feature_columns = feature_columns
        print(f"最终使用的特征列数: {len(self.feature_columns)}")
        
        # 4. 按批次划分数据（避免数据泄露）
        # split_data方法现在接受完整的DataFrame，内部会按批次划分并提取特征
        X_train, X_test, y_train, y_test = self.split_data(df, test_size=test_size)
        
        # 5. 特征标准化
        if standardize:
            self.fit_scaler(X_train)
            X_train = self.transform_features(X_train, is_train=False)
            X_test = self.transform_features(X_test, is_train=False)
        
        # 6. 保存预处理对象
        self.save_preprocessor()
        
        return X_train, X_test, y_train, y_test
    
    def process_test_data(self, test_df):
        """
        处理单个测试数据文件
        
        Parameters:
        -----------
        test_df : DataFrame
            测试数据
        
        Returns:
        --------
        X_test : ndarray
            处理后的特征矩阵
        """
        # 加载预处理对象（如果尚未加载）
        if self.feature_columns is None:
            self.load_preprocessor()
        
        # 填充缺失值
        test_df = self.fill_missing_values(test_df)
        
        # 准备特征
        X_test, _ = self.prepare_features(test_df, is_train=False)
        
        # 标准化
        X_test = self.transform_features(X_test, is_train=False)
        
        return X_test


def main():
    """主函数：演示数据清洗流程"""
    # 创建数据清洗器
    cleaner = DataCleaner(
        train_path='train/训练数据.csv',
        test_dir='test/',
        output_dir='model/'
    )
    
    # 清洗数据并划分训练集和测试集
    train_df, test_df = cleaner.clean_and_split_data(
        train_ratio=0.8,
        remove_outliers_flag=True,
        fill_missing_flag=True,
        random_state=42,
        label_zero_keep_ratio=0.06 # 保留50%的标签为0的样本
    )
    
    print("\n数据清洗完成！")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"训练集列名: {train_df.columns.tolist()[:10]}...")
    print(f"测试集列名: {test_df.columns.tolist()[:10]}...")


if __name__ == '__main__':
    main()

