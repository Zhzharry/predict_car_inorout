"""
数据清洗与预处理模块
功能：数据加载、异常值处理、缺失值填充、特征标准化、数据拆分
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
        
        # 特征工程（多项式特征，用于提升准确率）
        self.poly_features = None
        self.use_poly_features = True  # 启用多项式特征以提升准确率
        
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
    
    def remove_outliers(self, df, columns=None, threshold=2.5, method='std'):
        """
        移除异常值（使用更保守的方法以提高准确率）
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list or None
            要处理的列，None表示处理所有数值列
        threshold : float
            标准差倍数阈值（降低到2.5以提高数据保留率）
        method : str
            方法：'std'（标准差）或'iqr'（四分位距）
        
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
        
        print(f"正在处理异常值，方法: {method}, 阈值: {threshold}")
        initial_count = len(df_cleaned)
        outliers_count = 0
        
        for col in columns:
            if col in df_cleaned.columns:
                if method == 'iqr':
                    # 使用IQR方法（更保守，保留更多数据）
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                else:
                    # 使用标准差方法（降低阈值以保留更多数据）
                    mean = df_cleaned[col].mean()
                    std = df_cleaned[col].std()
                    
                    if std > 0:  # 避免除零
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                    else:
                        continue
                
                # 标记异常值
                mask = (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
                outliers_count += (~mask).sum()
                
                # 将异常值替换为边界值（而不是删除，保留更多数据）
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
        
        print(f"异常值处理完成，样本数: {initial_count}, 处理异常值: {outliers_count} 个")
        return df_cleaned
    
    def fill_missing_values(self, df, strategy='median', use_forward_fill=True):
        """
        填充缺失值（使用更精细的策略以提高准确率）
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        strategy : str
            填充策略：'mean', 'median', 'mode', 'zero', 'interpolate'
        use_forward_fill : bool
            是否先尝试前向填充（对时序数据更有效）
        
        Returns:
        --------
        df_filled : DataFrame
            填充后的数据
        """
        df_filled = df.copy()
        
        print(f"正在填充缺失值，策略: {strategy}, 前向填充: {use_forward_fill}")
        
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
                    # 先尝试前向填充（对时序数据更有效）
                    if use_forward_fill:
                        df_filled[col] = df_filled[col].ffill()
                        df_filled[col] = df_filled[col].bfill()
                    
                    # 如果还有缺失值，使用指定策略
                    if df_filled[col].isnull().any():
                        if strategy == 'interpolate':
                            # 使用线性插值（对时序数据更准确）
                            df_filled[col] = df_filled[col].interpolate(method='linear')
                            df_filled[col] = df_filled[col].bfill()
                            df_filled[col] = df_filled[col].ffill()
                        elif strategy == 'mean':
                            fill_value = df_filled[col].mean()
                            df_filled[col].fillna(fill_value, inplace=True)
                        elif strategy == 'median':
                            fill_value = df_filled[col].median()
                            df_filled[col].fillna(fill_value, inplace=True)
                        elif strategy == 'mode':
                            fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
                            df_filled[col].fillna(fill_value, inplace=True)
                        elif strategy == 'zero':
                            df_filled[col].fillna(0, inplace=True)
                        else:
                            fill_value = df_filled[col].median()
                            df_filled[col].fillna(fill_value, inplace=True)
        else:
            print("未发现缺失值")
        
        return df_filled
    
    def prepare_features(self, df, is_train=True):
        """
        准备特征矩阵和标签向量（优化：只使用有效特征列）
        
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
        # 排除非特征列（标识列和标签列）
        exclude_cols = ['time', 'group_name', 'sufficient_window_size', 
                       'labelMovement']
        
        if is_train:
            exclude_cols.append(self.label_column)
            exclude_cols.append('result')  # 结果列不作为特征
            
            # 只使用有效的特征列（如果已经通过get_valid_columns筛选过）
            if self.feature_columns is not None:
                # 从有效列中提取特征列（排除标识列和标签列）
                feature_cols = [col for col in self.feature_columns 
                              if col not in exclude_cols]
            else:
                # 如果还没有筛选，使用所有数值列
                feature_cols = [col for col in df.columns 
                              if col not in exclude_cols and
                              df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
            
            X = df[feature_cols].copy()
            y = df[self.label_column].copy()
            
            # 保存特征列名
            if self.feature_columns is None:
                # 如果还没有设置，保存当前使用的特征列
                self.feature_columns = feature_cols
            
            print(f"训练数据特征数: {len(feature_cols)}")
            return X, y
        else:
            # 测试数据不应该包含labelArea列（避免标签泄露）
            # 也不应该包含result列（这是预测结果列）
            exclude_cols_test = exclude_cols + ['labelArea', 'result']
            
            # 确保特征列顺序与训练时一致
            if self.feature_columns is not None:
                # 只保留训练时使用的特征列（排除标识列和标签列）
                feature_cols = [col for col in self.feature_columns 
                              if col not in exclude_cols_test]
                X = df[feature_cols].copy()
                
                # 如果缺少某些特征列，用0填充
                missing_cols = set(self.feature_columns) - set(exclude_cols_test) - set(feature_cols)
                if missing_cols:
                    print(f"警告: 测试数据缺少特征列: {len(missing_cols)} 个")
                    for col in missing_cols:
                        X[col] = 0
                
                # 确保列顺序一致（只保留特征列）
                feature_cols_ordered = [col for col in self.feature_columns 
                                       if col not in exclude_cols_test]
                X = X[feature_cols_ordered]
            else:
                # 如果还没有设置，使用所有数值列
                feature_cols = [col for col in df.columns 
                              if col not in exclude_cols_test and
                              df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
                X = df[feature_cols].copy()
            
            print(f"测试数据特征数: {X.shape[1]}")
            return X, None
    
    def split_data(self, X, y, test_size=0.2):
        """
        划分训练集和本地测试集
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            特征矩阵
        y : Series or ndarray
            标签向量
        test_size : float
            测试集比例
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : 
            划分后的数据
        """
        print(f"正在划分数据，测试集比例: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y  # 保持类别分布
        )
        print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        print(f"训练集类别分布:\n{pd.Series(y_train).value_counts().sort_index()}")
        print(f"测试集类别分布:\n{pd.Series(y_test).value_counts().sort_index()}")
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
        标准化特征（可选添加多项式特征以提升准确率）
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            特征矩阵
        is_train : bool
            是否为训练数据（训练数据需要fit，测试数据只需要transform）
        
        Returns:
        --------
        X_scaled : ndarray
            标准化后的特征矩阵（可能包含多项式特征）
        """
        # 标准化
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # 添加多项式特征（仅交互项，避免特征爆炸）
        if self.use_poly_features:
            if is_train:
                # 创建多项式特征生成器（仅交互项，degree=2）
                self.poly_features = PolynomialFeatures(
                    degree=2, 
                    interaction_only=True,  # 仅交互项，不包括平方项
                    include_bias=False
                )
                X_scaled = self.poly_features.fit_transform(X_scaled)
                print(f"多项式特征工程：{X_scaled.shape[1] - X.shape[1]} 个新特征")
            else:
                if self.poly_features is not None:
                    X_scaled = self.poly_features.transform(X_scaled)
        
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
            'feature_columns': self.feature_columns,
            'poly_features': self.poly_features,
            'use_poly_features': self.use_poly_features
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
            self.poly_features = preprocessor.get('poly_features', None)
            self.use_poly_features = preprocessor.get('use_poly_features', True)
            print(f"预处理对象已加载: {filepath}")
        else:
            print(f"警告: 预处理对象文件不存在: {filepath}")
    
    def get_valid_columns(self, df):
        """
        智能特征选择：筛选有用的特征列，移除无用列
        
        筛选策略：
        1. 移除低方差特征（几乎不变的特征）
        2. 移除高缺失值特征（缺失率>50%）
        3. 移除常数特征（方差为0）
        4. 移除明显无效的特征（如包含-999999999的特殊值）
        5. 保留与目标相关的特征
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        
        Returns:
        --------
        valid_columns : list
            有效列名列表
        """
        # 基础标识列（必须保留，但不作为特征）
        # 注意：只保留实际存在于DataFrame中的列
        potential_base_cols = ['time', 'group_name', 'sufficient_window_size', 'labelArea', 'labelMovement', 'result']
        base_cols = [col for col in potential_base_cols if col in df.columns]
        
        print("=" * 60)
        print("智能特征选择：筛选有用的特征列")
        print("=" * 60)
        
        all_cols = df.columns.tolist()
        print(f"原始总列数: {len(all_cols)}")
        print(f"基础标识列: {base_cols}")
        
        # 1. 排除基础标识列
        feature_candidates = [col for col in all_cols if col not in base_cols]
        print(f"排除标识列后: {len(feature_candidates)} 个候选特征")
        
        # 2. 选择数值列
        numeric_cols = [col for col in feature_candidates 
                       if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        print(f"数值特征列: {len(numeric_cols)} 个")
        
        # 3. 移除常数特征（方差为0或接近0）
        valid_features = []
        constant_features = []
        low_variance_features = []
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            # 计算方差
            variance = df[col].var()
            
            # 检查是否为常数
            if variance == 0 or (variance < 1e-10):
                constant_features.append(col)
                continue
            
            # 检查缺失值比例
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.5:  # 缺失率超过50%
                continue
            
            # 检查是否包含特殊无效值（如-999999999）
            if df[col].dtype in [np.float64, np.float32]:
                invalid_count = (df[col] == -999999999.0).sum()
                if invalid_count / len(df) > 0.5:  # 超过50%是无效值
                    continue
            
            # 检查方差是否过低（使用相对方差，避免因量纲问题误删）
            mean_abs = np.abs(df[col].mean())
            if mean_abs > 1e-6:  # 避免除零
                cv = np.sqrt(variance) / mean_abs  # 变异系数
                if cv < 0.001:  # 变异系数过小，说明几乎不变
                    low_variance_features.append(col)
                    continue
            
            valid_features.append(col)
        
        print(f"移除常数特征: {len(constant_features)} 个")
        print(f"移除低方差特征: {len(low_variance_features)} 个")
        print(f"保留有效特征: {len(valid_features)} 个")
        
        # 4. 移除高度相关的特征（可选，避免冗余）
        # 这里暂时不实现，因为计算量大，可以在后续步骤中通过QR分解处理
        
        # 5. 合并基础列和有效特征列
        valid_columns = base_cols + valid_features
        
        print(f"\n最终有效列数: {len(valid_columns)} (基础列: {len(base_cols)}, 特征列: {len(valid_features)})")
        print(f"特征减少率: {(len(all_cols) - len(valid_columns)) / len(all_cols) * 100:.1f}%")
        
        return valid_columns
    
    def clean_and_split_data(self, train_ratio=0.8, remove_outliers_flag=True, 
                            fill_missing_flag=True, random_state=None):
        """
        清洗数据并划分训练集和测试集
        只保留有效列，随机选取80%作为训练数据，20%作为测试数据
        
        Parameters:
        -----------
        train_ratio : float
            训练集比例（默认0.8）
        remove_outliers_flag : bool
            是否移除异常值
        fill_missing_flag : bool
            是否填充缺失值
        random_state : int or None
            随机种子
        
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
        
        # 2. 获取有效列（智能特征选择）
        valid_columns = self.get_valid_columns(df)
        df = df[valid_columns].copy()
        
        # 保存有效列信息（用于后续特征准备）
        self.feature_columns = [col for col in valid_columns 
                               if col not in ['time', 'group_name', 'sufficient_window_size', 
                                            'labelArea', 'labelMovement', 'result']]
        
        print(f"保留有效列后数据形状: {df.shape}")
        
        # 3. 数据清洗（使用更精细的策略以提高准确率）
        if remove_outliers_flag:
            df = self.remove_outliers(df, threshold=2.5, method='std')
        
        if fill_missing_flag:
            df = self.fill_missing_values(df, strategy='interpolate', use_forward_fill=True)
        
        # 4. 随机划分训练集和测试集
        print(f"正在随机划分数据，训练集比例: {train_ratio}")
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n_train = int(len(df_shuffled) * train_ratio)
        
        train_df = df_shuffled.iloc[:n_train].copy()
        test_df = df_shuffled.iloc[n_train:].copy()
        
        print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")
        
        # 5. 在训练数据中，在sufficient_window_size与labelArea之间插入result列（值为空）
        # 找到sufficient_window_size和labelArea的位置
        cols = train_df.columns.tolist()
        sufficient_idx = cols.index('sufficient_window_size')
        label_idx = cols.index('labelArea')
        
        # 在labelArea之前插入result列
        new_cols = cols[:label_idx] + ['result'] + cols[label_idx:]
        train_df = train_df.reindex(columns=new_cols)
        train_df['result'] = ''  # 设置为空字符串
        
        # 6. 保存清洗后的训练数据
        train_output_path = os.path.join(os.path.dirname(self.train_path), '清洗训练数据.csv')
        train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后的训练数据已保存: {train_output_path}")
        
        # 7. 保存测试数据（不包含labelArea列，避免标签泄露）
        test_output_path = os.path.join(os.path.dirname(self.train_path), '清洗测试数据.csv')
        # 测试数据需要添加result列（用于存放预测结果），但不包含labelArea列
        test_cols = [col for col in test_df.columns if col != 'labelArea']
        test_df_output = test_df[test_cols].copy()
        
        # 在sufficient_window_size之后插入result列
        test_cols_list = test_df_output.columns.tolist()
        if 'sufficient_window_size' in test_cols_list:
            sufficient_idx_test = test_cols_list.index('sufficient_window_size')
            test_cols_list.insert(sufficient_idx_test + 1, 'result')
        else:
            test_cols_list.append('result')
        test_df_output = test_df_output.reindex(columns=test_cols_list)
        test_df_output['result'] = ''  # 设置为空字符串
        
        test_df_output.to_csv(test_output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后的测试数据已保存: {test_output_path}")
        print(f"测试数据不包含labelArea列（避免标签泄露）")
        
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
        
        # 3. 准备特征和标签
        X, y = self.prepare_features(df, is_train=True)
        
        # 4. 数据划分
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
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
        random_state=42
    )
    
    print("\n数据清洗完成！")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"训练集列名: {train_df.columns.tolist()[:10]}...")
    print(f"测试集列名: {test_df.columns.tolist()[:10]}...")


if __name__ == '__main__':
    main()

