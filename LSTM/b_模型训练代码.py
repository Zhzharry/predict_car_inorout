"""
LSTM模型训练模块
功能：时序序列构造、LSTM网络构建、模型训练、模型保存
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from a_数据清洗代码 import DataCleaner


def convert_to_python_type(obj):
    """
    递归将 numpy 类型转换为 Python 原生类型（用于 JSON 序列化）
    
    Parameters:
    -----------
    obj : any
        需要转换的对象
    
    Returns:
    --------
    converted : any
        转换后的对象
    """
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


class LSTMTrainer:
    """LSTM模型训练类"""
    
    def __init__(self, model_dir='model/', random_state=42):
        self.model_dir = model_dir
        self.random_state = random_state
        os.makedirs(model_dir, exist_ok=True)
        np.random.seed(random_state)
        
        # 模型参数
        self.model = None
        self.sequence_length = 20  # 时间步长（滑动窗口大小）
        self.lstm_units = 64  # LSTM单元数量
        self.dropout_rate = 0.3  # Dropout率
        self.learning_rate = 0.001  # 学习率
        self.batch_size = 32  # 批次大小
        self.epochs = 50  # 训练轮数
        
        # 数据信息
        self.n_features = None
        self.n_classes = None
        self.classes = None
        self.class_weights = None
        
    def create_sequences(self, X, y, sequence_length=None):
        """
        将数据转换为时序序列
        
        Parameters:
        -----------
        X : ndarray
            特征矩阵 (n_samples, n_features)
        y : ndarray
            标签数组 (n_samples,)
        sequence_length : int
            时间步长（滑动窗口大小）
        
        Returns:
        --------
        X_sequences : ndarray
            时序序列 (n_sequences, sequence_length, n_features)
        y_sequences : ndarray
            序列标签 (n_sequences,)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        print(f"\n正在构造时序序列（窗口大小: {sequence_length}）...")
        
        n_samples = len(X)
        n_features = X.shape[1]
        
        # 计算可以生成的序列数量
        n_sequences = n_samples - sequence_length + 1
        
        if n_sequences <= 0:
            raise ValueError(f"样本数 ({n_samples}) 小于序列长度 ({sequence_length})，无法构造序列")
        
        X_sequences = np.zeros((n_sequences, sequence_length, n_features))
        y_sequences = np.zeros(n_sequences, dtype=int)
        
        # 确保 y 是 numpy 数组（避免 pandas Series 索引问题）
        if hasattr(y, 'values'):
            y = y.values
        y = np.asarray(y)
        
        for i in range(n_sequences):
            # 提取序列窗口
            X_sequences[i] = X[i:i+sequence_length]
            # 使用窗口中心样本的标签（或窗口内占比最高的类别）
            window_labels = y[i:i+sequence_length]
            # 使用窗口中心标签（现在 window_labels 是 numpy 数组，可以用整数索引）
            y_sequences[i] = window_labels[sequence_length // 2]
        
        print(f"原始样本数: {n_samples}")
        print(f"生成的序列数: {n_sequences}")
        print(f"序列形状: {X_sequences.shape}")
        print(f"标签分布:\n{pd.Series(y_sequences).value_counts().sort_index()}")
        
        return X_sequences, y_sequences
    
    def build_model(self, n_features, n_classes):
        """
        构建LSTM模型
        
        Parameters:
        -----------
        n_features : int
            特征维度
        n_classes : int
            类别数
        
        Returns:
        --------
        model : keras.Model
            LSTM模型
        """
        print("\n" + "=" * 60)
        print("构建LSTM模型")
        print("=" * 60)
        
        model = Sequential([
            # 第一层LSTM
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, n_features)),
            Dropout(self.dropout_rate),
            
            # 第二层LSTM（可选）
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            
            # 全连接层
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            
            # 输出层
            Dense(n_classes, activation='softmax')
        ])
        
        # 编译模型
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"模型结构:")
        model.summary()
        
        # 计算参数量
        total_params = model.count_params()
        print(f"\n总参数量: {total_params:,}")
        
        return model
    
    def compute_class_weights(self, y_train):
        """
        计算类别权重（处理类别不平衡）
        
        Parameters:
        -----------
        y_train : ndarray
            训练标签
        
        Returns:
        --------
        class_weights : dict
            类别权重字典
        """
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)
        
        # 计算类别权重
        class_weights_array = compute_class_weight(
            'balanced',
            classes=self.classes,
            y=y_train
        )
        
        self.class_weights = dict(zip(self.classes, class_weights_array))
        
        print(f"\n类别权重:")
        for cls, weight in self.class_weights.items():
            count = np.sum(y_train == cls)
            print(f"  类别 {cls}: {weight:.4f} (样本数: {count})")
        
        return self.class_weights
    
    def train_model(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
        """
        训练LSTM模型
        
        Parameters:
        -----------
        X_train_seq : ndarray
            训练序列 (n_sequences, sequence_length, n_features)
        y_train_seq : ndarray
            训练标签 (n_sequences,)
        X_val_seq : ndarray
            验证序列
        y_val_seq : ndarray
            验证标签
        
        Returns:
        --------
        history : dict
            训练历史
        """
        print("\n" + "=" * 60)
        print("训练LSTM模型")
        print("=" * 60)
        
        # 计算类别权重
        self.compute_class_weights(y_train_seq)
        
        # 构建模型
        self.n_features = X_train_seq.shape[2]
        self.model = self.build_model(self.n_features, self.n_classes)
        
        # 设置回调函数
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, 'lstm_best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # 训练模型
        print(f"\n开始训练...")
        print(f"训练序列数: {len(X_train_seq)}")
        print(f"验证序列数: {len(X_val_seq)}")
        print(f"批次大小: {self.batch_size}")
        print(f"最大轮数: {self.epochs}")
        
        start_time = time.time()
        
        history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val_seq, y_val_seq),
            class_weight=self.class_weights,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n训练完成，耗时: {training_time:.2f} 秒")
        
        return history
    
    def evaluate_model(self, X_test_seq, y_test_seq):
        """
        评估模型
        
        Parameters:
        -----------
        X_test_seq : ndarray
            测试序列
        y_test_seq : ndarray
            测试标签
        
        Returns:
        --------
        results : dict
            评估结果
        """
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        # 预测
        y_pred_proba = self.model.predict(X_test_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算准确率
        accuracy = accuracy_score(y_test_seq, y_pred)
        print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 各类别准确率
        print("\n各类别准确率:")
        for cls in self.classes:
            cls_mask = y_test_seq == cls
            if np.any(cls_mask):
                cls_accuracy = accuracy_score(y_test_seq[cls_mask], y_pred[cls_mask])
                print(f"  类别 {cls}: {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test_seq, y_pred, labels=self.classes)
        print("\n混淆矩阵:")
        print(cm)
        
        # 分类报告
        class_names = {0: '其他场景', 1: '进入地库', 2: '出地库'}
        target_names = [class_names.get(cls, f'类别{cls}') for cls in self.classes]
        print("\n分类报告:")
        print(classification_report(y_test_seq, y_pred, labels=self.classes, 
                                  target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_true': y_test_seq
        }
    
    def save_model(self, filename='lstm_model.h5'):
        """
        保存模型
        
        Parameters:
        -----------
        filename : str
            模型文件名
        """
        filepath = os.path.join(self.model_dir, filename)
        
        # 保存Keras模型
        self.model.save(filepath)
        
        print(f"\n模型已保存到: {filepath}")
        model_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"模型文件大小: {model_size:.2f} MB")
        
        # 保存模型配置
        config = {
            'model_type': 'LSTM',
            'sequence_length': int(self.sequence_length),
            'lstm_units': int(self.lstm_units),
            'dropout_rate': float(self.dropout_rate),
            'learning_rate': float(self.learning_rate),
            'batch_size': int(self.batch_size),
            'n_features': int(self.n_features) if self.n_features is not None else None,
            'n_classes': int(self.n_classes) if self.n_classes is not None else None,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'class_weights': self.class_weights,
            'total_params': int(self.model.count_params()) if self.model is not None else None,
            'training_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 转换所有 numpy 类型为 Python 原生类型
        config = convert_to_python_type(config)
        
        config_path = os.path.join(self.model_dir, 'lstm_model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"模型配置已保存到: {config_path}")
    
    def load_model(self, filename='lstm_model.h5'):
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
        
        # 加载Keras模型
        self.model = keras.models.load_model(filepath)
        
        # 加载配置
        config_path = os.path.join(self.model_dir, 'lstm_model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.sequence_length = config.get('sequence_length', 20)
            self.lstm_units = config.get('lstm_units', 64)
            self.dropout_rate = config.get('dropout_rate', 0.3)
            self.learning_rate = config.get('learning_rate', 0.001)
            self.n_features = config.get('n_features')
            self.n_classes = config.get('n_classes')
            self.classes = np.array(config.get('classes')) if config.get('classes') else None
            self.class_weights = config.get('class_weights')
        
        print(f"模型已加载: {filepath}")
        if self.classes is not None:
            print(f"类别数: {len(self.classes)}")


def main():
    """主函数：完整的模型训练流程"""
    print("=" * 60)
    print("LSTM模型训练流程")
    print("=" * 60)
    
    # 1. 数据清洗和预处理
    cleaner = DataCleaner(
        train_path='train/训练数据.csv',
        test_dir='test/',
        output_dir='model/'
    )
    
    X_train, X_test, y_train, y_test = cleaner.process_train_data(
        remove_outliers_flag=True,
        fill_missing_flag=True,
        standardize=True,
        test_size=0.3  # 7:3比例
    )
    
    # 2. 创建训练器
    trainer = LSTMTrainer(model_dir='model/', random_state=42)
    
    # 3. 构造时序序列
    X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = trainer.create_sequences(X_test, y_test)
    
    # 4. 从训练序列中划分验证集
    from sklearn.model_selection import train_test_split
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_train_seq, y_train_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_train_seq
    )
    
    # 5. 训练模型
    history = trainer.train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    # 6. 评估模型
    results = trainer.evaluate_model(X_test_seq, y_test_seq)
    
    # 7. 保存模型
    trainer.save_model('lstm_model.h5')
    
    # 8. 对清洗训练数据进行预测并回填结果
    print("\n" + "=" * 60)
    print("对清洗训练数据进行预测并回填结果")
    print("=" * 60)
    cleaned_train_path = 'train/清洗训练数据.csv'
    if os.path.exists(cleaned_train_path):
        df_train_cleaned = pd.read_csv(cleaned_train_path)
        print(f"加载清洗训练数据: {df_train_cleaned.shape}")
        
        # 准备特征
        exclude_cols = ['time', 'group_name', 'sufficient_window_size',
                        'labelMovement', 'result', 'labelArea']
        feature_cols = [col for col in df_train_cleaned.columns if col not in exclude_cols]
        
        if cleaner.feature_columns is not None:
            feature_cols = [col for col in cleaner.feature_columns if col in feature_cols]
            X_train_for_predict = df_train_cleaned[feature_cols].copy()
            missing_cols = set(cleaner.feature_columns) - set(feature_cols)
            if missing_cols:
                print(f"警告: 训练数据缺少特征列: {len(missing_cols)} 个")
                for col in missing_cols:
                    X_train_for_predict[col] = 0
            X_train_for_predict = X_train_for_predict[cleaner.feature_columns]
        else:
            X_train_for_predict = df_train_cleaned[feature_cols].copy()
        
        # 标准化
        X_train_scaled = cleaner.transform_features(X_train_for_predict.values, is_train=False)
        
        # 构造序列
        X_train_seq_for_predict, _ = trainer.create_sequences(X_train_scaled, 
                                                              np.zeros(len(X_train_scaled)))
        
        # 预测
        train_predictions_proba = trainer.model.predict(X_train_seq_for_predict, verbose=0)
        train_predictions = np.argmax(train_predictions_proba, axis=1)
        
        # 将预测结果映射回原始样本（使用序列中心标签）
        # 对于前sequence_length-1个样本，使用第一个序列的预测
        # 对于后续样本，使用对应序列的预测
        final_predictions = np.zeros(len(df_train_cleaned), dtype=int)
        seq_len = trainer.sequence_length
        
        # 前seq_len-1个样本使用第一个序列的预测
        final_predictions[:seq_len-1] = train_predictions[0]
        
        # 后续样本使用对应序列的预测
        for i in range(seq_len-1, len(df_train_cleaned)):
            seq_idx = i - (seq_len - 1)
            if seq_idx < len(train_predictions):
                final_predictions[i] = train_predictions[seq_idx]
            else:
                final_predictions[i] = train_predictions[-1]
        
        # 回填结果
        df_train_cleaned['result'] = final_predictions
        
        # 保存
        df_train_cleaned.to_csv(cleaned_train_path, index=False)
        print(f"清洗训练数据预测结果已回填并保存到: {cleaned_train_path}")
    
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

