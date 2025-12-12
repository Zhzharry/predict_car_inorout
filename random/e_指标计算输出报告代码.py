"""
指标计算和报告输出模块
功能：计算CSV文件中标签列和result列的各类评估指标，并输出详细的txt报告
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, hamming_loss, jaccard_score
)
import os
from datetime import datetime


class MetricsCalculator:
    """指标计算类"""
    
    def __init__(self, csv_path, label_column='labelArea', result_column='result', 
                 output_dir='results/'):
        """
        初始化指标计算器
        
        Parameters:
        -----------
        csv_path : str
            CSV文件路径
        label_column : str
            真实标签列名
        result_column : str
            预测结果列名
        output_dir : str
            报告输出目录
        """
        self.csv_path = csv_path
        self.label_column = label_column
        self.result_column = result_column
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据
        self.df = None
        self.y_true = None
        self.y_pred = None
        
        # 指标结果
        self.metrics = {}
        
    def load_data(self):
        """加载数据并提取标签和预测结果"""
        print("=" * 60)
        print("加载数据")
        print("=" * 60)
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"文件不存在: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")
        
        # 检查必要的列是否存在
        if self.label_column not in self.df.columns:
            raise ValueError(f"标签列 '{self.label_column}' 不存在")
        if self.result_column not in self.df.columns:
            raise ValueError(f"结果列 '{self.result_column}' 不存在")
        
        # 提取标签和预测结果
        self.y_true = self.df[self.label_column].copy()
        self.y_pred = self.df[self.result_column].copy()
        
        # 处理缺失值和空值（包括NaN、空字符串、空格等）
        # 将空字符串、空格等转换为NaN
        if self.y_true.dtype == 'object':
            self.y_true = self.y_true.replace(['', ' ', 'nan', 'NaN', 'None', np.nan], np.nan)
        else:
            # 如果是数值类型，只处理NaN
            self.y_true = self.y_true.replace([np.nan], np.nan)
        
        if self.y_pred.dtype == 'object':
            self.y_pred = self.y_pred.replace(['', ' ', 'nan', 'NaN', 'None', np.nan], np.nan)
        else:
            # 如果是数值类型，只处理NaN
            self.y_pred = self.y_pred.replace([np.nan], np.nan)
        
        # 过滤缺失值
        valid_mask = ~(self.y_true.isna() | self.y_pred.isna())
        self.y_true = self.y_true[valid_mask]
        self.y_pred = self.y_pred[valid_mask]
        
        # 检查是否有有效数据
        if len(self.y_true) == 0:
            # 统计原始数据情况
            original_true = self.df[self.label_column]
            original_pred = self.df[self.result_column]
            
            true_nan = original_true.isna().sum()
            pred_nan = original_pred.isna().sum()
            
            # 检查空字符串（如果是object类型）
            if original_true.dtype == 'object':
                true_empty = (original_true == '').sum()
            else:
                true_empty = 0
            
            if original_pred.dtype == 'object':
                pred_empty = (original_pred == '').sum()
            else:
                pred_empty = 0
            
            raise ValueError(
                f"错误：过滤后没有有效样本！\n"
                f"原始数据: {len(self.df)} 行\n"
                f"标签列 '{self.label_column}':\n"
                f"  - 缺失值(NaN): {true_nan} 个\n"
                f"  - 空字符串: {true_empty} 个\n"
                f"  - 数据类型: {original_true.dtype}\n"
                f"结果列 '{self.result_column}':\n"
                f"  - 缺失值(NaN): {pred_nan} 个\n"
                f"  - 空字符串: {pred_empty} 个\n"
                f"  - 数据类型: {original_pred.dtype}\n"
                f"\n提示：如果result列为空，请先运行模型训练代码(b_模型训练代码.py)来填充预测结果。"
            )
        
        print(f"有效样本数: {len(self.y_true)}")
        print(f"真实标签分布:\n{self.y_true.value_counts().sort_index()}")
        print(f"预测结果分布:\n{self.y_pred.value_counts().sort_index()}")
        
        # 转换为数值类型（处理可能的字符串类型）
        try:
            self.y_true = pd.to_numeric(self.y_true, errors='coerce').astype(int)
            self.y_pred = pd.to_numeric(self.y_pred, errors='coerce').astype(int)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"无法将标签或结果转换为整数类型。\n"
                f"标签列唯一值: {self.y_true.unique()[:10]}\n"
                f"结果列唯一值: {self.y_pred.unique()[:10]}\n"
                f"错误: {e}"
            )
        
        return self.y_true, self.y_pred
    
    def calculate_all_metrics(self):
        """计算所有评估指标"""
        print("\n" + "=" * 60)
        print("计算评估指标")
        print("=" * 60)
        
        if self.y_true is None or self.y_pred is None:
            raise ValueError("请先加载数据")
        
        # 检查数据是否为空
        if len(self.y_true) == 0 or len(self.y_pred) == 0:
            raise ValueError("数据为空，无法计算指标。请检查数据文件。")
        
        # 获取类别标签
        classes = sorted(set(self.y_true.unique()) | set(self.y_pred.unique()))
        
        # 检查是否有有效的类别
        if len(classes) == 0:
            raise ValueError("没有找到有效的类别标签。请检查数据。")
        
        class_names = {0: '其他场景', 1: '进入地库', 2: '出地库'}
        
        # 1. 基础指标
        accuracy = accuracy_score(self.y_true, self.y_pred)
        self.metrics['accuracy'] = accuracy
        
        # 2. 精确率、召回率、F1分数（每个类别和平均）
        precision_per_class = precision_score(self.y_true, self.y_pred, 
                                            labels=classes, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, 
                                       labels=classes, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, 
                               labels=classes, average=None, zero_division=0)
        
        precision_macro = precision_score(self.y_true, self.y_pred, 
                                        labels=classes, average='macro', zero_division=0)
        recall_macro = recall_score(self.y_true, self.y_pred, 
                                   labels=classes, average='macro', zero_division=0)
        f1_macro = f1_score(self.y_true, self.y_pred, 
                          labels=classes, average='macro', zero_division=0)
        
        precision_weighted = precision_score(self.y_true, self.y_pred, 
                                           labels=classes, average='weighted', zero_division=0)
        recall_weighted = recall_score(self.y_true, self.y_pred, 
                                      labels=classes, average='weighted', zero_division=0)
        f1_weighted = f1_score(self.y_true, self.y_pred, 
                             labels=classes, average='weighted', zero_division=0)
        
        precision_micro = precision_score(self.y_true, self.y_pred, 
                                        labels=classes, average='micro', zero_division=0)
        recall_micro = recall_score(self.y_true, self.y_pred, 
                                   labels=classes, average='micro', zero_division=0)
        f1_micro = f1_score(self.y_true, self.y_pred, 
                          labels=classes, average='micro', zero_division=0)
        
        self.metrics['precision_per_class'] = dict(zip(classes, precision_per_class))
        self.metrics['recall_per_class'] = dict(zip(classes, recall_per_class))
        self.metrics['f1_per_class'] = dict(zip(classes, f1_per_class))
        self.metrics['precision_macro'] = precision_macro
        self.metrics['recall_macro'] = recall_macro
        self.metrics['f1_macro'] = f1_macro
        self.metrics['precision_weighted'] = precision_weighted
        self.metrics['recall_weighted'] = recall_weighted
        self.metrics['f1_weighted'] = f1_weighted
        self.metrics['precision_micro'] = precision_micro
        self.metrics['recall_micro'] = recall_micro
        self.metrics['f1_micro'] = f1_micro
        
        # 3. 混淆矩阵
        cm = confusion_matrix(self.y_true, self.y_pred, labels=classes)
        self.metrics['confusion_matrix'] = cm
        self.metrics['classes'] = classes
        
        # 4. 支持数（每个类别的真实样本数）
        support = {}
        for cls in classes:
            support[cls] = int(np.sum(self.y_true == cls))
        self.metrics['support'] = support
        
        # 5. 其他指标
        kappa = cohen_kappa_score(self.y_true, self.y_pred)
        mcc = matthews_corrcoef(self.y_true, self.y_pred)
        hamming = hamming_loss(self.y_true, self.y_pred)
        
        # Jaccard分数（每个类别）
        jaccard_per_class = jaccard_score(self.y_true, self.y_pred, 
                                         labels=classes, average=None, zero_division=0)
        jaccard_macro = jaccard_score(self.y_true, self.y_pred, 
                                     labels=classes, average='macro', zero_division=0)
        jaccard_weighted = jaccard_score(self.y_true, self.y_pred, 
                                        labels=classes, average='weighted', zero_division=0)
        
        self.metrics['kappa'] = kappa
        self.metrics['mcc'] = mcc
        self.metrics['hamming_loss'] = hamming
        self.metrics['jaccard_per_class'] = dict(zip(classes, jaccard_per_class))
        self.metrics['jaccard_macro'] = jaccard_macro
        self.metrics['jaccard_weighted'] = jaccard_weighted
        
        # 6. 错误分析
        error_rate_per_class = {}
        for cls in classes:
            mask = self.y_true == cls
            if np.sum(mask) > 0:
                error_rate = 1 - recall_per_class[classes.index(cls)]
                error_rate_per_class[cls] = error_rate
            else:
                error_rate_per_class[cls] = 0.0
        self.metrics['error_rate_per_class'] = error_rate_per_class
        
        # 7. 分类报告（sklearn格式）
        self.metrics['classification_report'] = classification_report(
            self.y_true, self.y_pred, 
            labels=classes,
            target_names=[class_names.get(cls, f'类别{cls}') for cls in classes],
            output_dict=True,
            zero_division=0
        )
        
        print("指标计算完成")
        return self.metrics
    
    def generate_report(self, output_filename=None):
        """生成详细的txt报告"""
        if not self.metrics:
            raise ValueError("请先计算指标")
        
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.basename(self.csv_path).replace('.csv', '')
            output_filename = f"{filename}_指标报告_{timestamp}.txt"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        print(f"\n生成报告: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 报告头部
            f.write("=" * 80 + "\n")
            f.write("模型评估指标报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据文件: {self.csv_path}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总样本数: {len(self.y_true)}\n")
            f.write(f"标签列: {self.label_column}\n")
            f.write(f"结果列: {self.result_column}\n\n")
            
            # 1. 数据概览
            f.write("=" * 80 + "\n")
            f.write("1. 数据概览\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("真实标签分布:\n")
            true_dist = self.y_true.value_counts().sort_index()
            for cls, count in true_dist.items():
                percentage = count / len(self.y_true) * 100
                f.write(f"  类别 {cls}: {count} ({percentage:.2f}%)\n")
            
            f.write("\n预测结果分布:\n")
            pred_dist = self.y_pred.value_counts().sort_index()
            for cls, count in pred_dist.items():
                percentage = count / len(self.y_pred) * 100
                f.write(f"  类别 {cls}: {count} ({percentage:.2f}%)\n")
            
            f.write("\n")
            
            # 2. 总体指标
            f.write("=" * 80 + "\n")
            f.write("2. 总体指标\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"准确率 (Accuracy):           {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Cohen's Kappa:               {self.metrics['kappa']:.4f}\n")
            f.write(f"Matthews相关系数 (MCC):      {self.metrics['mcc']:.4f}\n")
            f.write(f"Hamming损失:                 {self.metrics['hamming_loss']:.4f}\n")
            f.write("\n")
            
            # 3. 宏平均指标
            f.write("=" * 80 + "\n")
            f.write("3. 宏平均指标 (Macro Average)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"精确率 (Precision):          {self.metrics['precision_macro']:.4f}\n")
            f.write(f"召回率 (Recall):             {self.metrics['recall_macro']:.4f}\n")
            f.write(f"F1分数 (F1-Score):           {self.metrics['f1_macro']:.4f}\n")
            f.write(f"Jaccard分数:                 {self.metrics['jaccard_macro']:.4f}\n")
            f.write("\n")
            
            # 4. 加权平均指标
            f.write("=" * 80 + "\n")
            f.write("4. 加权平均指标 (Weighted Average)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"精确率 (Precision):          {self.metrics['precision_weighted']:.4f}\n")
            f.write(f"召回率 (Recall):             {self.metrics['recall_weighted']:.4f}\n")
            f.write(f"F1分数 (F1-Score):           {self.metrics['f1_weighted']:.4f}\n")
            f.write(f"Jaccard分数:                 {self.metrics['jaccard_weighted']:.4f}\n")
            f.write("\n")
            
            # 5. 微平均指标
            f.write("=" * 80 + "\n")
            f.write("5. 微平均指标 (Micro Average)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"精确率 (Precision):          {self.metrics['precision_micro']:.4f}\n")
            f.write(f"召回率 (Recall):             {self.metrics['recall_micro']:.4f}\n")
            f.write(f"F1分数 (F1-Score):           {self.metrics['f1_micro']:.4f}\n")
            f.write("\n")
            
            # 6. 每个类别的详细指标
            f.write("=" * 80 + "\n")
            f.write("6. 每个类别的详细指标\n")
            f.write("=" * 80 + "\n\n")
            
            class_names = {0: '其他场景', 1: '进入地库', 2: '出地库'}
            
            f.write(f"{'类别':<10} {'类别名':<12} {'精确率':<12} {'召回率':<12} "
                   f"{'F1分数':<12} {'支持数':<10} {'错误率':<12} {'Jaccard':<12}\n")
            f.write("-" * 80 + "\n")
            
            for cls in self.metrics['classes']:
                cls_name = class_names.get(cls, f'类别{cls}')
                precision = self.metrics['precision_per_class'][cls]
                recall = self.metrics['recall_per_class'][cls]
                f1 = self.metrics['f1_per_class'][cls]
                support = self.metrics['support'][cls]
                error_rate = self.metrics['error_rate_per_class'][cls]
                jaccard = self.metrics['jaccard_per_class'][cls]
                
                f.write(f"{cls:<10} {cls_name:<12} {precision:<12.4f} {recall:<12.4f} "
                       f"{f1:<12.4f} {support:<10} {error_rate:<12.4f} {jaccard:<12.4f}\n")
            
            f.write("\n")
            
            # 7. 混淆矩阵
            f.write("=" * 80 + "\n")
            f.write("7. 混淆矩阵 (Confusion Matrix)\n")
            f.write("=" * 80 + "\n\n")
            
            cm = self.metrics['confusion_matrix']
            classes = self.metrics['classes']
            
            # 表头
            header = '真实\\预测'
            f.write(f"{header:<12}")
            for cls in classes:
                cls_name = class_names.get(cls, f'类别{cls}')
                f.write(f"{cls_name:<12}")
            f.write("总计\n")
            f.write("-" * 80 + "\n")
            
            # 矩阵内容
            for i, true_cls in enumerate(classes):
                true_name = class_names.get(true_cls, f'类别{true_cls}')
                f.write(f"{true_name:<12}")
                row_sum = 0
                for j, pred_cls in enumerate(classes):
                    value = cm[i, j]
                    row_sum += value
                    f.write(f"{value:<12}")
                f.write(f"{row_sum}\n")
            
            # 列总计
            f.write("-" * 80 + "\n")
            f.write(f"{'总计':<12}")
            col_sums = cm.sum(axis=0)
            for j, pred_cls in enumerate(classes):
                f.write(f"{col_sums[j]:<12}")
            f.write(f"{cm.sum()}\n")
            f.write("\n")
            
            # 混淆矩阵百分比
            f.write("混淆矩阵 (百分比):\n")
            header = '真实\\预测'
            f.write(f"{header:<12}")
            for cls in classes:
                cls_name = class_names.get(cls, f'类别{cls}')
                f.write(f"{cls_name:<12}")
            f.write("\n")
            f.write("-" * 80 + "\n")
            
            for i, true_cls in enumerate(classes):
                true_name = class_names.get(true_cls, f'类别{true_cls}')
                f.write(f"{true_name:<12}")
                row_sum = cm[i, :].sum()
                if row_sum > 0:
                    for j, pred_cls in enumerate(classes):
                        value = cm[i, j]
                        percentage = value / row_sum * 100
                        f.write(f"{percentage:>6.2f}%     ")
                else:
                    for j in range(len(classes)):
                        f.write(f"{'0.00%':<12}")
                f.write("\n")
            
            f.write("\n")
            
            # 8. 分类报告（sklearn格式）
            f.write("=" * 80 + "\n")
            f.write("8. 详细分类报告 (Classification Report)\n")
            f.write("=" * 80 + "\n\n")
            
            report = self.metrics['classification_report']
            
            # 每个类别的报告
            f.write(f"{'类别':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'支持数':<10}\n")
            f.write("-" * 60 + "\n")
            
            for cls in classes:
                cls_name = class_names.get(cls, f'类别{cls}')
                if str(cls) in report:
                    precision = report[str(cls)]['precision']
                    recall = report[str(cls)]['recall']
                    f1 = report[str(cls)]['f1-score']
                    support = report[str(cls)]['support']
                    f.write(f"{cls_name:<12} {precision:<12.4f} {recall:<12.4f} "
                           f"{f1:<12.4f} {support:<10}\n")
            
            f.write("-" * 60 + "\n")
            f.write(f"{'宏平均':<12} {report['macro avg']['precision']:<12.4f} "
                   f"{report['macro avg']['recall']:<12.4f} "
                   f"{report['macro avg']['f1-score']:<12.4f} "
                   f"{report['macro avg']['support']:<10}\n")
            f.write(f"{'加权平均':<12} {report['weighted avg']['precision']:<12.4f} "
                   f"{report['weighted avg']['recall']:<12.4f} "
                   f"{report['weighted avg']['f1-score']:<12.4f} "
                   f"{report['weighted avg']['support']:<10}\n")
            f.write("\n")
            
            # 9. 错误分析
            f.write("=" * 80 + "\n")
            f.write("9. 错误分析\n")
            f.write("=" * 80 + "\n\n")
            
            total_errors = np.sum(self.y_true != self.y_pred)
            f.write(f"总错误数: {total_errors} / {len(self.y_true)} "
                   f"({total_errors/len(self.y_true)*100:.2f}%)\n\n")
            
            # 每个类别的错误分析
            f.write("每个类别的错误情况:\n")
            for cls in classes:
                cls_name = class_names.get(cls, f'类别{cls}')
                true_mask = self.y_true == cls
                pred_mask = self.y_pred == cls
                
                correct = np.sum(true_mask & pred_mask)
                total = np.sum(true_mask)
                errors = total - correct
                
                if total > 0:
                    error_rate = errors / total * 100
                    f.write(f"  {cls_name} (类别{cls}): {errors}/{total} 错误 "
                           f"({error_rate:.2f}%), 正确: {correct}/{total} "
                           f"({100-error_rate:.2f}%)\n")
            
            f.write("\n")
            
            # 10. 混淆矩阵详细分析
            f.write("=" * 80 + "\n")
            f.write("10. 混淆矩阵详细分析\n")
            f.write("=" * 80 + "\n\n")
            
            for i, true_cls in enumerate(classes):
                true_name = class_names.get(true_cls, f'类别{true_cls}')
                f.write(f"真实类别: {true_name} (类别{true_cls})\n")
                row_sum = cm[i, :].sum()
                if row_sum > 0:
                    for j, pred_cls in enumerate(classes):
                        pred_name = class_names.get(pred_cls, f'类别{pred_cls}')
                        value = cm[i, j]
                        percentage = value / row_sum * 100 if row_sum > 0 else 0
                        if i == j:
                            f.write(f"  -> 正确预测为 {pred_name}: {value} ({percentage:.2f}%)\n")
                        else:
                            f.write(f"  -> 错误预测为 {pred_name}: {value} ({percentage:.2f}%)\n")
                f.write("\n")
            
            # 报告尾部
            f.write("=" * 80 + "\n")
            f.write("报告结束\n")
            f.write("=" * 80 + "\n")
        
        print(f"报告已保存到: {output_path}")
        return output_path


def main():
    """主函数：演示指标计算和报告生成"""
    print("=" * 60)
    print("指标计算和报告生成")
    print("=" * 60)
    
    # 示例：计算清洗训练数据的指标
    csv_path = 'train/清洗测试数据.csv'
    
    if os.path.exists(csv_path):
        calculator = MetricsCalculator(
            csv_path=csv_path,
            label_column='labelArea',
            result_column='result',
            output_dir='results/'
        )
        
        # 加载数据
        calculator.load_data()
        
        # 计算指标
        calculator.calculate_all_metrics()
        
        # 生成报告
        calculator.generate_report()
        
        print("\n" + "=" * 60)
        print("指标计算和报告生成完成！")
        print("=" * 60)
    else:
        print(f"文件不存在: {csv_path}")
        print("\n可以修改main函数中的csv_path来指定要分析的文件")


if __name__ == '__main__':
    main()

