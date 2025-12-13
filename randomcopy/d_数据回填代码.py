"""
数据回填模块
功能：从原始训练数据中通过时间戳匹配，将真实标签回填到清洗测试数据的result列后面
"""

import pandas as pd
import numpy as np
import os


class LabelBackfiller:
    """标签回填类"""
    
    def __init__(self, original_train_path='train/训练数据.csv', 
                 cleaned_test_path='train/清洗测试数据.csv'):
        """
        初始化标签回填器
        
        Parameters:
        -----------
        original_train_path : str
            原始训练数据路径（包含真实标签）
        cleaned_test_path : str
            清洗测试数据路径（需要回填标签）
        """
        self.original_train_path = original_train_path
        self.cleaned_test_path = cleaned_test_path
    
    def load_data(self):
        """
        加载原始训练数据和清洗测试数据
        
        Returns:
        --------
        df_original : DataFrame
            原始训练数据
        df_test : DataFrame
            清洗测试数据
        """
        print("=" * 60)
        print("加载数据")
        print("=" * 60)
        
        # 加载原始训练数据
        print(f"正在加载原始训练数据: {self.original_train_path}")
        df_original = pd.read_csv(self.original_train_path)
        print(f"原始训练数据形状: {df_original.shape}")
        
        # 加载清洗测试数据
        print(f"正在加载清洗测试数据: {self.cleaned_test_path}")
        df_test = pd.read_csv(self.cleaned_test_path)
        print(f"清洗测试数据形状: {df_test.shape}")
        
        return df_original, df_test
    
    def backfill_labels(self, output_path=None):
        """
        通过时间戳匹配，将真实标签回填到清洗测试数据
        
        Parameters:
        -----------
        output_path : str or None
            输出文件路径，None则覆盖原文件
        
        Returns:
        --------
        df_test_filled : DataFrame
            回填后的测试数据
        """
        # 1. 加载数据
        df_original, df_test = self.load_data()
        
        # 2. 检查原始数据是否有labelArea列
        if 'labelArea' not in df_original.columns:
            raise ValueError("原始训练数据中缺少labelArea列")
        
        # 2.5. 诊断信息：检查时间戳范围
        print("\n" + "=" * 60)
        print("数据诊断信息")
        print("=" * 60)
        print(f"原始训练数据时间戳范围: {df_original['time'].min()} ~ {df_original['time'].max()}")
        print(f"清洗测试数据时间戳范围: {df_test['time'].min()} ~ {df_test['time'].max()}")
        
        # 检查时间戳重叠情况
        orig_times = set(df_original['time'].unique())
        test_times = set(df_test['time'].unique())
        overlap_count = len(orig_times & test_times)
        print(f"时间戳重叠数量: {overlap_count} / {len(test_times)} ({overlap_count/len(test_times)*100:.2f}%)")
        
        if overlap_count == 0:
            print("⚠️  警告: 测试数据中的时间戳与原始训练数据中的时间戳完全没有重叠！")
            print("   这可能是因为数据拆分时使用了随机抽样，导致测试数据的时间戳不在原始数据中。")
            print("   建议: 使用时间戳范围匹配或近似匹配功能。")
        
        # 3. 创建时间戳到标签的映射字典
        print("\n" + "=" * 60)
        print("创建时间戳到标签的映射")
        print("=" * 60)
        
        # 使用time列作为主键，group_name作为辅助匹配（提高匹配准确性）
        # 创建复合键 (time, group_name) -> label 的映射
        time_label_map = {}  # time -> label (如果同一个time有多个label，保留第一个)
        time_group_label_map = {}  # (time, group_name) -> label
        time_duplicate_count = 0  # 统计重复的time_key数量
        
        for idx, row in df_original.iterrows():
            time_key = row['time']
            group_name = str(row.get('group_name', '')) if pd.notna(row.get('group_name', '')) else ''
            label = row['labelArea']
            
            # 如果group_name存在，使用复合键
            if group_name:
                composite_key = (time_key, group_name)
                time_group_label_map[composite_key] = label
            
            # 同时保存time -> label的映射（用于fallback）
            # 如果同一个time_key已经存在，检查label是否一致
            if time_key in time_label_map:
                if time_label_map[time_key] != label:
                    time_duplicate_count += 1
                    # 保留第一个匹配的label（或者可以选择保留最常见的label）
            else:
                time_label_map[time_key] = label
        
        print(f"创建了 {len(time_label_map)} 个时间戳到标签的映射")
        print(f"创建了 {len(time_group_label_map)} 个 (时间戳, 分组名) 到标签的映射")
        if time_duplicate_count > 0:
            print(f"警告: 发现 {time_duplicate_count} 个时间戳对应多个不同的标签（已保留第一个）")
        
        # 4. 在result列后面新增labelArea列
        print("\n" + "=" * 60)
        print("回填标签到清洗测试数据")
        print("=" * 60)
        
        df_test_filled = df_test.copy()
        
        # 如果labelArea列已存在，先删除它
        if 'labelArea' in df_test_filled.columns:
            df_test_filled = df_test_filled.drop(columns=['labelArea'])
        
        # 找到result列的位置
        cols = df_test_filled.columns.tolist()
        if 'result' in cols:
            result_idx = cols.index('result')
            # 在result列后面插入labelArea列
            cols.insert(result_idx + 1, 'labelArea')
        else:
            # 如果没有result列，在最后添加
            cols.append('labelArea')
        
        # 重新排列列顺序，并添加空的labelArea列
        df_test_filled = df_test_filled.reindex(columns=cols)
        df_test_filled['labelArea'] = np.nan  # 初始化为NaN
        
        # 5. 通过时间戳匹配回填标签
        matched_count = 0
        unmatched_count = 0
        match_by_composite = 0  # 通过复合键匹配的数量
        match_by_time_only = 0  # 仅通过time匹配的数量
        match_by_closest = 0  # 通过最近时间戳匹配的数量
        
        # 为了提高效率，将time_label_map的keys转换为numpy数组（用于最近匹配）
        time_list = np.array(list(time_label_map.keys()))
        time_list_sorted = np.sort(time_list)
        
        for idx, row in df_test_filled.iterrows():
            time_key = row['time']
            group_name = str(row.get('group_name', '')) if pd.notna(row.get('group_name', '')) else ''
            
            label_found = None
            match_method = None
            
            # 首先尝试使用复合键 (time, group_name) 精确匹配
            if group_name:
                composite_key = (time_key, group_name)
                if composite_key in time_group_label_map:
                    label_found = time_group_label_map[composite_key]
                    match_method = 'composite'
            
            # 如果复合键匹配失败，尝试仅使用time匹配
            if label_found is None and time_key in time_label_map:
                label_found = time_label_map[time_key]
                match_method = 'time_only'
            
            # 如果精确匹配失败，尝试查找最接近的时间戳
            if label_found is None:
                closest_time = self._find_closest_time_fast(time_key, time_list_sorted, time_label_map)
                if closest_time is not None:
                    label_found = time_label_map[closest_time]
                    match_method = 'closest'
            
            # 回填标签
            if label_found is not None:
                df_test_filled.at[idx, 'labelArea'] = label_found
                matched_count += 1
                if match_method == 'composite':
                    match_by_composite += 1
                elif match_method == 'time_only':
                    match_by_time_only += 1
                elif match_method == 'closest':
                    match_by_closest += 1
            else:
                # 无法匹配，设置为NaN
                df_test_filled.at[idx, 'labelArea'] = np.nan
                unmatched_count += 1
        
        print(f"\n匹配统计:")
        print(f"  成功匹配: {matched_count} 条")
        print(f"    - 通过(时间戳, 分组名)匹配: {match_by_composite} 条")
        print(f"    - 仅通过时间戳匹配: {match_by_time_only} 条")
        print(f"    - 通过最近时间戳匹配: {match_by_closest} 条")
        print(f"  未匹配: {unmatched_count} 条")
        print(f"  匹配率: {matched_count / len(df_test_filled) * 100:.2f}%")
        
        if unmatched_count > 0:
            print(f"\n提示: 有 {unmatched_count} 条数据未能匹配，可能原因：")
            print(f"  1. 测试数据中的时间戳在原始训练数据中不存在")
            print(f"  2. 时间戳差异过大（超过容差范围）")
            print(f"  3. 数据清洗过程中时间戳发生了变化")
        
        # 统计回填后的标签分布
        if matched_count > 0:
            print(f"\n回填后的标签分布:")
            print(df_test_filled['labelArea'].value_counts().sort_index())
        
        # 6. 保存结果
        if output_path is None:
            output_path = self.cleaned_test_path
        
        df_test_filled.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n回填后的数据已保存到: {output_path}")
        
        return df_test_filled
    
    def _find_closest_time(self, target_time, time_list, tolerance_ms=1000):
        """
        查找最接近的时间戳（旧版本，效率较低）
        
        Parameters:
        -----------
        target_time : int or float
            目标时间戳
        time_list : list
            时间戳列表
        tolerance_ms : int
            容差（毫秒），超过此容差则不匹配
        
        Returns:
        --------
        closest_time : int or None
            最接近的时间戳，如果超过容差则返回None
        """
        if not time_list:
            return None
        
        time_array = np.array(time_list)
        time_diff = np.abs(time_array - target_time)
        min_diff_idx = np.argmin(time_diff)
        min_diff = time_diff[min_diff_idx]
        
        if min_diff <= tolerance_ms:
            return time_list[min_diff_idx]
        else:
            return None
    
    def _find_closest_time_fast(self, target_time, time_list_sorted, time_label_map, tolerance_ms=1000):
        """
        快速查找最接近的时间戳（使用二分查找）
        
        Parameters:
        -----------
        target_time : int or float
            目标时间戳
        time_list_sorted : ndarray
            已排序的时间戳数组
        time_label_map : dict
            时间戳到标签的映射字典
        tolerance_ms : int
            容差（毫秒），超过此容差则不匹配
        
        Returns:
        --------
        closest_time : int or None
            最接近的时间戳，如果超过容差则返回None
        """
        if len(time_list_sorted) == 0:
            return None
        
        # 使用二分查找找到最接近的时间戳
        idx = np.searchsorted(time_list_sorted, target_time)
        
        # 检查左右两个候选值
        candidates = []
        if idx > 0:
            candidates.append((time_list_sorted[idx - 1], abs(time_list_sorted[idx - 1] - target_time)))
        if idx < len(time_list_sorted):
            candidates.append((time_list_sorted[idx], abs(time_list_sorted[idx] - target_time)))
        
        if not candidates:
            return None
        
        # 选择最接近的
        closest_time, min_diff = min(candidates, key=lambda x: x[1])
        
        if min_diff <= tolerance_ms:
            return int(closest_time)
        else:
            return None
    
    def verify_backfill(self, df_test_filled):
        """
        验证回填结果
        
        Parameters:
        -----------
        df_test_filled : DataFrame
            回填后的测试数据
        
        Returns:
        --------
        verification_stats : dict
            验证统计信息
        """
        print("\n" + "=" * 60)
        print("验证回填结果")
        print("=" * 60)
        
        # 检查缺失值
        missing_count = df_test_filled['labelArea'].isna().sum()
        total_count = len(df_test_filled)
        
        # 检查标签分布
        label_dist = df_test_filled['labelArea'].value_counts().sort_index()
        
        verification_stats = {
            'total_samples': total_count,
            'matched_samples': total_count - missing_count,
            'missing_samples': missing_count,
            'match_rate': (total_count - missing_count) / total_count if total_count > 0 else 0,
            'label_distribution': label_dist.to_dict()
        }
        
        print(f"总样本数: {total_count}")
        print(f"成功匹配: {total_count - missing_count}")
        print(f"缺失标签: {missing_count}")
        print(f"匹配率: {verification_stats['match_rate']:.4f}")
        print(f"\n标签分布:")
        print(label_dist)
        
        return verification_stats


def main():
    """主函数：演示标签回填流程"""
    print("=" * 60)
    print("数据标签回填")
    print("=" * 60)
    
    # 创建回填器
    backfiller = LabelBackfiller(
        original_train_path='train/训练数据.csv',
        cleaned_test_path='train/清洗测试数据.csv'
    )
    
    # 执行回填
    df_test_filled = backfiller.backfill_labels()
    
    # 验证回填结果
    verification_stats = backfiller.verify_backfill(df_test_filled)
    
    print("\n" + "=" * 60)
    print("标签回填完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

