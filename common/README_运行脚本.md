# 完整流程执行脚本使用说明

## 脚本文件

1. **run_all.sh** - Linux/Mac/Git Bash 版本
2. **run_all.bat** - Windows 版本

## 使用方法

### Windows 用户

1. **方法一：双击运行**
   - 直接双击 `run_all.bat` 文件即可

2. **方法二：命令行运行**
   ```cmd
   cd common
   run_all.bat
   ```

### Linux/Mac/Git Bash 用户

1. **赋予执行权限**（首次使用）
   ```bash
   cd common
   chmod +x run_all.sh
   ```

2. **执行脚本**
   ```bash
   ./run_all.sh
   ```
   或
   ```bash
   bash run_all.sh
   ```

## 执行流程

脚本会按以下顺序自动执行：

1. **a_数据清洗代码.py** - 数据清洗与预处理
2. **b_模型训练代码.py** - 模型训练
3. **c_模型调用以及结果分析代码.py** - 模型预测
4. **d_数据回填代码.py** - 数据回填
5. **e_指标计算输出报告代码.py** - 指标计算与报告生成

## 输出文件

执行完成后，会生成以下文件：

- `train/清洗训练数据.csv` - 清洗后的训练数据
- `train/清洗测试数据.csv` - 清洗后的测试数据（包含预测结果）
- `model/improved_lda_model.pkl` - 训练好的模型
- `model/preprocessor.pkl` - 预处理器对象
- `results/清洗测试数据_指标报告_*.txt` - 详细指标报告
- `results/evaluation_report.txt` - 评估报告

## 注意事项

1. **确保Python环境已安装**
   - 脚本会自动检测 `python` 或 `python3` 命令

2. **确保数据文件存在**
   - `train/训练数据.csv` 必须存在

3. **错误处理**
   - 如果任何步骤失败，脚本会立即停止并显示错误信息

4. **执行时间**
   - 完整流程可能需要几分钟到几十分钟，取决于数据量和硬件性能

## 单独执行

如果需要单独执行某个步骤，可以直接运行对应的Python文件：

```bash
# Windows
python a_数据清洗代码.py
python b_模型训练代码.py
python c_模型调用以及结果分析代码.py
python d_数据回填代码.py
python e_指标计算输出报告代码.py

# Linux/Mac
python3 a_数据清洗代码.py
python3 b_模型训练代码.py
python3 c_模型调用以及结果分析代码.py
python3 d_数据回填代码.py
python3 e_指标计算输出报告代码.py
```

