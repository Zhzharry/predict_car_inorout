@echo off
REM ============================================================================
REM 完整流程执行脚本 (Windows版本)
REM 功能：依次执行数据清洗、模型训练、模型预测、数据回填、指标计算
REM 使用方法：双击运行或在命令行执行 run_all.bat
REM ============================================================================

chcp 65001 >nul
setlocal enabledelayedexpansion

REM 设置脚本目录
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM 颜色定义（Windows CMD支持有限，使用简单标记）
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

echo ==========================================
echo 开始执行完整流程
echo ==========================================
echo.

REM 检查Python是否安装
echo %INFO% 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo %ERROR% Python未安装或未在PATH中
        pause
        exit /b 1
    ) else (
        set "PYTHON_CMD=python3"
    )
) else (
    set "PYTHON_CMD=python"
)

echo %INFO% 使用Python命令: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM 记录开始时间
set "TOTAL_START=%time%"

REM 步骤1: 数据清洗
echo ==========================================
echo %INFO% 步骤 1/5: 数据清洗
echo ==========================================
set "START_TIME=%time%"
%PYTHON_CMD% "a_数据清洗代码.py"
if errorlevel 1 (
    echo %ERROR% 数据清洗失败，终止执行
    pause
    exit /b 1
)
set "END_TIME=%time%"
echo %SUCCESS% a_数据清洗代码.py 执行成功
echo.

REM 步骤2: 模型训练
echo ==========================================
echo %INFO% 步骤 2/5: 模型训练
echo ==========================================
set "START_TIME=%time%"
%PYTHON_CMD% "b_模型训练代码.py"
if errorlevel 1 (
    echo %ERROR% 模型训练失败，终止执行
    pause
    exit /b 1
)
set "END_TIME=%time%"
echo %SUCCESS% b_模型训练代码.py 执行成功
echo.

REM 步骤3: 模型预测
echo ==========================================
echo %INFO% 步骤 3/5: 模型预测
echo ==========================================
set "START_TIME=%time%"
%PYTHON_CMD% "c_模型调用以及结果分析代码.py"
if errorlevel 1 (
    echo %ERROR% 模型预测失败，终止执行
    pause
    exit /b 1
)
set "END_TIME=%time%"
echo %SUCCESS% c_模型调用以及结果分析代码.py 执行成功
echo.

REM 步骤4: 数据回填
echo ==========================================
echo %INFO% 步骤 4/5: 数据回填
echo ==========================================
set "START_TIME=%time%"
%PYTHON_CMD% "d_数据回填代码.py"
if errorlevel 1 (
    echo %ERROR% 数据回填失败，终止执行
    pause
    exit /b 1
)
set "END_TIME=%time%"
echo %SUCCESS% d_数据回填代码.py 执行成功
echo.

REM 步骤5: 指标计算
echo ==========================================
echo %INFO% 步骤 5/5: 指标计算
echo ==========================================
set "START_TIME=%time%"
%PYTHON_CMD% "e_指标计算输出报告代码.py"
if errorlevel 1 (
    echo %ERROR% 指标计算失败，终止执行
    pause
    exit /b 1
)
set "END_TIME=%time%"
echo %SUCCESS% e_指标计算输出报告代码.py 执行成功
echo.

REM 输出总结
set "TOTAL_END=%time%"
echo ==========================================
echo %SUCCESS% 所有步骤执行完成！
echo ==========================================
echo.
echo %INFO% 生成的文件：
echo   - train/清洗训练数据.csv
echo   - train/清洗测试数据.csv
echo   - model/improved_lda_model.pkl
echo   - model/preprocessor.pkl
echo   - results/清洗测试数据_指标报告_*.txt
echo   - results/evaluation_report.txt
echo.
echo %SUCCESS% 流程执行成功！
echo.
pause

