#!/bin/bash

# ============================================================================
# 完整流程执行脚本
# 功能：依次执行数据清洗、模型训练、模型预测、数据回填、指标计算
# 使用方法：bash run_all.sh 或 ./run_all.sh
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python是否安装
check_python() {
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python未安装或未在PATH中"
        exit 1
    fi
    
    # 优先使用python3，如果没有则使用python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    log_info "使用Python命令: $PYTHON_CMD"
    $PYTHON_CMD --version
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件不存在: $1"
        return 1
    fi
    return 0
}

# 执行Python脚本
run_script() {
    local script_name=$1
    local script_path="$SCRIPT_DIR/$script_name"
    
    log_info "=========================================="
    log_info "开始执行: $script_name"
    log_info "=========================================="
    
    if ! check_file "$script_path"; then
        return 1
    fi
    
    local start_time=$(date +%s)
    
    if $PYTHON_CMD "$script_path"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "$script_name 执行成功 (耗时: ${duration}秒)"
        return 0
    else
        log_error "$script_name 执行失败"
        return 1
    fi
}

# 主函数
main() {
    log_info "=========================================="
    log_info "开始执行完整流程"
    log_info "=========================================="
    echo ""
    
    # 检查Python
    check_python
    echo ""
    
    # 记录开始时间
    total_start_time=$(date +%s)
    
    # 步骤1: 数据清洗
    log_info "步骤 1/5: 数据清洗"
    if ! run_script "a_数据清洗代码.py"; then
        log_error "数据清洗失败，终止执行"
        exit 1
    fi
    echo ""
    
    # 步骤2: 模型训练
    log_info "步骤 2/5: 模型训练"
    if ! run_script "b_模型训练代码.py"; then
        log_error "模型训练失败，终止执行"
        exit 1
    fi
    echo ""
    
    # 步骤3: 模型预测
    log_info "步骤 3/5: 模型预测"
    if ! run_script "c_模型调用以及结果分析代码.py"; then
        log_error "模型预测失败，终止执行"
        exit 1
    fi
    echo ""
    
    # 步骤4: 数据回填
    log_info "步骤 4/5: 数据回填"
    if ! run_script "d_数据回填代码.py"; then
        log_error "数据回填失败，终止执行"
        exit 1
    fi
    echo ""
    
    # 步骤5: 指标计算
    log_info "步骤 5/5: 指标计算"
    if ! run_script "e_指标计算输出报告代码.py"; then
        log_error "指标计算失败，终止执行"
        exit 1
    fi
    echo ""
    
    # 计算总耗时
    total_end_time=$(date +%s)
    total_duration=$((total_end_time - total_start_time))
    total_minutes=$((total_duration / 60))
    total_seconds=$((total_duration % 60))
    
    # 输出总结
    log_success "=========================================="
    log_success "所有步骤执行完成！"
    log_success "=========================================="
    log_info "总耗时: ${total_minutes}分${total_seconds}秒"
    log_info ""
    log_info "生成的文件："
    log_info "  - train/清洗训练数据.csv"
    log_info "  - train/清洗测试数据.csv"
    log_info "  - model/improved_lda_model.pkl"
    log_info "  - model/preprocessor.pkl"
    log_info "  - results/清洗测试数据_指标报告_*.txt"
    log_info "  - results/evaluation_report.txt"
    log_success ""
    log_success "流程执行成功！"
}

# 执行主函数
main

