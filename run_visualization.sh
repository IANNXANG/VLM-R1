#!/bin/bash
# ======================================================================
# 模型比较可视化脚本
# 此脚本用于运行visualize_model_comparison.py，比较两个模型在ScreenSpot任务上的表现
# 作者：[您的名字]
# 日期：2023-04-22
# ======================================================================

# ======================================================================
# 配置参数（请根据需要修改）
# ======================================================================
# 基础路径
BASE_PATH="/c22940/zy/code/VLM-R1"
# 训练后模型结果路径
TUNED_MODEL_PATH="logs/Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click/qwen2.5-vl-7b-grpo-screenspot-desktop-click-checkpoint-334"
# Baseline模型结果路径
BASELINE_MODEL_PATH="logs/Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click/qwen2_5vl_7b_instruct_baseline"
# 输出目录
OUTPUT_DIR="vis_results/model_comparison"
# 样本限制（0表示处理所有样本）
SAMPLE_LIMIT=20
# 点半径大小
POINT_RADIUS=5
# 边界框线条粗细
THICKNESS=3
# 边界框透明度
ALPHA=0.3
# 是否只处理训练后模型正确的样本 (true/false)
CORRECT_ONLY=false
# 是否只处理训练后模型错误的样本 (true/false)
INCORRECT_ONLY=false

# ======================================================================
# 函数定义
# ======================================================================
# 显示脚本使用方法
function show_usage {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help         显示此帮助信息"
    echo "  -p, --platform     指定平台 (desktop, mobile, web, all) [默认: all]"
    echo "  -l, --limit        设置样本限制 [默认: $SAMPLE_LIMIT]"
    echo "  -c, --correct-only 仅处理训练后模型正确的样本 [默认: $CORRECT_ONLY]"
    echo "  -i, --incorrect-only 仅处理训练后模型错误的样本 [默认: $INCORRECT_ONLY]"
    echo "  -o, --output-dir   指定输出目录 [默认: $OUTPUT_DIR]"
    echo ""
    echo "示例:"
    echo "  $0 --platform desktop --limit 10 --correct-only true"
    echo "  $0 -p mobile -l 50 -o vis_results/mobile_only"
    echo "  $0 --platform all --limit 0"
}

# 运行可视化比较
function run_visualization {
    local platform=$1
    local tuned_file="${TUNED_MODEL_PATH}/click_results_screenspot_${platform}.json"
    local baseline_file="${BASELINE_MODEL_PATH}/click_results_screenspot_${platform}_qwen2_5vl_7b_instruct_baseline.json"
    local output_dir="${OUTPUT_DIR}/${platform}"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 构建命令参数
    local cmd_args="--tuned $tuned_file --baseline $baseline_file --output $output_dir --point-radius $POINT_RADIUS --thickness $THICKNESS --alpha $ALPHA"
    
    # 添加样本限制（如果非零）
    if [ "$SAMPLE_LIMIT" -gt 0 ]; then
        cmd_args="$cmd_args --limit $SAMPLE_LIMIT"
    fi
    
    # 添加其他选项
    if [ "$CORRECT_ONLY" = true ]; then
        cmd_args="$cmd_args --correct-only"
    fi
    
    if [ "$INCORRECT_ONLY" = true ]; then
        cmd_args="$cmd_args --incorrect-only"
    fi
    
    # 显示将要执行的命令
    echo "执行命令: python visualize_model_comparison.py $cmd_args"
    
    # 执行命令
    python visualize_model_comparison.py $cmd_args
    
    # 检查命令执行结果
    if [ $? -eq 0 ]; then
        echo "✅ ${platform}平台可视化完成，结果保存在: $output_dir"
    else
        echo "❌ ${platform}平台可视化失败!"
    fi
    
    echo ""
}

# ======================================================================
# 参数解析
# ======================================================================
PLATFORM="all"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -l|--limit)
            SAMPLE_LIMIT="$2"
            shift 2
            ;;
        -c|--correct-only)
            CORRECT_ONLY="$2"
            shift 2
            ;;
        -i|--incorrect-only)
            INCORRECT_ONLY="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ======================================================================
# 主程序
# ======================================================================
# 检查参数有效性
if [ "$CORRECT_ONLY" = true ] && [ "$INCORRECT_ONLY" = true ]; then
    echo "错误: --correct-only 和 --incorrect-only 不能同时为true"
    exit 1
fi

# 切换到基础目录
cd "$BASE_PATH" || { echo "错误: 无法切换到目录 $BASE_PATH"; exit 1; }

# 检查可视化脚本是否存在
if [ ! -f "visualize_model_comparison.py" ]; then
    echo "错误: visualize_model_comparison.py 文件不存在!"
    exit 1
fi

# 根据指定的平台运行可视化
echo "=== 开始模型比较可视化 ==="
echo "平台: $PLATFORM"
echo "样本限制: $SAMPLE_LIMIT"
echo "只处理正确样本: $CORRECT_ONLY"
echo "只处理错误样本: $INCORRECT_ONLY"
echo "输出目录: $OUTPUT_DIR"
echo ""

if [ "$PLATFORM" = "all" ] || [ "$PLATFORM" = "desktop" ]; then
    echo "=== 处理桌面平台 ==="
    run_visualization "desktop"
fi

if [ "$PLATFORM" = "all" ] || [ "$PLATFORM" = "mobile" ]; then
    echo "=== 处理移动平台 ==="
    run_visualization "mobile"
fi

if [ "$PLATFORM" = "all" ] || [ "$PLATFORM" = "web" ]; then
    echo "=== 处理网页平台 ==="
    run_visualization "web"
fi

echo "=== 所有可视化任务完成 ==="
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "使用示例:"
echo "1. 查看桌面平台可视化结果: open ${OUTPUT_DIR}/desktop"
echo "2. 查看移动平台可视化结果: open ${OUTPUT_DIR}/mobile"
echo "3. 查看网页平台可视化结果: open ${OUTPUT_DIR}/web"
echo ""

# 脚本结束
exit 0 