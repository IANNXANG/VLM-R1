#!/bin/bash
# 提取测试结果并生成CSV文件

# 设置环境
cd /c22940/zy/code/VLM-R1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /c22940/zy/conda_envs/vlm-r1

# 命令行参数
LOG_DIR="${1:-logs}"  # 默认为 logs 目录
OUTPUT_CSV="${2:-test_results.csv}"  # 默认输出文件名

# 创建CSV文件头
echo "Model,Dataset,Accuracy,Steps" > "$OUTPUT_CSV"

# 查找所有结果文件，包括baseline和检查点
find "$LOG_DIR" -name "rec_results_*.json" | sort | while read -r result_file; do
    # 从文件路径中提取信息
    filename=$(basename "$result_file")
    dirpath=$(dirname "$result_file")
    model_dir=$(basename "$dirpath")
    
    # 根据文件路径判断是baseline还是检查点
    if [[ "$dirpath" == *"/baseline/"* ]]; then
        # 基线模型
        model_type="baseline"
        model_name="$model_dir"
        
        # 从文件名中提取数据集名称
        # 文件格式: rec_results_${dataset}_$model_name.json
        dataset=$(echo "$filename" | sed -E 's/rec_results_(.+)_'$model_name'\.json/\1/')
        steps="N/A"
    else
        # 检查点模型
        model_type="checkpoint"
        model_name="$model_dir"
        
        # 从文件名中提取数据集名称
        # 文件格式: rec_results_${dataset}.json
        dataset=$(echo "$filename" | sed -E 's/rec_results_(.+)\.json/\1/')
        
        # 从模型名称中提取步数
        if [[ "$model_name" == "original-model" ]]; then
            steps="0"
        else
            steps=$(echo "$model_name" | grep -o -E 'checkpoint-[0-9]+' | grep -o -E '[0-9]+')
            # 如果没有找到steps，设为N/A
            if [ -z "$steps" ]; then
                steps="N/A"
            fi
        fi
    fi
    
    # 提取准确率
    if [ -f "$result_file" ]; then
        # 尝试使用jq解析JSON (如果安装了jq)
        if command -v jq &> /dev/null; then
            accuracy=$(jq -r '.accuracy' "$result_file" 2>/dev/null)
        else
            # 回退到使用grep
            accuracy=$(grep -o '"accuracy": [0-9.]*' "$result_file" | cut -d' ' -f2)
        fi
        
        # 如果提取失败，使用N/A
        if [ -z "$accuracy" ]; then
            accuracy="N/A"
        fi
        
        # 添加到CSV
        echo "$model_name,$dataset,$accuracy,$steps" >> "$OUTPUT_CSV"
    fi
done

# 排序CSV文件 (跳过首行)
TEMP_FILE=$(mktemp)
head -n 1 "$OUTPUT_CSV" > "$TEMP_FILE"
tail -n +2 "$OUTPUT_CSV" | sort -t ',' -k2,2 -k4,4n >> "$TEMP_FILE"
mv "$TEMP_FILE" "$OUTPUT_CSV"

echo "结果已保存到 $OUTPUT_CSV"

# 可选：生成简单的分析报告
echo ""
echo "===== 数据集性能摘要 ====="
for dataset in $(tail -n +2 "$OUTPUT_CSV" | cut -d ',' -f2 | sort | uniq); do
    echo "数据集: $dataset"
    echo "------------------------"
    
    # 基线模型
    echo "基线模型:"
    grep ",$dataset," "$OUTPUT_CSV" | grep -v "checkpoint" | sort -t ',' -k3,3nr | head -5 | while read -r line; do
        model=$(echo "$line" | cut -d ',' -f1)
        acc=$(echo "$line" | cut -d ',' -f3)
        echo "  $model: $acc"
    done
    
    # 检查点模型 (按步数排序，只显示最佳性能)
    echo "最佳检查点性能:"
    grep ",$dataset," "$OUTPUT_CSV" | grep "checkpoint" | sort -t ',' -k3,3nr | head -5 | while read -r line; do
        model=$(echo "$line" | cut -d ',' -f1)
        acc=$(echo "$line" | cut -d ',' -f3)
        steps=$(echo "$line" | cut -d ',' -f4)
        echo "  $model (step $steps): $acc"
    done
    
    echo ""
done

echo "提取完成！结果已保存到 $OUTPUT_CSV" 