#!/bin/bash
# 评估多个模型在多个数据集上的性能 
# 主要需要修改data_root和run_name和启动的.py 文件

# 设置环境
cd /c22940/zy/code/VLM-R1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /c22940/zy/conda_envs/vlm-r1
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 定义基础参数
DATA_ROOT="/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2/converted_data_click"
IMAGE_ROOT="/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2"
NUM_GENERATIONS=16  # 每个样本的生成次数

# 定义要测试的温度值
TEMPERATURES=(0.3 0.5 0.7 0.9 1.0)

# 定义要评估的数据集 - 可以根据需要修改
DATASETS=("screenspot_desktop")
#DATASETS=("refcoco_val" "refcocop_val" "refcocog_val")
# DATASETS=("lisa_test")  # 如果要评估LISA数据集，取消注释这行并注释上面一行

# 定义要评估的模型配置
# 基线模型 - 如果要同时评估多个基线模型，可以添加多项
BASELINE_MODELS=(
  "/c22940/zy/model/Qwen2.5-VL-7B-Instruct|qwen2_5vl_7b_instruct_baseline"
  # 可以添加更多基线模型，格式: "模型路径|模型名称"
)

# 训练检查点 - 指定训练名称和检查点步数
# Checkpoint目录
CHECKPOINT_DIR="/c22940/zy/code/VLM-R1/src/open-r1-multimodal/output"
# 可能的训练名，注意根据实际情况调整
RUN_NAME="Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click"
# 需要测试的检查点步数列表
CHECKPOINTS=(0 334)  # 0表示原始模型，其他为检查点步数

# 创建日志目录
mkdir -p logs

# 开始循环测试不同温度值
for temp in "${TEMPERATURES[@]}"; do
  echo "====================================================="
  echo "开始使用温度值 $temp 进行评估"
  echo "====================================================="

  # 评估基线模型
  for model_config in "${BASELINE_MODELS[@]}"; do
    # 解析模型配置
    IFS="|" read -r model_path model_name <<< "$model_config"
    
    echo "======================================"
    echo "开始评估基线模型: $model_name"
    echo "模型路径: $model_path"
    echo "温度值: $temp"
    echo "======================================"
    
    # 为模型创建包含温度值的日志目录
    mkdir -p logs/$RUN_NAME/$model_name/temp_$temp
    
    # 运行评估脚本
    torchrun --nproc_per_node=4 src/eval/test_rec_baseline_click_maj.py \
      --model_path "$model_path" \
      --model_name "$model_name" \
      --run_name "$RUN_NAME" \
      --data_root "$DATA_ROOT" \
      --image_root "$IMAGE_ROOT" \
      --datasets "${DATASETS[@]}" \
      --num_generations $NUM_GENERATIONS \
      --temperature $temp
    
    echo "基线模型 $model_name 使用温度 $temp 评估完成"
    echo "结果保存在: logs/$RUN_NAME/$model_name/temp_$temp/"
    echo "======================================"
    echo ""
    
    # 给系统一些时间清理资源
    sleep 5
  done

  # 评估训练检查点
  for steps in "${CHECKPOINTS[@]}"; do
    # 如果是0，表示使用原始模型
    if [ "$steps" -eq 0 ]; then
      MODEL_NAME="original-model"
    else
      MODEL_NAME="${RUN_NAME,,}-checkpoint-$steps"  # 转换为小写
    fi
    
    echo "======================================"
    echo "开始评估检查点: $MODEL_NAME"
    echo "步数: $steps"
    echo "温度值: $temp"
    echo "======================================"
    
    # 创建包含温度值的日志目录
    mkdir -p logs/$RUN_NAME/$MODEL_NAME/temp_$temp
    
    # 运行评估脚本，显式指定checkpoint_dir路径
    torchrun --nproc_per_node=4 src/eval/test_rec_r1_click_maj.py \
      --steps $steps \
      --run_name "$RUN_NAME" \
      --model_name "$MODEL_NAME" \
      --data_root "$DATA_ROOT" \
      --image_root "$IMAGE_ROOT" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --datasets "${DATASETS[@]}" \
      --num_generations $NUM_GENERATIONS \
      --temperature $temp
    
    echo "检查点 $MODEL_NAME 使用温度 $temp 评估完成"
    echo "结果保存在: logs/$RUN_NAME/$MODEL_NAME/temp_$temp/"
    echo "======================================"
    echo ""
    
    # 给系统一些时间清理资源
    sleep 5
  done

  echo "温度值 $temp 的所有模型评估完成!"
  echo "====================================================="
  echo ""
done

echo "所有温度值和模型评估完成!"
echo "结果保存在 logs/ 目录下的相应子文件夹中"

# 输出所有模型在每个温度值下的准确率摘要
echo ""
echo "准确率摘要:"
echo "====================================================="
for temp in "${TEMPERATURES[@]}"; do
  echo "温度值: $temp"
  echo "---------------------------------------------------"
  
  for dataset in "${DATASETS[@]}"; do
    echo "数据集: $dataset"
    echo "-------------------------------------"
    
    # 基线模型准确率
    for model_config in "${BASELINE_MODELS[@]}"; do
      IFS="|" read -r _ model_name <<< "$model_config"
      result_file="logs/$RUN_NAME/$model_name/temp_$temp/click_results_${dataset}_$model_name.json"
      
      if [ -f "$result_file" ]; then
        accuracy=$(grep -o '"accuracy": [0-9.]*' "$result_file" | cut -d' ' -f2)
        echo "$model_name: $accuracy%"
      else
        echo "$model_name: 结果文件不存在"
      fi
    done
    
    # 检查点模型准确率
    for steps in "${CHECKPOINTS[@]}"; do
      if [ "$steps" -eq 0 ]; then
        MODEL_NAME="original-model"
      else
        MODEL_NAME="${RUN_NAME,,}-checkpoint-$steps"
      fi
      
      result_file="logs/$RUN_NAME/$MODEL_NAME/temp_$temp/click_results_${dataset}.json"
      
      if [ -f "$result_file" ]; then
        accuracy=$(grep -o '"accuracy": [0-9.]*' "$result_file" | cut -d' ' -f2)
        echo "$MODEL_NAME: $accuracy%"
      else
        echo "$MODEL_NAME: 结果文件不存在"
      fi
    done
    
    echo "-------------------------------------"
  done
  
  echo "====================================================="
done 