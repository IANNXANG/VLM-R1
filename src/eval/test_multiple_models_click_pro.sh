#!/bin/bash
# 评估多个模型在ScreenSpot-Pro数据集上的性能 
# 主要需要修改data_root和run_name和启动的.py 文件

# 设置环境
cd /c22940/zy/code/VLM-R1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /c22940/zy/conda_envs/vlm-r1
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 定义基础参数 - 更新为ScreenSpot-Pro数据路径
DATA_ROOT="/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-Pro/converted_data_pro"
IMAGE_ROOT="/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-Pro"

# 定义要评估的数据集 - 使用ScreenSpot-Pro的数据集
# 可以根据需要选择部分应用或全部应用评估
DATASETS=(
  "screenspot_pro_android_studio_macos"
  "screenspot_pro_autocad_windows"
  "screenspot_pro_blender_windows"
  "screenspot_pro_davinci_macos" 
  "screenspot_pro_excel_macos"
  "screenspot_pro_eviews_windows"
  "screenspot_pro_fruitloops_windows"
  "screenspot_pro_illustrator_windows"
  "screenspot_pro_inventor_windows"
  "screenspot_pro_linux_common_linux"
  "screenspot_pro_macos_common_macos"
  "screenspot_pro_matlab_macos"
  "screenspot_pro_origin_windows"
  "screenspot_pro_photoshop_windows"
  "screenspot_pro_powerpoint_windows"
  "screenspot_pro_premiere_windows"
  "screenspot_pro_pycharm_macos"
  "screenspot_pro_quartus_windows"
  "screenspot_pro_solidworks_windows"
  "screenspot_pro_stata_windows"
  "screenspot_pro_unreal_engine_windows"
  "screenspot_pro_vivado_windows"
  "screenspot_pro_vmware_macos"
  "screenspot_pro_vscode_macos"
  "screenspot_pro_windows_common_windows"
  "screenspot_pro_word_macos"
)

# 也可以按操作系统分组评估
# 仅评估macOS应用
#DATASETS=(
#  "screenspot_pro_android_studio_macos"
#  "screenspot_pro_davinci_macos"
#  "screenspot_pro_excel_macos"
#  "screenspot_pro_macos_common_macos"
#  "screenspot_pro_matlab_macos"
#  "screenspot_pro_pycharm_macos"
#  "screenspot_pro_vmware_macos"
#  "screenspot_pro_vscode_macos"
#  "screenspot_pro_word_macos"
#)

# 仅评估Windows应用
#DATASETS=(
#  "screenspot_pro_autocad_windows"
#  "screenspot_pro_blender_windows" 
#  "screenspot_pro_eviews_windows"
#  "screenspot_pro_fruitloops_windows"
#  "screenspot_pro_illustrator_windows"
#  "screenspot_pro_inventor_windows"
#  "screenspot_pro_origin_windows"
#  "screenspot_pro_photoshop_windows"
#  "screenspot_pro_powerpoint_windows"
#  "screenspot_pro_premiere_windows"
#  "screenspot_pro_quartus_windows"
#  "screenspot_pro_solidworks_windows"
#  "screenspot_pro_stata_windows"
#  "screenspot_pro_unreal_engine_windows"
#  "screenspot_pro_vivado_windows"
#  "screenspot_pro_windows_common_windows"
#)

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
CHECKPOINTS=(334 50 100 150 200 250 300 0)  # 0表示原始模型，其他为检查点步数

# 创建日志目录
mkdir -p logs


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
  echo "======================================"
  
  # 创建日志目录
  mkdir -p logs/$RUN_NAME/$MODEL_NAME
  
  # 运行评估脚本，显式指定checkpoint_dir路径
  torchrun --nproc_per_node=4 src/eval/test_rec_r1_click.py \
    --steps $steps \
    --run_name "$RUN_NAME" \
    --model_name "$MODEL_NAME" \
    --data_root "$DATA_ROOT" \
    --image_root "$IMAGE_ROOT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --datasets "${DATASETS[@]}"
  
  echo "检查点 $MODEL_NAME 评估完成"
  echo "结果保存在: logs/$RUN_NAME/$MODEL_NAME/"
  echo "======================================"
  echo ""
  
  # 给系统一些时间清理资源
  sleep 5
done

echo "所有模型评估完成!"
echo "结果保存在 logs/ 目录下的相应子文件夹中"

# 输出所有模型的准确率摘要（可选）
echo ""
echo "准确率摘要:"
echo "======================================"
for dataset in "${DATASETS[@]}"; do
  echo "数据集: $dataset"
  echo "-------------------------------------"
  
  # 基线模型准确率
  for model_config in "${BASELINE_MODELS[@]}"; do
    IFS="|" read -r _ model_name <<< "$model_config"
    result_file="logs/$RUN_NAME/$model_name/click_results_${dataset}_$model_name.json"
    
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
    
    result_file="logs/$RUN_NAME/$MODEL_NAME/click_results_${dataset}.json"
    
    if [ -f "$result_file" ]; then
      accuracy=$(grep -o '"accuracy": [0-9.]*' "$result_file" | cut -d' ' -f2)
      echo "$MODEL_NAME: $accuracy%"
    else
      echo "$MODEL_NAME: 结果文件不存在"
    fi
  done
  
  echo "======================================"
done 