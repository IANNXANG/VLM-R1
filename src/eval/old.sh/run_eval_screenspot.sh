#!/bin/bash
# 评估多个检查点在ScreenSpot数据集上的性能

# 设置环境
cd /c22940/zy/code/VLM-R1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /c22940/zy/conda_envs/vlm-r1
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 定义要评估的检查点步数
CHECKPOINTS=(0 10 20 30 40 50 60 70 80 90 100 200 300 334)

# 为每个检查点创建日志目录
for steps in "${CHECKPOINTS[@]}"; do
  if [ "$steps" -eq 0 ]; then
    mkdir -p logs/screenspot-original-model
  else
    mkdir -p logs/screenspot-checkpoint-$steps
  fi
done

# 依次评估每个检查点
for steps in "${CHECKPOINTS[@]}"; do
  echo "开始评估 checkpoint-$steps"
  echo "===================="
  
  # 运行评估脚本，传入steps参数
  torchrun --nproc_per_node=4 src/eval/test_rec_r1_screenspot.py --steps $steps
  
  echo "===================="
  echo "完成评估 checkpoint-$steps"
  echo ""
  
  # 给系统一些时间清理资源
  sleep 10
done

# 提取结果
python src/eval/extract_screenspot_results.py

echo "所有检查点评估完成!"
echo "结果保存在 logs/ 目录下的相应子文件夹中" 

