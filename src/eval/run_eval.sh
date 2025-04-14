#!/bin/bash
cd /c22940/zy/code/VLM-R1

# 激活环境

#conda activate /c22940/zy/conda_envs/vlm-r1

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 运行分布式评估
torchrun --nproc_per_node=4 src/eval/test_rec_r1.py

echo "评估完成！" 