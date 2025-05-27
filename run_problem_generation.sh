#!/bin/bash
# 运行图片到问题生成脚本 (四个问题版本)

# 设置工作目录
cd /c22940/zy/code/VLM-R1

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /c22940/zy/conda_envs/vlm-r1

# 默认参数
DATA_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_desktop.json"
OUTPUT_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_desktop_4x_problems.json"
IMAGE_ROOT="otherdata/ScreenSpot-v2"
TEMPERATURE=0.3
DELAY=0.5
START_INDEX=0
API_URL="http://localhost:8001/v1"
MODEL_NAME="8001vllm"
PROMPT="Generate 4 different tasks that require clicking on SINGLE, SPECIFIC elements in this screenshot. Each task should target a different clickable element and be formulated as a short imperative statement without mentioning clicking. Each task must have only ONE correct location to interact with. Format your response as: 1. [task1] 2. [task2] 3. [task3] 4. [task4]. Examples: '1. close this window 2. minimize this window 3. view daily challenges 4. open settings menu'"

# 解析命令行参数
while [ $# -gt 0 ]; do
  case "$1" in
    --start=*)
      START_INDEX="${1#*=}"
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      case "$DATASET" in
        desktop)
          DATA_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_desktop.json"
          OUTPUT_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_desktop_4x_problems.json"
          ;;
        mobile)
          DATA_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_mobile.json"
          OUTPUT_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_mobile_4x_problems.json"
          ;;
        web)
          DATA_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_web.json"
          OUTPUT_PATH="otherdata/ScreenSpot-v2/converted_data_click/screenspot_web_4x_problems.json"
          ;;
        *)
          echo "未知数据集: $DATASET，使用默认值 desktop"
          ;;
      esac
      ;;
    --temp=*)
      TEMPERATURE="${1#*=}"
      ;;
    --delay=*)
      DELAY="${1#*=}"
      ;;
    --model=*)
      MODEL_NAME="${1#*=}"
      ;;
    --api=*)
      API_URL="${1#*=}"
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
  shift
done

echo "==============================================="
echo "   ScreenSpot 图片问题生成 (4倍数据量版本)"
echo "==============================================="
echo "数据集路径: $DATA_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "模型: $MODEL_NAME"
echo "API URL: $API_URL"
echo "温度: $TEMPERATURE"
echo "起始索引: $START_INDEX"
echo "请求延迟: $DELAY 秒"
echo "==============================================="
echo "注意: 每张图片将生成4个不同的问题，数据量将扩大4倍"
echo "==============================================="

# 运行Python脚本
python generate_problems_from_images.py \
  --data_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  --image_root "$IMAGE_ROOT" \
  --temperature "$TEMPERATURE" \
  --start_index "$START_INDEX" \
  --delay "$DELAY" \
  --api_url "$API_URL" \
  --model_name "$MODEL_NAME" \
  --prompt "$PROMPT"

echo "任务完成！数据量已扩大4倍。"