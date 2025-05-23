from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="REC模型基线评估脚本")
    parser.add_argument("--model_path", type=str, default="/c22940/zy/model/Qwen2.5-VL-7B-Instruct", 
                        help="模型路径")
    parser.add_argument("--model_name", type=str, default="qwen2_5vl_7b_instruct_baseline",
                        help="模型名称，用于输出结果文件命名")
    parser.add_argument("--run_name", type=str, default="Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click",
                        help="训练运行名称，用于构建日志目录路径")
    parser.add_argument("--data_root", type=str, default="/c22940/zy/code/VLM-R1/test_data/rec_jsons_processed",
                        help="数据集根目录")
    parser.add_argument("--image_root", type=str, default="/c22940/zy/code/VLM-R1/data/images",
                        help="图像根目录")
    parser.add_argument("--datasets", type=str, nargs='+', 
                        default=['refcoco_val', 'refcocop_val', 'refcocog_val'],
                        help="要评估的数据集列表")
    parser.add_argument("--num_generations", type=int, default=16,
                        help="每个样本的生成次数")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="生成温度，控制采样随机性")
    return parser.parse_args()

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"

# 解析命令行参数
args = parse_args()

# 使用命令行参数
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name
RUN_NAME = args.run_name
DATA_ROOT = args.data_root
IMAGE_ROOT = args.image_root
TEST_DATASETS = args.datasets
NUM_GENERATIONS = args.num_generations
TEMPERATURE = args.temperature

# 设置输出路径，添加温度信息
OUTPUT_PATH = f"./logs/{RUN_NAME}/{MODEL_NAME}/temp_{TEMPERATURE}/click_results_{{DATASET}}_{MODEL_NAME}.json"

if rank == 0:
    print(f"Using temperature: {TEMPERATURE}")
    print(f"Results will be saved to: {OUTPUT_PATH.format(DATASET='<dataset>')}")

BSZ=4

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_point_answer(content):
    # 首先尝试匹配二维点坐标
    point_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'
    point_match = re.search(point_pattern, content)
    if point_match:
        point = [int(point_match.group(1)), int(point_match.group(2))]
        return point
    
    # 如果没有找到二维点坐标，尝试匹配四维边界框坐标
    bbox_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    bbox_match = re.search(bbox_pattern, content)
    if bbox_match:
        # 提取边界框坐标
        x1 = int(bbox_match.group(1))
        y1 = int(bbox_match.group(2))
        x2 = int(bbox_match.group(3))
        y2 = int(bbox_match.group(4))
        # 计算中点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return [center_x, center_y]
    
    # 如果都没有找到，返回默认值
    return [0, 0]

def point_in_bbox(point, bbox):
    """检查点坐标是否在边界框内"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

num_samples = 2000
for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.seed(42)
    random.shuffle(data)
    data = data[:num_samples]
    
    # 使用点击任务的提示词模板
    SYSTEM_PROMPT = (
     "Identify the click position as a **2D point** (x, y). Provide exactly **one pair** of coordinates in the format [x, y]."
    )
    QUESTION_TEMPLATE = "{Question}"
    
    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]
    
    messages = []

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        batch_outputs = []
        for _ in range(NUM_GENERATIONS):  # 顺序生成多次
            generated_ids = model.generate(
                **inputs, 
                use_cache=True, 
                max_new_tokens=256, 
                do_sample=True,  # 使用采样
                temperature=TEMPERATURE,  # 使用命令行参数设置的温度
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            batch_outputs.append(batch_output_text)
        
        # 将多次生成的输出组织成列表
        for i in range(len(batch_outputs[0])):  # 对每个输入样本
            sample_outputs = [batch_outputs[j][i] for j in range(NUM_GENERATIONS)]  # 收集所有生成的输出
            rank_outputs.append(sample_outputs)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        total_correct = 0

        for input_example, model_outputs in zip(data, all_outputs):
            original_outputs = model_outputs  # 现在是一个包含多个输出的列表
            ground_truth = input_example['solution']
            
            # 提取所有坐标点
            model_answers = [extract_point_answer(output) for output in original_outputs]
            
            # 计算多次预测的平均正确率
            correct_scores = []
            for answer in model_answers:
                if answer is not None:
                    correct_scores.append(1 if point_in_bbox(answer, ground_truth) else 0)
                else:
                    correct_scores.append(0)
            avg_correct = sum(correct_scores) / len(correct_scores)
            total_correct += avg_correct
            
            # 创建结果字典
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_outputs': original_outputs,  # 保存所有输出
                'extracted_answers': model_answers,  # 保存所有坐标
                'correct_scores': correct_scores,  # 保存每次预测的正确性
                'avg_correct': avg_correct  # 保存平均正确率
            }
            final_output.append(result)

        # 计算总体准确率
        accuracy = (total_correct / len(data)) * 100
        print(f"\nAccuracy of {ds} with temperature {TEMPERATURE}: {accuracy:.2f}%")

        # 保存结果到JSON文件
        output_path = OUTPUT_PATH.format(DATASET=ds)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'temperature': TEMPERATURE,  # 保存使用的温度值
                'results': final_output,
                'num_generations': NUM_GENERATIONS  # 使用全局参数
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)
    
    # Synchronize all processes
    dist.barrier()





