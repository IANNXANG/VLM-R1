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
    parser.add_argument("--data_root", type=str, default="/c22940/zy/code/VLM-R1/test_data/rec_jsons_processed",
                        help="数据集根目录")
    parser.add_argument("--image_root", type=str, default="/c22940/zy/code/VLM-R1/data/images",
                        help="图像根目录")
    parser.add_argument("--datasets", type=str, nargs='+', 
                        default=['refcoco_val', 'refcocop_val', 'refcocog_val'],
                        help="要评估的数据集列表")
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
DATA_ROOT = args.data_root
IMAGE_ROOT = args.image_root
TEST_DATASETS = args.datasets

# 设置输出路径
OUTPUT_PATH = f"./logs/baseline/{MODEL_NAME}/rec_results_{{DATASET}}_{MODEL_NAME}.json"

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
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs.extend(batch_output_text)

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
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution']
            # 使用点提取而不是边界框
            model_answer = extract_point_answer(original_output)
            
            # 评估点是否在真实边界框内
            correct = 0
            if model_answer is not None:
                if point_in_bbox(model_answer, ground_truth):
                    correct = 1
            correct_number += correct
            
            # 创建结果字典
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH.format(DATASET=ds)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)
    
    # Synchronize all processes
    dist.barrier()





