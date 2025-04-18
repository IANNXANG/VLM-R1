from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import argparse


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 新增：解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="ScreenSpot模型评估脚本")
    parser.add_argument("--steps", type=int, default=0, help="检查点步数，0表示原始模型")
    return parser.parse_args()

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

# 解析命令行参数
args = parse_args()
# 设置评估的检查点步数
steps = args.steps
if rank == 0:
    print("Steps: ", steps)

# 修改：使用ScreenSpot模型的路径
RUN_NAME = "Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click"

if steps != 0:
    MODEL_PATH=f"/c22940/zy/code/VLM-R1/src/open-r1-multimodal/output/{RUN_NAME}/checkpoint-{steps}" 
    # 新增：为log添加子目录
    MODEL_LOG_DIR = f"screenspot-click-checkpoint-{steps}"
else:
    MODEL_PATH = "/c22940/zy/model/Qwen2.5-VL-7B-Instruct"
    # 新增：为log添加子目录
    MODEL_LOG_DIR = "screenspot-click-original-model"
# 修改：将日志存储在相应子目录中
OUTPUT_PATH=f"./logs/{MODEL_LOG_DIR}/screenspot_results_{{DATASET}}_{{STEPS}}.json"

BSZ=4
# 修改：使用ScreenSpot数据路径
DATA_ROOT = "/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2/converted_data_click"

# 修改：使用ScreenSpot测试数据集
TEST_DATASETS = ['screenspot_desktop', 'screenspot_mobile', 'screenspot_web']
# 修改：使用ScreenSpot图像路径
IMAGE_ROOT = "/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2"

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
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    point_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'
    
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        point_match = re.search(point_pattern, content_answer)
        if point_match:
            point = [int(point_match.group(1)), int(point_match.group(2))]
            return point
    return [0, 0]

def point_in_bbox(point, bbox):
    """检查点坐标是否在边界框内"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

# 不限制样本数量，使用全部ScreenSpot数据
# 修复：使用足够大的整数值而不是无穷大
num_samples = 100000
for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.seed(42)
    random.shuffle(data)
    data = data[:num_samples]

    # 系统提示词
    SYSTEM_PROMPT = (
        "You are a GUI agent assistant that helps users interact with graphical interfaces. "
        "When asked to perform an action on the interface, you should provide the exact coordinates "
        "where to click. Look at the image carefully, identify the correct element, and provide "
        "the single most appropriate click point [x, y] that would allow the user to interact with "
        "the element. Provide your reasoning within <think> </think> tags, and your final answer "
        "with the coordinates in <answer> </answer> tags, using the format [x, y]."
    )

    # 使用点击任务的提示词模板
    QUESTION_TEMPLATE = "{Question} Look at the image and identify where to click. First reason about the correct location in <think> </think> tags, then provide the exact [x, y] coordinates in <answer> </answer> tags."

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
            # 使用点提取函数代替边界框提取
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
        output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
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





