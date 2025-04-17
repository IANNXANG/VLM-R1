#!/usr/bin/env python3
# 将ScreenSpot-v2数据转换为RefCOCO格式

import json
import os
from pathlib import Path
from PIL import Image

# 输入文件
INPUT_FILES = [
    "screenspot_desktop_v2.json",
    "screenspot_mobile_v2.json", 
    "screenspot_web_v2.json"
]

# 输出目录
OUTPUT_DIR = "converted_data_click"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图像目录 - 修正路径
IMAGE_DIR = "images/screenspotv2_image"

def convert_to_refcoco_format(input_file, output_file, dataset_type):
    """
    转换ScreenSpot-v2数据为RefCOCO格式，保持原始顺序不变
    
    参数:
        input_file: 输入的ScreenSpot-v2 JSON文件路径
        output_file: 输出的RefCOCO格式JSON文件路径
        dataset_type: 数据集类型 (desktop, mobile, web)
    """
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 不进行随机打乱，保持原始顺序
    
    # 统计可用的图像文件
    available_images = []
    for item in data:
        image_path = os.path.join(IMAGE_DIR, item['img_filename'])
        if os.path.exists(image_path):
            available_images.append(item)
        else:
            pass  # 不打印警告，减少输出信息
    
    print(f"原始数据: {len(data)}条, 有效数据(图像存在): {len(available_images)}条")
    
    # 使用可用的图像
    data = available_images
    
    # 不划分训练和验证集，保持原始数据完整
    
    # 转换函数
    def convert_item(item, id_counter):
        # 获取图像文件路径
        img_filename = item['img_filename']
        image_path = os.path.join(IMAGE_DIR, img_filename)
        
        # 读取图像获取高度和宽度
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"警告: 无法读取图像 {image_path}: {e}")
            width, height = 0, 0
        
        # 将ScreenSpot的[x, y, width, height]转换为RefCOCO的[x, y, x+width, y+height]
        bbox = item["bbox"]
        solution = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        
        # 计算0-1000范围的归一化坐标
        normalized_solution = [
            int(solution[0] / width * 1000) if width > 0 else 0,
            int(solution[1] / height * 1000) if height > 0 else 0,
            int(solution[2] / width * 1000) if width > 0 else 0,
            int(solution[3] / height * 1000) if height > 0 else 0
        ]
        
        # 构建问题描述
        problem = item['instruction']
        
        # 构建转换后的条目 - 仅保留RefCOCO中实际使用的字段
        converted_item = {
            "dataset": f"screenspot_{dataset_type}",  # 使用标准的 dataset 字段
            "text_type": "caption",
            "height": height,
            "width": width,
            "normal_caption": item["instruction"],
            "image": f"{IMAGE_DIR}/{img_filename}",
            "problem": problem,
            "solution": solution,
            "normalized_solution": normalized_solution
        }
        
        return converted_item
    
    # 转换所有数据
    converted_data = []
    id_counter = 0
    for item in data:
        converted_data.append(convert_item(item, id_counter))
        id_counter += 1
    
    # 保存转换后的数据
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    print(f"已保存数据: {output_file}, 共 {len(converted_data)} 条记录")
    
    return converted_data

def main():
    # 处理每个输入文件，保持独立不合并
    for input_file in INPUT_FILES:
        print(f"处理文件: {input_file}")
        dataset_type = input_file.split("_")[1]  # 提取desktop/mobile/web
        output_file = os.path.join(OUTPUT_DIR, f"screenspot_{dataset_type}.json")
        
        convert_to_refcoco_format(input_file, output_file, dataset_type)
    
    # 创建数据集配置文件（yaml格式，用于训练）
    yaml_config = {
        "datasets": [
            {
                "json_path": os.path.abspath(os.path.join(OUTPUT_DIR, "screenspot_desktop.json")),
                "sampling_strategy": "all"
            },
            {
                "json_path": os.path.abspath(os.path.join(OUTPUT_DIR, "screenspot_mobile.json")),
                "sampling_strategy": "all"
            },
            {
                "json_path": os.path.abspath(os.path.join(OUTPUT_DIR, "screenspot_web.json")),
                "sampling_strategy": "all"
            }
        ]
    }
    
    yaml_output = os.path.join(OUTPUT_DIR, "screenspot_train_config.yaml")
    with open(yaml_output, 'w') as f:
        f.write("# ScreenSpot数据集配置文件\n")
        f.write("datasets:\n")
        for dataset in yaml_config["datasets"]:
            f.write(f"  - json_path: {dataset['json_path']}\n")
            f.write(f"    sampling_strategy: {dataset['sampling_strategy']}\n")
    
    print(f"已创建训练配置文件: {yaml_output}")
    
    # 输出使用说明
    print("\n使用说明:")
    print("1. 训练数据已转换为RefCOCO格式并保存")
    print("2. 要使用该数据集进行训练，请在训练脚本中指定:")
    print(f"   --dataset_name {os.path.abspath(yaml_output)}")
    print(f"   --image_root {os.path.abspath('.')}")
    print("3. 要使用该数据集进行测试，请修改test_rec_r1.py脚本中的:")
    print(f"   DATA_ROOT = \"{os.path.abspath(OUTPUT_DIR)}\"")
    print(f"   TEST_DATASETS = ['screenspot_desktop', 'screenspot_mobile', 'screenspot_web']")
    print(f"   IMAGE_ROOT = \"{os.path.abspath('.')}\"")

if __name__ == "__main__":
    main() 