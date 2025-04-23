#!/usr/bin/env python3
# 将ScreenSpot-Pro数据转换为RefCOCO格式用于训练

import json
import os
import glob
from pathlib import Path
from PIL import Image

# 输入和输出目录 - 使用相对路径
INPUT_DIR = "annotations"
OUTPUT_DIR = "converted_data_pro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图像根目录 - 使用相对路径
IMAGE_ROOT = "images"

def convert_to_refcoco_format(input_file, output_file):
    """
    转换ScreenSpot-Pro数据为RefCOCO格式，保持原始顺序不变
    
    参数:
        input_file: 输入的ScreenSpot-Pro JSON文件路径
        output_file: 输出的RefCOCO格式JSON文件路径
    """
    # 提取数据集类型（应用名称和平台）
    filename = os.path.basename(input_file)
    app_platform = os.path.splitext(filename)[0]  # 如 android_studio_macos
    
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"处理文件: {input_file}")
    print(f"原始数据: {len(data)}条")
    
    # 统计可用的图像文件
    available_images = []
    for item in data:
        image_path = os.path.join(IMAGE_ROOT, item['img_filename'])
        if os.path.exists(image_path):
            available_images.append(item)
        else:
            print(f"警告: 图像不存在 {image_path}")
    
    print(f"有效数据(图像存在): {len(available_images)}条")
    
    # 使用可用的图像
    data = available_images
    
    # 转换函数
    def convert_item(item, id_counter):
        # 获取图像文件路径
        img_filename = item['img_filename']
        image_path = os.path.join(IMAGE_ROOT, img_filename)
        
        # 读取图像获取高度和宽度
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"警告: 无法读取图像 {image_path}: {e}")
            # 使用img_size字段
            if 'img_size' in item:
                width, height = item['img_size']
            else:
                width, height = 0, 0
        
        # 获取边界框
        bbox = item["bbox"]
        # ScreenSpot-Pro中bbox格式为[x1, y1, x2, y2]，无需转换
        solution = bbox
        
        # 计算0-1000范围的归一化坐标
        normalized_solution = [
            int(solution[0] / width * 1000) if width > 0 else 0,
            int(solution[1] / height * 1000) if height > 0 else 0,
            int(solution[2] / width * 1000) if width > 0 else 0,
            int(solution[3] / height * 1000) if height > 0 else 0
        ]
        
        # 构建问题描述 - 使用英文指令
        problem = item['instruction']
        
        # 构建转换后的条目
        converted_item = {
            "dataset": f"screenspot_pro_{app_platform}",
            "text_type": "caption",
            "height": height,
            "width": width,
            "normal_caption": problem,
            "image": f"{IMAGE_ROOT}/{img_filename}",  # 使用相对于ScreenSpot-Pro目录的路径
            "problem": problem,
            "solution": solution,
            "normalized_solution": normalized_solution,
            "ui_type": item.get("ui_type", "unknown"),  # 添加界面元素类型
            "application": item.get("application", "unknown"),  # 添加应用名称
            "platform": item.get("platform", "unknown")  # 添加平台
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
    
    return converted_data, app_platform

def main():
    # 获取所有输入JSON文件
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"发现 {len(input_files)} 个JSON文件")
    
    # 转换每个文件
    all_datasets = []
    for input_file in input_files:
        filename = os.path.basename(input_file)
        app_platform = os.path.splitext(filename)[0]
        output_file = os.path.join(OUTPUT_DIR, f"screenspot_pro_{app_platform}.json")
        
        converted_data, dataset_name = convert_to_refcoco_format(input_file, output_file)
        
        # 如果有数据，添加到数据集列表
        if converted_data:
            all_datasets.append({
                "json_path": os.path.abspath(output_file),
                "sampling_strategy": "all"
            })
    
    # 创建数据集配置文件（yaml格式，用于训练）
    yaml_output = os.path.join(OUTPUT_DIR, "screenspot_pro_train_config.yaml")
    with open(yaml_output, 'w') as f:
        f.write("# ScreenSpot-Pro数据集配置文件\n")
        f.write("datasets:\n")
        for dataset in all_datasets:
            f.write(f"  - json_path: {dataset['json_path']}\n")
            f.write(f"    sampling_strategy: {dataset['sampling_strategy']}\n")
    
    print(f"已创建训练配置文件: {yaml_output}")
    
    # 创建测试配置文件列表
    test_datasets = [f"screenspot_pro_{os.path.splitext(os.path.basename(f))[0]}" for f in input_files]
    test_datasets_str = str(test_datasets).replace("'", "\"")
    
    # 输出使用说明
    print("\n使用说明:")
    print("1. 训练数据已转换为RefCOCO格式并保存")
    print("2. 要使用该数据集进行训练，请在训练脚本中指定:")
    print(f"   --dataset_name {os.path.abspath(yaml_output)}")
    print(f"   --image_root {os.path.abspath('.')}")
    print("3. 要使用该数据集进行测试，请修改test_rec_r1.py脚本中的:")
    print(f"   DATA_ROOT = \"{os.path.abspath(OUTPUT_DIR)}\"")
    print(f"   TEST_DATASETS = {test_datasets_str}")
    print(f"   IMAGE_ROOT = \"{os.path.abspath('.')}\"")
    print("4. 也可以仅测试特定应用，例如仅测试Android Studio:")
    print("   TEST_DATASETS = [\"screenspot_pro_android_studio_macos\"]")

if __name__ == "__main__":
    main() 