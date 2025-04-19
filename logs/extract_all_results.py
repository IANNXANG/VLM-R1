#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用结果提取脚本，用于从logs目录下的各个实验结果中提取准确率，并生成CSV摘要文件。
使用方法: python extract_all_results.py

特点:
1. 支持多类型实验目录结构
2. 自动识别数据集和模型检查点
3. 生成每个类别的准确率摘要CSV文件
4. 排序顺序: baseline -> original-model -> 按序号排序的checkpoints
"""

import os
import json
import re
import csv
import glob
from collections import defaultdict

# 基础配置
BASE_DIR = "logs"
OUTPUT_DIR = "logs"  # 输出CSV文件目录

# =====================================================================
# [如何添加新的实验类型]
# 要添加新的实验类型，请按以下步骤操作：
# 1. 在EXPERIMENT_TYPES中添加新的实验目录名和对应结果文件前缀
# 2. 在EXPERIMENT_CSV_NAMES中添加CSV输出文件的命名规则
# 3. 如果新实验有特殊的数据集命名需要统一，在DATASET_MAPPING中添加映射关系
# =====================================================================

# 实验类型定义 (目录名 -> 结果文件前缀)
EXPERIMENT_TYPES = {
    "Qwen2.5-VL-7B-GRPO-REC-lora": "rec_results_",  # REC任务目录 -> REC结果文件前缀
    "Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click": "click_results_",  # ScreenSpot点击任务目录 -> 点击结果文件前缀

    # [添加新实验]
    # 添加格式: "logs中的目录名": "结果文件前缀_",
    #
    # 示例1 - 添加目标检测(OD)任务:
    # "od": "od_results_",  # OD任务目录 -> 目标检测结果文件前缀
    #
    # 示例2 - 添加视觉问答(VQA)任务:
    # "vqa": "vqa_results_",  # VQA任务目录 -> 视觉问答结果文件前缀
    # 
    # 注意: 结果文件前缀必须与实际JSON文件名前缀一致
}

# 额外的实验名称映射，用于CSV文件命名
EXPERIMENT_CSV_NAMES = {
    "Qwen2.5-VL-7B-GRPO-REC-lora": "rec_lora",
    "Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click": "screenspot_click",

    # [添加新实验的CSV输出文件命名]
    # 添加格式: "logs中的目录名": "输出CSV文件名前缀",
    #
    # 示例1 - 为OD任务指定CSV文件名:
    # "od": "object_detection",  # 将生成 object_detection_accuracy.csv
    #
    # 示例2 - 为VQA任务指定CSV文件名:
    # "vqa": "visual_qa",  # 将生成 visual_qa_accuracy.csv
    #
    # 如果不添加映射，将默认使用目录名作为CSV文件名前缀
}

# 数据集标准化映射
DATASET_MAPPING = {
    # REC任务标准数据集名称
    "refcoco_val_qwen2_5vl_7b_instruct_baseline": "refcoco_val",
    "refcocop_val_qwen2_5vl_7b_instruct_baseline": "refcocop_val",
    "refcocog_val_qwen2_5vl_7b_instruct_baseline": "refcocog_val",
    # ScreenSpot标准数据集名称
    "screenspot_desktop_qwen2_5vl_7b_instruct_baseline": "screenspot_desktop",
    "screenspot_mobile_qwen2_5vl_7b_instruct_baseline": "screenspot_mobile", 
    "screenspot_web_qwen2_5vl_7b_instruct_baseline": "screenspot_web",

    # [添加新的数据集名称标准化映射]
    # 添加格式: "原始数据集名称": "标准化后的数据集名称",
    #
    # 目的: 确保不同实验中相同的数据集使用统一的名称，以便在CSV中正确对齐
    #
    # 示例1 - COCO数据集的不同命名统一:
    # "coco_val_modelA": "coco_val",  # 将modelA使用的coco_val映射到标准名称
    # "coco_validation": "coco_val",  # 将不同命名方式的数据集映射到标准名称
    #
    # 示例2 - 处理带有模型名称的数据集:
    # "imagenet_val_baseline": "imagenet_val",  # 将baseline模型的验证集映射到标准名称
}

def extract_checkpoint_number(path):
    """从路径中提取检查点数字"""
    if "baseline" in path.lower():
        return -1  # 使用-1表示baseline，这样排序时会出现在最前面
    if "original-model" in path.lower():
        return 0  # 使用0表示原始模型
    
    # 尝试提取checkpoint-xxx格式的数字
    match = re.search(r'checkpoint-(\d+)', path.lower())
    if match:
        return int(match.group(1))
    
    # 尝试提取文件名中的数字（例如xxx_100.json）
    match = re.search(r'_(\d+)\.json$', path.lower())
    if match:
        return int(match.group(1))
    
    return 999999  # 无法识别的格式，放在最后

    # [如何支持其他检查点命名格式]
    # 如果新实验使用不同的检查点命名方式，可在此函数中添加相应的正则表达式匹配逻辑
    # 例如，对于格式为"model_step1000.json"的文件:
    # match = re.search(r'model_step(\d+)\.json$', path.lower())
    # if match:
    #     return int(match.group(1))

def get_model_name(path):
    """从路径获取模型名称，用于CSV中的标识"""
    if "baseline" in path.lower():
        return "baseline"
    if "original-model" in path.lower():
        return "original-model"
    
    # 尝试提取checkpoint-xxx格式
    match = re.search(r'checkpoint-(\d+)', path.lower())
    if match:
        return f"checkpoint-{match.group(1)}"
    
    # 获取目录名作为模型名称
    basename = os.path.basename(path)
    return basename

    # [如何支持其他模型命名格式]
    # 如果新实验使用不同的模型命名方式，可在此函数中添加相应的处理逻辑
    # 例如，对于格式为"model_stepN"的目录:
    # match = re.search(r'model_step(\d+)', path.lower())
    # if match:
    #     return f"step-{match.group(1)}"

def find_result_files(experiment_dir, result_prefix):
    """在指定目录中查找所有结果文件"""
    result_files = []
    
    # 场景1: 实验目录直接包含结果文件
    direct_files = glob.glob(os.path.join(experiment_dir, f"{result_prefix}*.json"))
    if direct_files:
        result_files.extend(direct_files)
    
    # 场景2: 实验目录包含子目录，子目录中包含结果文件
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if os.path.isdir(item_path):
            # 在子目录中查找结果文件
            sub_files = glob.glob(os.path.join(item_path, f"{result_prefix}*.json"))
            result_files.extend(sub_files)
            
            # 如果子目录中没有找到文件，递归搜索更深一层
            if not sub_files:
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        sub_sub_files = glob.glob(os.path.join(subitem_path, f"{result_prefix}*.json"))
                        result_files.extend(sub_sub_files)
    
    return result_files

    # [如何支持不同的目录结构]
    # 如果新实验使用不同的目录结构，例如更深的嵌套或不同的组织方式，
    # 可以修改此函数以适应特定的目录结构。当前实现支持最多三层目录结构。

def normalize_dataset_name(dataset_name):
    """标准化数据集名称，确保不同实验使用相同的数据集标识符"""
    # 使用映射表进行标准化
    return DATASET_MAPPING.get(dataset_name, dataset_name)

def extract_dataset_name(filename, result_prefix):
    """从结果文件名中提取数据集名称"""
    basename = os.path.basename(filename)
    
    # 去掉前缀和.json后缀
    name_part = basename[len(result_prefix):-5]
    
    # 如果名称中包含下划线，可能是格式: dataset_checkpoint.json 或 dataset_name_checkpoint.json
    parts = name_part.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        # 最后一部分是数字，去掉它
        dataset = '_'.join(parts[:-1])
    else:
        dataset = name_part
    
    # 标准化数据集名称
    return normalize_dataset_name(dataset)

    # [如何支持其他结果文件命名格式]
    # 如果新实验使用不同的文件命名约定，可以修改此函数的逻辑
    # 当前支持格式:
    # - prefix_dataset.json
    # - prefix_dataset_123.json (123为检查点编号)
    #
    # 例如，对于格式为"prefix_model_dataset.json"的文件:
    # parts = name_part.split('_')
    # if len(parts) >= 3:  # 至少有model和dataset两部分
    #     return normalize_dataset_name(parts[-1])  # 取最后一部分作为数据集名称

def extract_accuracy(result_file):
    """从结果文件中提取准确率"""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            # 大多数结果文件都有顶级accuracy字段
            if 'accuracy' in data:
                return float(data['accuracy'])
            
            # 如果没有直接的accuracy字段，尝试计算结果列表中的正确率
            if 'results' in data:
                correct = sum(1 for r in data['results'] if r.get('correct', 0) == 1)
                total = len(data['results'])
                if total > 0:
                    return 100.0 * correct / total
    except (json.JSONDecodeError, IOError) as e:
        print(f"无法解析文件 {result_file}: {e}")
    
    return None

    # [如何支持不同的结果文件格式]
    # 如果新实验的结果文件有不同的格式或字段名称，请修改此函数
    # 例如，如果准确率存储在"score"字段中:
    # if 'score' in data:
    #     return float(data['score']) * 100  # 如果score是0-1之间的值，转换为百分比
    #
    # 或者如果正确/错误存储在不同的字段:
    # if 'evaluation' in data:
    #     correct = sum(1 for r in data['evaluation'] if r.get('is_correct', False))
    #     total = len(data['evaluation'])
    #     if total > 0:
    #         return 100.0 * correct / total

def process_experiment(experiment_name, result_prefix):
    """处理单个实验目录，提取所有结果"""
    experiment_dir = os.path.join(BASE_DIR, experiment_name)
    if not os.path.exists(experiment_dir):
        print(f"实验目录不存在: {experiment_dir}")
        return None
    
    # 查找所有结果文件
    result_files = find_result_files(experiment_dir, result_prefix)
    if not result_files:
        print(f"在目录 {experiment_dir} 中未找到 {result_prefix}*.json 结果文件")
        return None
    
    # 整理结果
    results = []
    dataset_names = set()
    
    for result_file in result_files:
        # 提取数据集名称
        dataset = extract_dataset_name(result_file, result_prefix)
        dataset_names.add(dataset)
        
        # 获取模型/检查点信息
        model_path = os.path.dirname(result_file)
        checkpoint_num = extract_checkpoint_number(result_file if "baseline" in result_file or "original-model" in result_file else model_path)
        model_name = get_model_name(model_path)
        
        # 提取准确率
        accuracy = extract_accuracy(result_file)
        if accuracy is not None:
            results.append({
                'checkpoint_num': checkpoint_num,
                'model_name': model_name,
                'dataset': dataset,
                'accuracy': accuracy,
                'file': result_file  # 用于调试
            })
            print(f"读取: {model_name} 在 {dataset} 上的准确率: {accuracy:.2f}%")
    
    return {
        'results': results,
        'datasets': sorted(list(dataset_names))
    }

def write_csv(experiment_name, data):
    """将结果写入CSV文件"""
    if not data or not data['results']:
        print(f"没有要写入的数据: {experiment_name}")
        return
    
    # 获取CSV文件名
    csv_name = EXPERIMENT_CSV_NAMES.get(experiment_name, experiment_name)
    csv_path = os.path.join(OUTPUT_DIR, f"{csv_name}_accuracy.csv")
    
    # 按照checkpoint_num排序结果
    # 先按checkpoint_num排序，然后是model_name
    sorted_results = sorted(data['results'], key=lambda x: (x['checkpoint_num'], x['model_name']))
    
    # 按model_name分组
    model_results = defaultdict(dict)
    for result in sorted_results:
        model_name = result['model_name']
        dataset = result['dataset']
        model_results[model_name][dataset] = result['accuracy']
    
    # 为每个模型计算平均值
    for model_name in model_results:
        accuracies = [acc for acc in model_results[model_name].values() if acc is not None]
        if accuracies:
            model_results[model_name]['average'] = sum(accuracies) / len(accuracies)
    
    # 写入CSV
    fieldnames = ['checkpoint'] + data['datasets'] + ['average']
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 确保按照排序写入（baseline -> original-model -> 检查点）
        model_names_sorted = sorted(model_results.keys(), 
                                   key=lambda x: (-1 if x == "baseline" else 
                                                0 if x == "original-model" else 
                                                int(x.split('-')[-1]) if 'checkpoint-' in x else 999))
        
        for model_name in model_names_sorted:
            row = {'checkpoint': model_name}
            # 填充数据集准确率
            for dataset in data['datasets']:
                if dataset in model_results[model_name]:
                    row[dataset] = f"{model_results[model_name][dataset]:.2f}"
                else:
                    row[dataset] = "N/A"
            
            # 添加平均准确率
            if 'average' in model_results[model_name]:
                row['average'] = f"{model_results[model_name]['average']:.2f}"
            else:
                row['average'] = "N/A"
                
            writer.writerow(row)
    
    print(f"结果已保存到 {csv_path}")

    # [如何自定义CSV输出格式]
    # 如果需要修改CSV格式，例如添加更多统计信息或自定义列，请修改此函数
    # 例如添加中位数、最高值等统计数据，或自定义列排序

def main():
    """主函数：处理所有实验类型"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理每种实验类型
    for experiment_name, result_prefix in EXPERIMENT_TYPES.items():
        print(f"处理实验: {experiment_name} (前缀: {result_prefix})")
        data = process_experiment(experiment_name, result_prefix)
        if data:
            write_csv(experiment_name, data)
            print(f"成功处理实验: {experiment_name}")
        else:
            print(f"处理实验失败: {experiment_name}")
    
    print("所有实验处理完毕！")

    # [如何添加对特定实验的处理]
    # 如果只想处理特定的实验类型而非全部，可修改main函数:
    # experiment_to_process = ["rec", "screenspot"]  # 只处理这些实验
    # for experiment_name in experiment_to_process:
    #     if experiment_name in EXPERIMENT_TYPES:
    #         result_prefix = EXPERIMENT_TYPES[experiment_name]
    #         print(f"处理实验: {experiment_name} (前缀: {result_prefix})")
    #         ...

if __name__ == "__main__":
    main() 