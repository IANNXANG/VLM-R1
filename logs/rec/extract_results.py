#!/usr/bin/env python3
# 提取所有检查点在三个数据集上的准确率
import os
import json
import glob
import csv
import re

# 输出CSV文件路径
OUTPUT_CSV = "logs/rec/accuracy_summary.csv"

# 定义数据集列表
DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']

# 定义baseline模型目录
BASELINE_DIR = "logs/rec/baseline-7b-instruct"

def extract_checkpoint_number(path):
    """从路径中提取检查点数字"""
    if "original-model" in path:
        return 0
    if "baseline" in path:
        return -1  # 使用-1表示baseline，这样排序时会出现在最前面
    match = re.search(r'checkpoint-(\d+)', path)
    if match:
        return int(match.group(1))
    return -2  # 无法识别的格式

def main():
    # 收集所有检查点目录
    checkpoint_dirs = []
    
    # 添加baseline目录（如果存在）
    if os.path.exists(BASELINE_DIR):
        checkpoint_dirs.append(BASELINE_DIR)
    
    # 添加其他检查点目录
    for item in os.listdir("logs/rec"):
        full_path = os.path.join("logs/rec", item)
        if os.path.isdir(full_path) and (item.startswith("checkpoint-") or item == "original-model"):
            checkpoint_dirs.append(full_path)
    
    # 如果没有找到检查点目录，尝试直接查找结果文件
    if not checkpoint_dirs:
        print("没有找到检查点目录，尝试直接查找结果文件...")
        result_files = glob.glob("logs/rec/rec_results_*_*.json") + glob.glob("logs/rec/baseline-7b-instruct/rec_results_*.json")
        results = []
        
        for file in result_files:
            # 检查是否是baseline文件
            is_baseline = "baseline" in file
            
            if is_baseline:
                checkpoint = -1
                match = re.search(r'rec_results_(\w+)_', os.path.basename(file))
                if match:
                    dataset = match.group(1)
            else:
                match = re.search(r'rec_results_(\w+)_\w+_(\d+)\.json', os.path.basename(file))
                if match:
                    dataset = match.group(1)
                    checkpoint = int(match.group(2))
                    
            if match:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        accuracy = data.get('accuracy', 0)
                        results.append({
                            'checkpoint': checkpoint,
                            'dataset': dataset,
                            'accuracy': accuracy
                        })
                except:
                    print(f"无法解析文件: {file}")
        
        # 组织数据并写入CSV
        if results:
            # 按检查点号排序
            checkpoints = sorted(list(set([r['checkpoint'] for r in results])))
            
            with open(OUTPUT_CSV, 'w', newline='') as csvfile:
                fieldnames = ['checkpoint'] + DATASETS
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for checkpoint in checkpoints:
                    row = {'checkpoint': "baseline" if checkpoint == -1 else checkpoint}
                    for dataset in DATASETS:
                        for r in results:
                            if r['checkpoint'] == checkpoint and r['dataset'] == dataset:
                                row[dataset] = f"{r['accuracy']:.2f}"
                                break
                        if dataset not in row:
                            row[dataset] = "N/A"  # 缺失数据
                    writer.writerow(row)
                
            print(f"结果已保存到 {OUTPUT_CSV}")
            return
            
        print("无法找到结果文件，退出。")
        return
    
    # 收集所有结果
    results = []
    
    # 处理baseline目录
    if os.path.exists(BASELINE_DIR):
        for dataset in DATASETS:
            # 尝试不同的文件命名模式
            patterns = [
                f"{BASELINE_DIR}/rec_results_{dataset}_*.json",
                f"{BASELINE_DIR}/rec_results_{dataset}.json"
            ]
            
            result_file = None
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    result_file = matches[0]
                    break
            
            if result_file and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    try:
                        data = json.load(f)
                        accuracy = data.get('accuracy', 0)
                        results.append({
                            'checkpoint': -1,  # 使用-1表示baseline
                            'dataset': dataset,
                            'accuracy': accuracy
                        })
                        print(f"已读取 baseline 在 {dataset} 上的准确率: {accuracy:.2f}%")
                    except json.JSONDecodeError:
                        print(f"无法解析文件: {result_file}")
    
    # 处理其他检查点目录
    for checkpoint_dir in checkpoint_dirs:
        if "baseline" in checkpoint_dir:
            continue  # 跳过baseline目录，因为已经处理过了
            
        checkpoint_num = extract_checkpoint_number(checkpoint_dir)
        
        # 查找该检查点目录下的所有结果文件
        for dataset in DATASETS:
            # 尝试不同的文件命名模式
            patterns = [
                f"{checkpoint_dir}/rec_results_{dataset}_{checkpoint_num}.json",
                f"{checkpoint_dir}/rec_results_{dataset}_*.json",
                f"{checkpoint_dir}/rec_results_{dataset}.json"
            ]
            
            result_file = None
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    result_file = matches[0]
                    break
            
            if result_file and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    try:
                        data = json.load(f)
                        accuracy = data.get('accuracy', 0)
                        results.append({
                            'checkpoint': checkpoint_num,
                            'dataset': dataset,
                            'accuracy': accuracy
                        })
                        print(f"已读取 checkpoint-{checkpoint_num} 在 {dataset} 上的准确率: {accuracy:.2f}%")
                    except json.JSONDecodeError:
                        print(f"无法解析文件: {result_file}")
            else:
                print(f"未找到 checkpoint-{checkpoint_num} 在 {dataset} 上的结果文件")
    
    # 如果没有找到结果，尝试搜索整个logs/rec目录
    if not results:
        print("在检查点目录中未找到结果文件，尝试搜索整个logs/rec目录...")
        result_files = glob.glob("logs/rec/**/rec_results_*.json", recursive=True)
        
        for file in result_files:
            # 检查是否是baseline文件
            is_baseline = "baseline" in file
            
            # 尝试从文件名提取信息
            for dataset in DATASETS:
                if dataset in file:
                    if is_baseline:
                        checkpoint = -1
                    else:
                        match = re.search(r'(\d+)\.json$', file)
                        checkpoint = 0
                        if match:
                            checkpoint = int(match.group(1))
                        elif "original-model" in file:
                            checkpoint = 0
                    
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            accuracy = data.get('accuracy', 0)
                            results.append({
                                'checkpoint': checkpoint,
                                'dataset': dataset,
                                'accuracy': accuracy
                            })
                            print(f"已读取 checkpoint-{checkpoint} 在 {dataset} 上的准确率: {accuracy:.2f}%")
                    except:
                        print(f"无法解析文件: {file}")
    
    # 按检查点号排序
    results.sort(key=lambda x: x['checkpoint'])
    
    # 组织数据以便写入CSV
    checkpoints = sorted(list(set([r['checkpoint'] for r in results])))
    
    # 写入CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['checkpoint'] + DATASETS + ['average']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for checkpoint in checkpoints:
            if checkpoint == -1:
                row = {'checkpoint': "baseline"}
            else:
                row = {'checkpoint': checkpoint}
                
            total_accuracy = 0
            count = 0
            
            for dataset in DATASETS:
                found = False
                for r in results:
                    if r['checkpoint'] == checkpoint and r['dataset'] == dataset:
                        row[dataset] = f"{r['accuracy']:.2f}"
                        total_accuracy += r['accuracy']
                        count += 1
                        found = True
                        break
                if not found:
                    row[dataset] = "N/A"
            
            # 计算平均准确率
            if count > 0:
                row['average'] = f"{total_accuracy / count:.2f}"
            else:
                row['average'] = "N/A"
                
            writer.writerow(row)
    
    print(f"结果已保存到 {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 