#!/usr/bin/env python3
# coding=utf-8

import os
import json
import pandas as pd

# 定义需要提取的数据集和模型
datasets = ['screenspot_desktop', 'screenspot_mobile', 'screenspot_web']
steps = [0, 100]  # 0代表原始模型，100代表微调后的模型

results = []

# 遍历每个数据集和模型
for dataset in datasets:
    for step in steps:
        # 确定结果文件路径
        if step == 0:
            result_dir = f"logs/screenspot-original-model"
        else:
            result_dir = f"logs/screenspot-checkpoint-{step}"
        
        result_file = f"{result_dir}/screenspot_results_{dataset}_{step}.json"
        
        # 如果结果文件存在，读取并提取精度
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                
                # 提取准确率数据
                if 'accuracy' in data:
                    accuracy = data['accuracy'] * 100  # 转换为百分比
                    
                    # 添加到结果列表
                    model_name = "原始模型" if step == 0 else f"微调checkpoint-{step}"
                    results.append({
                        "模型": model_name,
                        "数据集": dataset,
                        "准确率(%)": round(accuracy, 2)  # 保留两位小数
                    })
                else:
                    print(f"警告: 在{result_file}中未找到准确率数据")
        else:
            print(f"警告: 未找到结果文件 {result_file}")

# 创建数据表格
if results:
    df = pd.DataFrame(results)
    
    # 按照数据集和模型排序
    df = df.sort_values(by=['数据集', '模型'])
    
    # 保存到CSV文件
    csv_file = "logs/screenspot_accuracy_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"结果已保存到 {csv_file}")
    
    # 打印结果表格
    print("\nScreenSpot评估结果摘要:")
    print(df.to_string(index=False))
else:
    print("未找到任何结果数据") 