#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用准确率折线图绘制脚本
自动绘制所有accuracy.csv文件的折线图

使用方法:
python plot_accuracy.py [csv_file1] [csv_file2] ...
如果不提供参数，则自动处理logs目录下所有*_accuracy.csv文件
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
from matplotlib.ticker import MaxNLocator

# 不使用中文字体配置，避免乱码问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 配置图表样式
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True

# 为不同的数据集设置不同的颜色和标记
DATASET_STYLES = {
    # REC相关数据集
    "refcoco_val": {"color": "blue", "marker": "o", "label": "RefCOCO Val"},
    "refcocop_val": {"color": "green", "marker": "s", "label": "RefCOCO+ Val"},
    "refcocog_val": {"color": "red", "marker": "^", "label": "RefCOCOg Val"},
    "average": {"color": "black", "marker": "*", "label": "Average", "linewidth": 2},
    
    # ScreenSpot相关数据集
    "screenspot_desktop": {"color": "purple", "marker": "o", "label": "Desktop"},
    "screenspot_mobile": {"color": "orange", "marker": "s", "label": "Mobile"},
    "screenspot_web": {"color": "brown", "marker": "^", "label": "Web"}
}

# 实验类型配置 (自动从文件名生成)
EXPERIMENT_CONFIG = {
    # 会根据CSV文件自动生成
}

def generate_experiment_config(csv_files):
    """根据CSV文件名生成实验配置"""
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).replace("_accuracy.csv", "")
        output_path = os.path.join(os.path.dirname(csv_file), f"{base_name}_accuracy_plot.png")
        
        # 根据文件名设置标题
        if "rec_lora" in base_name:
            title = "REC Task Performance"
        elif "screenspot_click" in base_name:
            title = "ScreenSpot Click Task Performance"
        elif "screenspot_desktop" in base_name or "rec_screenspot" in base_name:
            title = "ScreenSpot Desktop Task Performance"
        else:
            # 将下划线转换为空格, 首字母大写
            title = " ".join(word.capitalize() for word in base_name.split("_")) + " Performance"
        
        EXPERIMENT_CONFIG[base_name] = {
            "title": title,
            "output": output_path,
            "y_label": "Accuracy (%)"
        }

def extract_checkpoint_number(checkpoint_str):
    """从checkpoint字符串中提取数字，用于正确排序"""
    if checkpoint_str == "baseline":
        return -1  # baseline放在最前面
    if checkpoint_str == "original-model":
        return 0   # original-model放在baseline之后
    
    match = re.search(r'checkpoint-(\d+)', checkpoint_str)
    if match:
        return int(match.group(1))
    
    return 999  # 无法识别的格式放在最后

def format_xtick_label(label):
    """格式化X轴标签"""
    if label == "baseline":
        return "baseline"
    if label == "original-model":
        return "original (0)"
    
    # 对于checkpoint-xxx格式，简化为数字
    match = re.search(r'checkpoint-(\d+)', label)
    if match:
        return match.group(1)
    
    return label

def plot_csv(csv_file):
    """为单个CSV文件创建折线图"""
    # 确保配置已生成
    if not EXPERIMENT_CONFIG:
        generate_experiment_config([csv_file])
        
    # 获取实验配置
    file_basename = os.path.basename(csv_file).replace("_accuracy.csv", "")
    plot_config = EXPERIMENT_CONFIG.get(file_basename, {
        "title": f"{file_basename.replace('_', ' ').title()} Performance",
        "output": os.path.join(os.path.dirname(csv_file), f"{file_basename}_accuracy_plot.png"),
        "y_label": "Accuracy (%)"
    })
    
    # 读取CSV
    df = pd.read_csv(csv_file)
    
    # 转换所有非数值列为数值
    for col in df.columns:
        if col != 'checkpoint':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 按检查点排序
    df['checkpoint_num'] = df['checkpoint'].apply(extract_checkpoint_number)
    df = df.sort_values('checkpoint_num')
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 获取所有数据列(排除checkpoint和checkpoint_num)
    data_columns = [col for col in df.columns if col not in ['checkpoint', 'checkpoint_num']]
    
    # 绘制每个数据集的折线
    for col in data_columns:
        # 跳过缺失值过多的列
        if df[col].isna().sum() > len(df) / 2:
            print(f"Skipping column {col} due to too many missing values")
            continue
        
        style = DATASET_STYLES.get(col, {"color": "gray", "marker": ".", "label": col})
        plt.plot(df['checkpoint_num'], df[col], 
                 marker=style.get("marker", "o"),
                 color=style.get("color", "blue"),
                 label=style.get("label", col),
                 linewidth=style.get("linewidth", 1.5))
    
    # 设置x轴标签，确保baseline和original位置正确
    formatted_labels = [format_xtick_label(label) for label in df['checkpoint']]
    plt.xticks(df['checkpoint_num'], formatted_labels, rotation=45, ha='right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加图表标题和轴标签
    plt.title(plot_config["title"], fontsize=16)
    plt.xlabel("Checkpoint", fontsize=14)
    plt.ylabel(plot_config["y_label"], fontsize=14)
    
    # 添加图例
    plt.legend(loc='best', fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围 - 通常准确率介于0-100之间
    y_min = max(0, df[data_columns].min().min() - 5)  # 最小值再减5，但不小于0
    y_max = min(100, df[data_columns].max().max() + 5)  # 最大值再加5，但不超过100
    plt.ylim(y_min, y_max)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(plot_config["output"], dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {plot_config['output']}")
    
    # 关闭图表
    plt.close()

def find_csv_files():
    """查找logs目录下所有accuracy.csv文件"""
    logs_dir = "/c22940/zy/code/VLM-R1/logs"  # 固定logs目录
    return glob.glob(os.path.join(logs_dir, "*_accuracy.csv"))

def main():
    """主函数：处理所有CSV文件并生成图表"""
    # 如果命令行提供了CSV文件，则处理这些文件
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
        print(f"Processing {len(csv_files)} specified CSV files")
    else:
        # 否则，查找所有CSV文件
        csv_files = find_csv_files()
        print(f"Found {len(csv_files)} CSV files to process")
    
    # 生成实验配置
    generate_experiment_config(csv_files)
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Processing CSV file: {csv_file}")
            plot_csv(csv_file)
        else:
            print(f"Error: CSV file not found {csv_file}")

if __name__ == "__main__":
    main() 