#!/usr/bin/env python3
# 在图像上可视化边界框标注

import json
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# 输入目录和输出目录
CONVERTED_DATA_DIR = "converted_data"
OUTPUT_DIR = "visualized_images"

def visualize_bboxes(dataset_name, thickness=3, alpha=0.3):
    """
    在图像上绘制边界框并保存到新文件夹
    
    参数:
        dataset_name: 数据集名称 (screenspot_desktop, screenspot_mobile, screenspot_web)
        thickness: 边界框线条粗细
        alpha: 边界框填充区域的透明度
    """
    # 读取JSON数据
    json_file = os.path.join(CONVERTED_DATA_DIR, f"{dataset_name}.json")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"处理数据集: {dataset_name}, 共{len(data)}张图像")
    
    # 处理每张图像
    success_count = 0
    for i, item in enumerate(data):
        try:
            # 获取图像路径
            image_path = item['image']
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                continue
            
            # 获取边界框坐标
            bbox = item['solution']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 创建边界框的副本以避免修改原图像
            img_with_bbox = img.copy()
            
            # 绘制半透明填充矩形
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # 填充红色矩形
            cv2.addWeighted(overlay, alpha, img_with_bbox, 1 - alpha, 0, img_with_bbox)  # 添加透明度
            
            # 绘制边界框
            cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), thickness)  # 红色边界框
            
            # 保存图像
            image_filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(image_filename)
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_vis{ext}")
            cv2.imwrite(output_path, img_with_bbox)
            
            success_count += 1
            
            # 打印进度
            if (i + 1) % 50 == 0 or i + 1 == len(data):
                print(f"已处理: {i + 1}/{len(data)} 图像")
        
        except Exception as e:
            print(f"处理图像 {i} 时出错: {e}")
    
    print(f"完成数据集 {dataset_name} 的可视化: 成功 {success_count}/{len(data)}")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='在图像上可视化边界框标注')
    parser.add_argument('--datasets', nargs='+', default=['screenspot_desktop', 'screenspot_mobile', 'screenspot_web'],
                        help='要处理的数据集列表')
    parser.add_argument('--thickness', type=int, default=3, help='边界框线条粗细')
    parser.add_argument('--alpha', type=float, default=0.3, help='边界框填充区域的透明度')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理每个数据集
    total_success = 0
    total_images = 0
    
    for dataset_name in args.datasets:
        success_count = visualize_bboxes(dataset_name, args.thickness, args.alpha)
        
        # 获取数据集中的图像总数
        json_file = os.path.join(CONVERTED_DATA_DIR, f"{dataset_name}.json")
        with open(json_file, 'r') as f:
            data_count = len(json.load(f))
        
        total_success += success_count
        total_images += data_count
    
    print(f"\n总结:")
    print(f"成功标注图像: {total_success}/{total_images} ({total_success/total_images*100:.2f}%)")
    print(f"所有标注图像已保存到: {os.path.abspath(OUTPUT_DIR)}")
    print("\n用法示例:")
    print(f"1. 查看标注图像: 打开 {os.path.abspath(OUTPUT_DIR)} 文件夹")
    print("2. 修改标注参数: 使用 --thickness 和 --alpha 参数调整标注样式")
    print("   例如: python visualize_bboxes.py --thickness 5 --alpha 0.5")
    print("3. 仅处理特定数据集: 使用 --datasets 参数")
    print("   例如: python visualize_bboxes.py --datasets screenspot_desktop")

if __name__ == "__main__":
    main() 