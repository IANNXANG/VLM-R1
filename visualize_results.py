#!/usr/bin/env python3
# 可视化评估结果中的边界框，同时显示模型输出和真实标签

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def visualize_results(results_file, output_dir, sample_limit=None, thickness=3, alpha=0.3, correct_only=False, incorrect_only=False):
    """
    在图像上绘制边界框并保存到新文件夹，展示模型预测结果和真实标签
    
    参数:
        results_file: 评估结果JSON文件路径
        output_dir: 输出目录
        sample_limit: 限制处理的样本数量（None表示处理所有样本）
        thickness: 边界框线条粗细
        alpha: 边界框填充区域的透明度
        correct_only: 仅处理正确预测的样本
        incorrect_only: 仅处理错误预测的样本
    """
    # 读取JSON结果数据
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # 提取准确率和结果
    accuracy = data.get('accuracy', 0)
    results = data.get('results', [])
    
    print(f"处理结果文件: {results_file}")
    print(f"模型准确率: {accuracy:.2f}%")
    print(f"样本总数: {len(results)}")
    
    # 处理限制条件
    if correct_only and incorrect_only:
        print("警告: correct_only 和 incorrect_only 不能同时为 True，将处理所有样本")
        correct_only = False
        incorrect_only = False
    
    if correct_only:
        results = [r for r in results if r.get('correct') == 1]
        print(f"仅处理正确预测样本，筛选后样本数: {len(results)}")
    
    if incorrect_only:
        results = [r for r in results if r.get('correct') == 0]
        print(f"仅处理错误预测样本，筛选后样本数: {len(results)}")
    
    # 限制样本数量
    if sample_limit is not None and sample_limit > 0:
        results = results[:sample_limit]
        print(f"限制处理样本数为: {sample_limit}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个样本
    success_count = 0
    for i, item in enumerate(results):
        try:
            # 获取图像路径和问题
            image_path = item['image']
            question = item['question']
            
            # 检查图像路径是否是绝对路径
            if not os.path.isabs(image_path):
                # 尝试不同的基础路径
                base_dirs = [
                    "/c22940/zy/code/VLM-R1",
                    "/c22940/zy/code/VLM-R1/data",
                    "/c22940/zy/code/VLM-R1/otherdata"
                ]
                
                for base_dir in base_dirs:
                    test_path = os.path.join(base_dir, image_path)
                    if os.path.exists(test_path):
                        image_path = test_path
                        break
            
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                continue
            
            # 获取边界框坐标
            ground_truth = item['ground_truth']  # 真实标签
            extracted_answer = item['extracted_answer']  # 模型输出
            is_correct = item['correct'] == 1  # 是否正确
            
            # 转换为整数坐标
            gt_x1, gt_y1, gt_x2, gt_y2 = [int(coord) for coord in ground_truth]
            pred_x1, pred_y1, pred_x2, pred_y2 = [int(coord) for coord in extracted_answer]
            
            # 创建边界框的副本以避免修改原图像
            img_with_bbox = img.copy()
            
            # 为真实标签绘制半透明填充矩形（绿色）
            overlay = img.copy()
            cv2.rectangle(overlay, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), -1)  # 绿色填充
            cv2.addWeighted(overlay, alpha, img_with_bbox, 1 - alpha, 0, img_with_bbox)
            
            # 为模型预测绘制半透明填充矩形（红色）
            overlay = img_with_bbox.copy()
            color = (0, 0, 255)  # 红色
            cv2.rectangle(overlay, (pred_x1, pred_y1), (pred_x2, pred_y2), color, -1)
            cv2.addWeighted(overlay, alpha, img_with_bbox, 1 - alpha, 0, img_with_bbox)
            
            # 绘制边界框
            cv2.rectangle(img_with_bbox, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), thickness)  # 绿色=真实标签
            cv2.rectangle(img_with_bbox, (pred_x1, pred_y1), (pred_x2, pred_y2), color, thickness)  # 红色=模型预测
            
            # 添加文本标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_with_bbox, "Ground Truth", (gt_x1, gt_y1 - 10), font, 0.5, (0, 255, 0), 2)
            label = "Prediction (Correct)" if is_correct else "Prediction (Wrong)"
            cv2.putText(img_with_bbox, label, (pred_x1, pred_y1 - 10), font, 0.5, color, 2)
            
            # 在图像底部添加问题文本
            # 将问题文本分成多行以适应图像宽度
            img_height, img_width = img_with_bbox.shape[:2]
            text_width = int(img_width - 20)  # 留出左右边距
            font_scale = 0.5
            font_thickness = 1
            line_height = 25
            
            # 创建一个扩展的图像，底部添加问题文本区域
            question_lines = []
            words = question.split()
            current_line = words[0]
            
            for word in words[1:]:
                test_line = current_line + " " + word
                # 估算文本宽度
                text_size = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]
                if text_size[0] <= text_width:
                    current_line = test_line
                else:
                    question_lines.append(current_line)
                    current_line = word
            
            if current_line:
                question_lines.append(current_line)
            
            # 计算需要的额外高度
            extra_height = len(question_lines) * line_height + 40  # 添加额外空间
            extended_img = np.zeros((img_height + extra_height, img_width, 3), dtype=np.uint8)
            # 设置背景颜色为黑色
            extended_img[:] = (0, 0, 0)
            # 复制原图
            extended_img[:img_height, :] = img_with_bbox
            
            # 添加问题文本
            for i, line in enumerate(question_lines):
                y_pos = img_height + 30 + i * line_height
                cv2.putText(extended_img, line, (10, y_pos), font, font_scale, (255, 255, 255), font_thickness)
            
            # 保存图像
            image_filename = os.path.basename(str(image_path))
            base_name, ext = os.path.splitext(image_filename)
            result_label = "correct" if is_correct else "wrong"
            output_path = os.path.join(output_dir, f"{base_name}_{result_label}{ext}")
            cv2.imwrite(output_path, extended_img)
            
            success_count += 1
            
            # 打印进度
            if (i + 1) % 10 == 0 or i + 1 == len(results):
                print(f"已处理: {i + 1}/{len(results)} 样本")
        
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
    
    print(f"完成可视化: 成功 {success_count}/{len(results)}")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='可视化评估结果中的边界框')
    parser.add_argument('--results', type=str, required=True, 
                        help='评估结果JSON文件路径')
    parser.add_argument('--output', type=str, default='visualized_results',
                        help='输出目录')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理的样本数量')
    parser.add_argument('--thickness', type=int, default=3, 
                        help='边界框线条粗细')
    parser.add_argument('--alpha', type=float, default=0.3, 
                        help='边界框填充区域的透明度')
    parser.add_argument('--correct-only', action='store_true',
                        help='仅处理正确预测的样本')
    parser.add_argument('--incorrect-only', action='store_true',
                        help='仅处理错误预测的样本')
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.results):
        print(f"错误: 结果文件不存在: {args.results}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理评估结果
    success_count = visualize_results(
        args.results, 
        args.output,
        args.limit,
        args.thickness, 
        args.alpha,
        args.correct_only,
        args.incorrect_only
    )
    
    print(f"\n总结:")
    print(f"成功处理样本: {success_count}")
    print(f"所有可视化图像已保存到: {os.path.abspath(args.output)}")
    print("\n用法示例:")
    print(f"1. 查看可视化图像: 打开 {os.path.abspath(args.output)} 文件夹")
    print("2. 仅查看错误预测: python visualize_results.py --results <结果文件> --incorrect-only")
    print("3. 限制处理样本数: python visualize_results.py --results <结果文件> --limit 20")

if __name__ == "__main__":
    main() 