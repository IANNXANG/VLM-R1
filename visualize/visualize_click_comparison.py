#!/usr/bin/env python3
# 可视化比较两个模型的点击任务评估结果

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def visualize_comparison(tuned_results_file, baseline_results_file, output_dir, sample_limit=None, point_radius=5, thickness=3, alpha=0.3, correct_only=False, incorrect_only=False, at_least_one_wrong=False):
    """
    在图像上绘制不同模型的预测结果，并保存到新文件夹
    
    参数:
        tuned_results_file: 训练后模型的评估结果JSON文件路径
        baseline_results_file: baseline模型的评估结果JSON文件路径
        output_dir: 输出目录
        sample_limit: 限制处理的样本数量（None表示处理所有样本）
        point_radius: 点的半径大小
        thickness: 边界框线条粗细
        alpha: 边界框填充区域的透明度
        correct_only: 仅处理正确预测的样本
        incorrect_only: 仅处理错误预测的样本
        at_least_one_wrong: 仅处理至少有一个模型预测错误的样本
    """
    # 读取训练后模型JSON结果数据
    with open(tuned_results_file, 'r') as f:
        tuned_data = json.load(f)
    
    # 读取baseline模型JSON结果数据
    with open(baseline_results_file, 'r') as f:
        baseline_data = json.load(f)
    
    # 提取准确率和结果
    tuned_accuracy = tuned_data.get('accuracy', 0)
    tuned_results = tuned_data.get('results', [])
    
    baseline_accuracy = baseline_data.get('accuracy', 0)
    baseline_results = baseline_data.get('results', [])
    
    print(f"处理训练后模型结果文件: {tuned_results_file}")
    print(f"训练后模型准确率: {tuned_accuracy:.2f}%")
    print(f"训练后模型样本总数: {len(tuned_results)}")
    
    print(f"处理baseline模型结果文件: {baseline_results_file}")
    print(f"Baseline模型准确率: {baseline_accuracy:.2f}%")
    print(f"Baseline模型样本总数: {len(baseline_results)}")
    
    # 创建映射以便根据图像路径快速查找baseline结果
    baseline_map = {}
    for item in baseline_results:
        key = (item['image'], item['question'])  # 使用图像路径和问题作为组合键
        baseline_map[key] = item
    
    # 处理限制条件
    if correct_only and incorrect_only:
        print("警告: correct_only 和 incorrect_only 不能同时为 True，将处理所有样本")
        correct_only = False
        incorrect_only = False
    
    
    if correct_only:
        tuned_results = [r for r in tuned_results if r.get('correct') == 1]
        print(f"仅处理训练后模型正确预测样本，筛选后样本数: {len(tuned_results)}")
    
    if incorrect_only:
        tuned_results = [r for r in tuned_results if r.get('correct') == 0]
        print(f"仅处理训练后模型错误预测样本，筛选后样本数: {len(tuned_results)}")
        
    if at_least_one_wrong:
        # 创建筛选后的结果列表
        filtered_results = []
        for tuned_item in tuned_results:
            image_path = tuned_item['image']
            question = tuned_item['question']
            key = (image_path, question)
            
            if key in baseline_map:
                baseline_item = baseline_map[key]
                # 如果至少有一个模型预测错误
                if tuned_item['correct'] == 0 or baseline_item['correct'] == 0:
                    filtered_results.append(tuned_item)
        
        tuned_results = filtered_results
        print(f"仅处理至少有一个模型预测错误的样本，筛选后样本数: {len(tuned_results)}")
    
    # 限制样本数量
    if sample_limit is not None and sample_limit > 0:
        tuned_results = tuned_results[:sample_limit]
        print(f"限制处理样本数为: {sample_limit}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个样本
    success_count = 0
    both_correct = 0
    both_wrong = 0
    tuned_better = 0
    baseline_better = 0
    
    for i, tuned_item in enumerate(tuned_results):
        try:
            # 获取图像路径和问题
            image_path = tuned_item['image']
            question = tuned_item['question']
            
            # 查找对应的baseline结果
            baseline_item = baseline_map.get((image_path, question))
            if not baseline_item:
                print(f"警告: 在baseline结果中找不到图像 {image_path} 和问题 {question}")
                continue
            
            # 检查图像路径是否是绝对路径
            if not os.path.isabs(image_path):
                # 使用正确的图片路径
                base_image_path = "/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2/images/screenspotv2_image"
                image_filename = os.path.basename(image_path)
                image_path = os.path.join(base_image_path, image_filename)
                
                # 如果上面的路径不存在，尝试其他可能的路径
                if not os.path.exists(image_path):
                    base_dirs = [
                        "/c22940/zy/code/VLM-R1",
                        "/c22940/zy/code/VLM-R1/otherdata",
                        "/c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2"
                    ]
                    
                    for base_dir in base_dirs:
                        test_path = os.path.join(base_dir, tuned_item['image'])
                        if os.path.exists(test_path):
                            image_path = test_path
                            break
            
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                continue
            
            # 获取边界框坐标和点击坐标
            ground_truth = tuned_item['ground_truth']  # 真实标签（边界框）
            tuned_answer = tuned_item['extracted_answer']  # 训练后模型输出（点击坐标）
            baseline_answer = baseline_item['extracted_answer']  # baseline模型输出（点击坐标）
            
            # 注释掉调试输出
            #print(f"处理图像 {image_path}:")
            #print(f"  Ground Truth: {ground_truth}")
            #print(f"  Tuned Model答案: {tuned_answer}")
            #print(f"  Baseline答案: {baseline_answer}")
            
            tuned_correct = tuned_item['correct'] == 1  # 训练后模型是否正确
            baseline_correct = baseline_item['correct'] == 1  # baseline模型是否正确
            
            # 统计结果差异
            if tuned_correct and baseline_correct:
                both_correct += 1
            elif not tuned_correct and not baseline_correct:
                both_wrong += 1
            elif tuned_correct and not baseline_correct:
                tuned_better += 1
            elif not tuned_correct and baseline_correct:
                baseline_better += 1
            
            # 转换为整数坐标
            gt_x1, gt_y1, gt_x2, gt_y2 = [int(coord) for coord in ground_truth]
            tuned_x, tuned_y = [int(coord) for coord in tuned_answer]
            baseline_x, baseline_y = [int(coord) for coord in baseline_answer]
            
            # 创建边界框的副本以避免修改原图像
            img_with_bbox = img.copy()
            
            # 为真实标签绘制半透明填充矩形（绿色）
            overlay = img.copy()
            cv2.rectangle(overlay, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), -1)  # 绿色填充
            cv2.addWeighted(overlay, alpha, img_with_bbox, 1 - alpha, 0, img_with_bbox)
            
            # 绘制边界框（绿色）
            cv2.rectangle(img_with_bbox, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), thickness)
            
            # 绘制训练后模型点击点（红色）
            cv2.circle(img_with_bbox, (tuned_x, tuned_y), point_radius, (0, 0, 255), -1)
            
            # 绘制baseline模型点击点（橙色）
            cv2.circle(img_with_bbox, (baseline_x, baseline_y), point_radius, (0, 165, 255), -1)
            
            # 添加文本标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            gt_coord_str = f"Ground Truth [{gt_x1},{gt_y1},{gt_x2},{gt_y2}]"
            cv2.putText(img_with_bbox, gt_coord_str, (gt_x1, gt_y1 - 10), font, 0.5, (0, 255, 0), 2)
            
            tuned_label = "Tuned Model"
            if tuned_correct:
                tuned_label += " (Correct)"
            else:
                tuned_label += " (Wrong)"
            tuned_label += f" [{tuned_x},{tuned_y}]"
            cv2.putText(img_with_bbox, tuned_label, (tuned_x, tuned_y - 10), font, 0.5, (0, 0, 255), 2)
            
            baseline_label = "Baseline"
            if baseline_correct:
                baseline_label += " (Correct)"
            else:
                baseline_label += " (Wrong)"
            baseline_label += f" [{baseline_x},{baseline_y}]"
            cv2.putText(img_with_bbox, baseline_label, (baseline_x, baseline_y + 20), font, 0.5, (0, 165, 255), 2)
            
            # 在图像底部添加问题文本和准确率比较
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
            
            # 移除准确率比较信息
            # 计算需要的额外高度 (只包含问题文本)
            extra_height = len(question_lines) * line_height + 40  # 添加额外空间
            extended_img = np.zeros((img_height + extra_height, img_width, 3), dtype=np.uint8)
            # 设置背景颜色为黑色
            extended_img[:] = (0, 0, 0)
            # 复制原图
            extended_img[:img_height, :] = img_with_bbox
            
            # 添加问题文本
            for j, line in enumerate(question_lines):
                y_pos = img_height + 30 + j * line_height
                cv2.putText(extended_img, line, (10, y_pos), font, font_scale, (255, 255, 255), font_thickness)
            
            # 保存图像
            image_filename = os.path.basename(str(image_path))
            base_name, ext = os.path.splitext(image_filename)
            
            # 为文件名添加标记，标识哪个模型正确
            if tuned_correct and baseline_correct:
                result_label = "both_correct"
            elif not tuned_correct and not baseline_correct:
                result_label = "both_wrong"
            elif tuned_correct and not baseline_correct:
                result_label = "tuned_better"
            else:
                result_label = "baseline_better"
            
            # 添加序号前缀到文件名
            output_path = os.path.join(output_dir, f"{i+1:03d}_{base_name}_{result_label}{ext}")
            cv2.imwrite(output_path, extended_img)
            
            success_count += 1
            
            # 打印进度
            if (i + 1) % 10 == 0 or i + 1 == len(tuned_results):
                print(f"已处理: {i + 1}/{len(tuned_results)} 样本")
        
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
    
    print(f"完成可视化: 成功 {success_count}/{len(tuned_results)}")
    print(f"比较统计: 两模型均正确: {both_correct}, 两模型均错误: {both_wrong}, 训练后模型更好: {tuned_better}, Baseline更好: {baseline_better}")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='可视化比较两个模型的点击任务评估结果')
    parser.add_argument('--tuned', type=str, required=True, 
                        help='训练后模型的评估结果JSON文件路径')
    parser.add_argument('--baseline', type=str, required=True, 
                        help='baseline模型的评估结果JSON文件路径')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='输出目录')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理的样本数量')
    parser.add_argument('--point-radius', type=int, default=5,
                        help='点的半径大小')
    parser.add_argument('--thickness', type=int, default=3, 
                        help='边界框线条粗细')
    parser.add_argument('--alpha', type=float, default=0.3, 
                        help='边界框填充区域的透明度')
    parser.add_argument('--correct-only', action='store_true',
                        help='仅处理训练后模型正确预测的样本')
    parser.add_argument('--incorrect-only', action='store_true',
                        help='仅处理训练后模型错误预测的样本')
    parser.add_argument('--at-least-one-wrong', action='store_true',
                        help='仅处理至少有一个模型预测错误的样本')
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.tuned):
        print(f"错误: 训练后模型结果文件不存在: {args.tuned}")
        return
    
    if not os.path.exists(args.baseline):
        print(f"错误: Baseline模型结果文件不存在: {args.baseline}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理评估结果
    success_count = visualize_comparison(
        args.tuned, 
        args.baseline,
        args.output,
        args.limit,
        args.point_radius,
        args.thickness, 
        args.alpha,
        args.correct_only,
        args.incorrect_only,
        args.at_least_one_wrong
    )
    
    print(f"\n总结:")
    print(f"成功处理样本: {success_count}")
    print(f"所有可视化图像已保存到: {os.path.abspath(args.output)}")
    print("\n用法示例:")
    print(f"1. 查看可视化图像: 打开 {os.path.abspath(args.output)} 文件夹")
    print("2. 仅查看训练后模型正确的样本: python visualize_model_comparison.py --tuned <训练后模型结果> --baseline <基线模型结果> --correct-only")
    print("3. 限制处理样本数: python visualize_model_comparison.py --tuned <训练后模型结果> --baseline <基线模型结果> --limit 20")

if __name__ == "__main__":
    main() 