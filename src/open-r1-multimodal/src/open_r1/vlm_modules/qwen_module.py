from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "click":
                return "{Question} Look at the image and identify where to click. First reason about the correct location in <think> </think> tags, then provide the exact [x, y] coordinates in <answer> </answer> tags."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_click(completions, **kwargs):
        """Check if the Qwen model output matches the point coordinate format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\[\s*\d+\s*,\s*\d+\s*\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards
        
    @staticmethod
    def click_reward(completions, solution, **kwargs):
        """Calculate reward for click coordinates - reward is 1.0 if point is inside the bounding box, 0.0 otherwise."""
        import re
        import os
        from datetime import datetime
        
        def point_in_bbox(point, bbox):
            """Check if point [x, y] is inside bounding box [x1, y1, x2, y2]."""
            x, y = point
            x1, y1, x2, y2 = bbox
            return x1 <= x <= x2 and y1 <= y <= y2
            
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        point_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    point_match = re.search(point_pattern, content_answer)
                    if point_match:
                        point = [int(point_match.group(1)), int(point_match.group(2))]
                        if point_in_bbox(point, sol):
                            reward = 1.0
            except Exception as e:
                pass  # Continue if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Click Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution (bbox): {sol}\n")
        return rewards

    @staticmethod
    def majority_click_reward(completions, prompts=None, **kwargs):
        """
        Compute the majority click reward for a list of completions using global clustering across all processes.
        
        Args:
            completions: List of completion strings
            prompts: List of prompt strings (optional)
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            List of rewards (floats)
        """
        import re
        import os
        import numpy as np
        import torch
        import torch.distributed as dist
        from datetime import datetime
        from sklearn.cluster import DBSCAN
        
        DEBUG_MODE = os.getenv("DEBUG_MODE") == "true"
        
        if DEBUG_MODE:
            debug_file = "src/open-r1-multimodal/debug_log_Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click-MajorityVoting-Temp0.7.txt"
            
        def extract_point(content):
            """从模型输出中提取点坐标"""
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            point_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'
            
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    point_match = re.search(point_pattern, content_answer)
                    if point_match:
                        return [int(point_match.group(1)), int(point_match.group(2))]
            except Exception:
                pass
            return None
            
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        # 获取分布式训练信息
        is_distributed = dist.is_available() and dist.is_initialized()
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        
        # 提取当前进程的所有坐标点，并记录有效性
        local_points = []
        local_valid_mask = []  # 记录每个点是否有效
        for content in contents:
            point = extract_point(content)
            if point:
                local_points.append(point)
                local_valid_mask.append(True)
            else:
                # 无效坐标不参与聚类，但需要在奖励中标记为0
                local_points.append([-999, -999])  # 占位符，不会用于聚类
                local_valid_mask.append(False)
        
        if is_distributed:
            # 收集所有进程的有效点进行全局聚类
            
            # 获取当前设备
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
            
            # 收集所有进程的有效点
            local_valid_points = [p for p, valid in zip(local_points, local_valid_mask) if valid]
            
            # 将有效点转换为tensor进行分布式收集
            if local_valid_points:
                local_valid_points_tensor = torch.tensor(local_valid_points, dtype=torch.float32, device=device)
                local_valid_count = torch.tensor([len(local_valid_points)], dtype=torch.int32, device=device)
            else:
                local_valid_points_tensor = torch.empty((0, 2), dtype=torch.float32, device=device)
                local_valid_count = torch.tensor([0], dtype=torch.int32, device=device)
            
            # 收集所有进程的有效点数量
            all_valid_counts = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(world_size)]
            dist.all_gather(all_valid_counts, local_valid_count)
            all_valid_counts = [count.item() for count in all_valid_counts]
            
            # 收集所有进程的有效点
            max_points = max(all_valid_counts) if all_valid_counts else 0
            if max_points > 0:
                # 将本地tensor填充到最大长度
                if local_valid_points_tensor.size(0) < max_points:
                    padding = torch.full((max_points - local_valid_points_tensor.size(0), 2), -999.0, dtype=torch.float32, device=device)
                    local_valid_points_tensor = torch.cat([local_valid_points_tensor, padding], dim=0)
                
                # 收集所有进程的点
                all_points_list = [torch.zeros((max_points, 2), dtype=torch.float32, device=device) for _ in range(world_size)]
                dist.all_gather(all_points_list, local_valid_points_tensor)
                
                # 合并所有有效点
                global_valid_points = []
                for i, (points_tensor, count) in enumerate(zip(all_points_list, all_valid_counts)):
                    if count > 0:
                        valid_points_from_process = points_tensor[:count].cpu().numpy().tolist()
                        global_valid_points.extend(valid_points_from_process)
            else:
                global_valid_points = []
            
            # 如果全局没有足够的有效点，返回全0奖励
            if len(global_valid_points) < 2:
            return [0.0] * len(contents)
        
            # 在所有进程上进行相同的全局聚类
            global_valid_points_array = np.array(global_valid_points)
            eps = 40  # 聚类的最大距离
            dbscan = DBSCAN(eps=eps, min_samples=1).fit(global_valid_points_array)
            global_cluster_labels = dbscan.labels_
        
        # 找出最大的簇
        label_counts = {}
            for label in global_cluster_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        largest_cluster_label = max(label_counts, key=label_counts.get) if label_counts else -1
        
            # 将全局聚类结果映射到当前进程的点
            # 需要找到当前进程的有效点在全局列表中的位置
            current_process_start = sum(all_valid_counts[:local_rank])
            current_process_end = current_process_start + all_valid_counts[local_rank]
            current_process_cluster_labels = global_cluster_labels[current_process_start:current_process_end]
            
        else:
            # 非分布式训练，使用局部聚类
            local_valid_points = [p for p, valid in zip(local_points, local_valid_mask) if valid]
            
            if len(local_valid_points) < 2:
                return [0.0] * len(contents)
            
            valid_points_array = np.array(local_valid_points)
            eps = 40
            dbscan = DBSCAN(eps=eps, min_samples=1).fit(valid_points_array)
            current_process_cluster_labels = dbscan.labels_
            
            label_counts = {}
            for label in current_process_cluster_labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            
            largest_cluster_label = max(label_counts, key=label_counts.get) if label_counts else -1
            global_valid_points = local_valid_points
            global_cluster_labels = current_process_cluster_labels
        
        # 生成奖励：将聚类结果映射回原始序列
        rewards = []
        valid_idx = 0
        for valid in local_valid_mask:
            if valid:
                # 有效点：根据聚类结果给奖励
                cluster_label = current_process_cluster_labels[valid_idx]
                rewards.append(1.0 if cluster_label == largest_cluster_label else 0.0)
                valid_idx += 1
            else:
                # 无效点：直接给0奖励
                rewards.append(0.0)
        
        # 重建cluster_labels用于调试输出（包含无效点标记）
        cluster_labels = []
        valid_idx = 0
        for valid in local_valid_mask:
            if valid:
                cluster_labels.append(int(current_process_cluster_labels[valid_idx]))  # 转换为Python int
                valid_idx += 1  
            else:
                cluster_labels.append(-2)  # 用-2标记无效点
        
        if DEBUG_MODE:
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Majority Click Reward -------------\n")
                
                # Log complete prompts and responses
                if prompts:
                    f.write(f"\n--- Complete Prompts ---\n")
                    for i, prompt in enumerate(prompts):
                        f.write(f"Prompt {i+1}: {prompt}\n")
                
                f.write(f"\n--- Complete Responses ---\n")
                for i, content in enumerate(contents):
                    f.write(f"Response {i+1}: {content}\n")
                
                f.write(f"\n--- Clustering Results ---\n")
                f.write(f"Distributed Training: {is_distributed}\n")
                f.write(f"Local Rank: {local_rank}, World Size: {world_size}\n")
                f.write(f"Local Points: {local_points}\n")
                f.write(f"Local Valid Mask: {local_valid_mask}\n")
                if is_distributed:
                    f.write(f"Global Valid Points (Total: {len(global_valid_points)}): {global_valid_points}\n")
                    f.write(f"Global Cluster Labels: {global_cluster_labels.tolist()}\n")
                    f.write(f"Current Process Cluster Labels: {current_process_cluster_labels.tolist()}\n")
                else:
                    f.write(f"Local Valid Points: {global_valid_points}\n")
                f.write(f"Final Cluster labels: {cluster_labels}\n")
                f.write(f"Largest cluster: {largest_cluster_label}\n")
                f.write(f"Rewards: {rewards}\n")
                
        return rewards