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
    def majority_click_reward(completions, **kwargs):
        """
        使用majority voting（聚类）生成伪标签并计算奖励。
        奖励规则：属于最大簇的点获得1.0的奖励，其他点获得0.0的奖励。
        """
        import re
        import os
        import numpy as np
        from datetime import datetime
        from sklearn.cluster import DBSCAN
        
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
        
        # 提取所有坐标点
        points = []
        for content in contents:
            point = extract_point(content)
            if point:
                points.append(point)
            else:
                # 如果无法提取坐标，添加一个远离的点以保持数组大小一致
                points.append([-999, -999])
        
        # 如果没有足够的有效点，返回全0奖励
        if len([p for p in points if p[0] != -999]) < 2:
            return [0.0] * len(contents)
        
        # 使用DBSCAN进行聚类
        points_array = np.array(points)
        eps = 40  # 聚类的最大距离，与visualize_reward_hit_rate.py中相同
        dbscan = DBSCAN(eps=eps, min_samples=1).fit(points_array)
        cluster_labels = dbscan.labels_
        
        # 找出最大的簇
        label_counts = {}
        for label in cluster_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        largest_cluster_label = max(label_counts, key=label_counts.get) if label_counts else -1
        
        # 生成奖励：在最大簇中的点为1，其他为0
        rewards = [1.0 if label == largest_cluster_label else 0.0 for label in cluster_labels]
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Majority Click Reward -------------\n")
                f.write(f"Points: {points}\n")
                f.write(f"Cluster labels: {cluster_labels}\n")
                f.write(f"Largest cluster: {largest_cluster_label}\n")
                f.write(f"Rewards: {rewards}\n")
                
        return rewards