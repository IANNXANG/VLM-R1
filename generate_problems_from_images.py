import json
import os
import base64
import time
from tqdm import tqdm
import random
import argparse
from openai import OpenAI
import re

# 配置路径
DEFAULT_DATA_PATH = "otherdata/ScreenSpot-v2/converted_data_click/screenspot_desktop.json"
DEFAULT_IMAGE_ROOT = "otherdata/ScreenSpot-v2"
DEFAULT_OUTPUT_PATH = "otherdata/ScreenSpot-v2/converted_data_click/screenspot_desktop_without_problem.json"

# VLLM API配置 - 使用OpenAI客户端通信
DEFAULT_VLLM_URL = "http://localhost:8001/v1"
DEFAULT_MODEL_NAME = "8001vllm"

# 错误处理配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='根据图片生成problem描述')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help=f'数据文件路径 (默认: {DEFAULT_DATA_PATH})')
    parser.add_argument('--image_root', type=str, default=DEFAULT_IMAGE_ROOT,
                        help=f'图片根目录 (默认: {DEFAULT_IMAGE_ROOT})')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f'输出文件路径 (默认: {DEFAULT_OUTPUT_PATH})')
    parser.add_argument('--api_url', type=str, default=DEFAULT_VLLM_URL,
                        help=f'VLLM API URL (默认: {DEFAULT_VLLM_URL})')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help=f'模型名称 (默认: {DEFAULT_MODEL_NAME})')
    parser.add_argument('--start_index', type=int, default=0,
                        help='开始处理的索引 (默认: 0)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='API请求间隔时间，秒 (默认: 0.5)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='生成温度 (默认: 0.2)')
    parser.add_argument('--prompt', type=str, 
                        default="Describe a task that requires clicking on a SINGLE, SPECIFIC element in this screenshot. Formulate your answer as a short imperative statement without mentioning clicking. The task must have only ONE correct location to interact with. Examples: 'close this window', 'minimize this window', 'view daily challenges', etc.",
                        help='发送给模型的提示词')
    return parser.parse_args()

def encode_image(image_path):
    """将图片编码为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败: {image_path}, 错误: {e}")
        return None

def clean_problem_text(text):
    """清理和格式化生成的问题文本"""
    # 去除多余的标点符号
    text = text.strip()
    
    # 移除引号
    text = text.replace('"', '').replace("'", "")
    
    # 处理常见的动作词前缀
    prefixes_to_remove = [
        "I need to ", "You need to ", "The task is to ", 
        "You should ", "I should ", "Task: ", "Click to ",
        "Click on ", "Tap on ", "Press "
    ]
    
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
    
    # 确保第一个单词首字母小写（任务描述通常以动词开始）
    if len(text) > 0 and text[0].isupper() and ' ' in text:
        first_word_end = text.find(' ')
        text = text[:1].lower() + text[1:]
    
    # 去除句号和多余的空格
    text = re.sub(r'\.+$', '', text)
    text = text.strip()
    
    return text

def generate_problem_from_image(image_path, api_url, model_name, prompt, temperature, retries=0):
    """使用OpenAI客户端通过VLLM API生成problem描述"""
    if retries >= MAX_RETRIES:
        print(f"达到最大重试次数 ({MAX_RETRIES})，跳过图片: {image_path}")
        return None
    
    try:
        # 编码图像
        base64_image = encode_image(image_path)
        if not base64_image:
            return None
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url=api_url,
            api_key="dummy-key"  # vLLM不需要真实的API密钥
        )
        
        # 发送请求
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=50
        )
        
        # 解析响应
        problem = response.choices[0].message.content.strip()
        
        # 清理和格式化问题文本
        problem = clean_problem_text(problem)
        
        return problem
    
    except Exception as e:
        print(f"API请求错误: {e}")
        # 随机延迟后重试
        retry_time = RETRY_DELAY + random.uniform(0, 2)
        print(f"将在 {retry_time:.2f} 秒后重试 (尝试 {retries+1}/{MAX_RETRIES})")
        time.sleep(retry_time)
        return generate_problem_from_image(image_path, api_url, model_name, prompt, temperature, retries + 1)

def process_dataset(args):
    """处理数据集，为每个图片生成problem"""
    # 加载原始数据
    print(f"正在读取数据: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 记录起始索引
    start_index = args.start_index
    total = len(data)
    
    if start_index >= total:
        print(f"起始索引 {start_index} 超出数据总数 {total}，无需处理")
        return
    
    print(f"共有 {total} 条数据，从索引 {start_index} 开始处理")
    
    try:
        # 处理每条数据
        for i, item in enumerate(tqdm(data[start_index:], initial=start_index, total=total)):
            current_index = start_index + i
            image_path = os.path.join(args.image_root, item['image'])
            
            # 如果图片路径存在
            if os.path.exists(image_path):
                # 保存原始问题用于显示
                original_problem = item['problem']
                
                # 生成新的问题
                new_problem = generate_problem_from_image(
                    image_path, 
                    args.api_url, 
                    args.model_name, 
                    args.prompt, 
                    args.temperature
                )
                
                if new_problem:
                    # 替换问题
                    item['problem'] = new_problem
                    print(f"[{current_index+1}/{total}] 原问题: '{original_problem}' -> 新问题: '{new_problem}'")
                else:
                    print(f"[{current_index+1}/{total}] 警告: 无法生成问题，保留原问题: '{original_problem}'")
                
                # 限制API请求频率
                time.sleep(args.delay)
            else:
                print(f"[{current_index+1}/{total}] 错误: 图片不存在: {image_path}")
    
    except KeyboardInterrupt:
        print("\n处理被用户中断！")
    except Exception as e:
        print(f"\n处理出错: {e}")
    
    # 保存处理后的数据
    print(f"正在保存处理后的数据到: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("处理完成!")

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args) 