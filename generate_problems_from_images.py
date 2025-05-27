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
                        default="Generate 4 different tasks that require clicking on SINGLE, SPECIFIC elements in this screenshot. Each task should target a different clickable element and be formulated as a short imperative statement without mentioning clicking. Each task must have only ONE correct location to interact with. Format your response as: 1. [task1] 2. [task2] 3. [task3] 4. [task4]. Examples: '1. close this window 2. minimize this window 3. view daily challenges 4. open settings menu'",
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

def parse_four_problems(response_text):
    """从模型响应中解析出四个不同的问题"""
    problems = []
    
    # 尝试匹配编号格式：1. xxx 2. xxx 3. xxx 4. xxx
    pattern = r'(\d+)\s*[.:]\s*([^\d]+?)(?=\s*\d+\s*[.:]|$)'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if len(matches) >= 4:
        for i in range(4):
            problem = clean_problem_text(matches[i][1])
            if problem:
                problems.append(problem)
    
    # 如果编号格式解析失败，尝试按行分割
    if len(problems) < 4:
        problems = []
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines[:4]:  # 只取前4行
            # 移除行首的编号
            cleaned_line = re.sub(r'^\d+[.:]\s*', '', line)
            problem = clean_problem_text(cleaned_line)
            if problem:
                problems.append(problem)
    
    # 如果仍然不足4个，用原始响应填充
    while len(problems) < 4:
        fallback_problem = clean_problem_text(response_text)
        if not fallback_problem:
            fallback_problem = "interact with interface element"
        problems.append(f"{fallback_problem} (variant {len(problems)+1})")
    
    return problems[:4]  # 确保只返回4个

def generate_problems_from_image(image_path, api_url, model_name, prompt, temperature, retries=0):
    """使用OpenAI客户端通过VLLM API生成四个problem描述"""
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
            max_tokens=200  # 增加token数量以容纳4个问题
        )
        
        # 解析响应
        response_text = response.choices[0].message.content.strip()
        
        # 解析出四个问题
        problems = parse_four_problems(response_text)
        
        return problems
    
    except Exception as e:
        print(f"API请求错误: {e}")
        # 随机延迟后重试
        retry_time = RETRY_DELAY + random.uniform(0, 2)
        print(f"将在 {retry_time:.2f} 秒后重试 (尝试 {retries+1}/{MAX_RETRIES})")
        time.sleep(retry_time)
        return generate_problems_from_image(image_path, api_url, model_name, prompt, temperature, retries + 1)

def process_dataset(args):
    """处理数据集，为每个图片生成四个problem"""
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
    print(f"预计生成 {(total - start_index) * 4} 条新数据")
    
    # 新的数据列表，用于存储扩展后的数据
    new_data = []
    
    try:
        # 处理每条数据
        for i, item in enumerate(tqdm(data[start_index:], initial=start_index, total=total)):
            current_index = start_index + i
            image_path = os.path.join(args.image_root, item['image'])
            
            # 如果图片路径存在
            if os.path.exists(image_path):
                # 保存原始问题用于显示
                original_problem = item['problem']
                
                # 生成四个新的问题
                new_problems = generate_problems_from_image(
                    image_path, 
                    args.api_url, 
                    args.model_name, 
                    args.prompt, 
                    args.temperature
                )
                
                if new_problems and len(new_problems) == 4:
                    # 为每个问题创建一个新的数据条目
                    for j, problem in enumerate(new_problems):
                        new_item = item.copy()  # 复制原始数据
                        new_item['problem'] = problem
                        # 可选：添加变体标识
                        if 'id' in new_item:
                            new_item['id'] = f"{new_item['id']}_v{j+1}"
                        new_data.append(new_item)
                    
                    print(f"[{current_index+1}/{total}] 原问题: '{original_problem}'")
                    for j, problem in enumerate(new_problems):
                        print(f"  -> 新问题{j+1}: '{problem}'")
                else:
                    print(f"[{current_index+1}/{total}] 警告: 无法生成4个问题，跳过该图片")
                    # 如果生成失败，可以选择保留原始数据或跳过
                    # new_data.append(item)  # 取消注释以保留原始数据
                
                # 限制API请求频率
                time.sleep(args.delay)
            else:
                print(f"[{current_index+1}/{total}] 错误: 图片不存在: {image_path}")
    
    except KeyboardInterrupt:
        print("\n处理被用户中断！")
    except Exception as e:
        print(f"\n处理出错: {e}")
    
    # 保存处理后的数据
    print(f"\n正在保存处理后的数据到: {args.output_path}")
    print(f"原始数据量: {total}, 新数据量: {len(new_data)}")
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print("处理完成!")

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)