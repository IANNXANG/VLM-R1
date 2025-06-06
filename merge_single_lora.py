#!/usr/bin/env python3
"""
单个LoRA模型合并脚本 - 用于逐个合并模型以避免内存问题
"""

import os
import sys
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
import torch

def merge_single_lora(lora_checkpoint_path, base_model_path, output_path):
    """
    合并单个LoRA模型
    """
    print(f"LoRA检查点: {lora_checkpoint_path}")
    print(f"基础模型: {base_model_path}")
    print(f"输出路径: {output_path}")
    
    # 检查路径
    if not os.path.exists(lora_checkpoint_path):
        print(f"错误: LoRA检查点不存在: {lora_checkpoint_path}")
        return False
    
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型不存在: {base_model_path}")
        return False
    
    adapter_config_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"错误: 未找到adapter_config.json: {adapter_config_path}")
        return False
    
    try:
        print("\n1. 加载基础模型...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 基础模型加载完成")
        
        print("\n2. 加载LoRA适配器...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_checkpoint_path,
            torch_dtype=torch.bfloat16
        )
        print("✓ LoRA适配器加载完成")
        
        print("\n3. 合并LoRA权重...")
        merged_model = model.merge_and_unload()
        print("✓ LoRA权重合并完成")
        
        print("\n4. 创建输出目录...")
        os.makedirs(output_path, exist_ok=True)
        print(f"✓ 输出目录创建: {output_path}")
        
        print("\n5. 保存合并后的模型...")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        print("✓ 模型保存完成")
        
        print("\n6. 保存tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
        print("✓ Tokenizer保存完成")
        
        print("\n7. 保存processor...")
        try:
            processor = AutoProcessor.from_pretrained(base_model_path)
            processor.save_pretrained(output_path)
            print("✓ Processor保存完成")
        except Exception as e:
            print(f"⚠ Processor保存失败 (可忽略): {e}")
        
        print(f"\n🎉 成功合并模型到: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理内存
        print("\n8. 清理内存...")
        if 'base_model' in locals():
            del base_model
        if 'model' in locals():
            del model
        if 'merged_model' in locals():
            del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ 内存清理完成")

def main():
    parser = argparse.ArgumentParser(description="合并单个LoRA模型")
    parser.add_argument("--lora_path", required=True, help="LoRA检查点路径")
    parser.add_argument("--base_model", default="/c22940/zy/model/Qwen2.5-VL-7B-Instruct", help="基础模型路径")
    parser.add_argument("--output_path", required=True, help="输出路径")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LoRA模型合并工具")
    print("=" * 80)
    
    success = merge_single_lora(args.lora_path, args.base_model, args.output_path)
    
    if success:
        print("\n✅ 合并成功!")
        print(f"合并后的完整模型保存在: {args.output_path}")
        print("现在可以直接使用这个完整模型进行推理。")
        sys.exit(0)
    else:
        print("\n❌ 合并失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()