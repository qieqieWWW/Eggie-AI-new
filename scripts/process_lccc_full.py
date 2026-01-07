"""
LCCC数据集处理脚本 - 完整数据集版本
支持处理大规模LCCC数据集
"""

import os
os.environ['HF_HOME'] = 'D:/AIChat/cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/AIChat/cache/datasets'

from datasets import load_dataset
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import random
import argparse


class LCCCProcessor:
    def __init__(self, output_dir: str = "data/lccc_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_lccc_dataset(self, max_conversations: int = None):
        print(f"正在加载LCCC数据集...")
        
        try:
            dataset_path = Path("data/lccc_raw/LCCD.json")
            if not dataset_path.exists():
                raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
                
            print(f"正在读取文件: {dataset_path}")
            
            import json
            conversations = []
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if max_conversations:
                    conversations = data[:max_conversations]
                    print(f"读取前 {max_conversations} 条对话")
                else:
                    conversations = data
                    print(f"读取全部 {len(data)} 条对话")
                
            print(f"数据集加载成功！")
            print(f"总对话数: {len(conversations)}")
            
            dataset = {"train": []}
            for conv in conversations:
                dataset["train"].append({"conversation": conv})
                
            return dataset
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return None
            
    def is_eggie_style(self, text: str) -> bool:
        if not text or len(text) < 2:
            return False
            
        if len(text) > 200:
            return False
                
        return True
    
    def clean_text(self, text: str) -> str:
        if not text:
            return text
        
        text = re.sub(r'\s+', '', text)
        text = text.strip()
        
        return text
        
    def filter_and_save_conversations(self, dataset, system_prompt: str, 
                                       max_samples: int = None, 
                                       train_ratio: float = 0.9):
        print(f"正在筛选符合Eggie风格的对话并保存...")
        
        total_conversations = len(dataset['train'])
        skipped_count = 0
        filtered_count = 0
        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        train_filename = f"train_data_{timestamp}.jsonl"
        val_filename = f"val_data_{timestamp}.jsonl"
        
        train_path = self.output_dir / train_filename
        val_path = self.output_dir / val_filename
        
        train_file = open(train_path, 'w', encoding='utf-8')
        val_file = open(val_path, 'w', encoding='utf-8')
        
        try:
            for idx, conversation in enumerate(dataset['train']):
                if max_samples and filtered_count >= max_samples:
                    break
                    
                conversation_data = conversation['conversation']
                
                if not conversation_data or len(conversation_data) < 2:
                    skipped_count += 1
                    continue
                
                for i in range(0, len(conversation_data) - 1, 2):
                    user_msg = conversation_data[i]
                    assistant_msg = conversation_data[i + 1]
                    
                    if not user_msg or not assistant_msg:
                        continue
                    
                    user_msg = self.clean_text(user_msg)
                    assistant_msg = self.clean_text(assistant_msg)
                        
                    if not self.is_eggie_style(assistant_msg):
                        continue
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ]
                    
                    formatted_item = {"messages": messages}
                    json_str = json.dumps(formatted_item, ensure_ascii=False) + '\n'
                    
                    if random.random() < train_ratio:
                        train_file.write(json_str)
                    else:
                        val_file.write(json_str)
                    
                    filtered_count += 1
                    
                    if max_samples and filtered_count >= max_samples:
                        break
                
                if (idx + 1) % 100000 == 0:
                    print(f"已处理 {idx + 1}/{total_conversations} 条对话，筛选出 {filtered_count} 条有效对话...")
            
            print(f"筛选完成！")
            print(f"  处理对话数: {total_conversations}")
            print(f"  筛选结果: {filtered_count} 条")
            print(f"  跳过对话: {skipped_count} 条")
            
            return train_filename, val_filename
            
        finally:
            train_file.close()
            val_file.close()
        
    def merge_with_template(self, lccc_data: List[Dict], template_path: str):
        print("正在合并模板数据...")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
            
        template_conversations = template['conversations']
        
        formatted_template = []
        for conv in template_conversations:
            formatted_template.append({
                "messages": conv['messages']
            })
            
        merged_data = formatted_template + lccc_data
        
        print(f"合并完成！模板数据: {len(formatted_template)} 条")
        print(f"LCCC数据: {len(lccc_data)} 条")
        print(f"总计: {len(merged_data)} 条")
        
        return merged_data


def main():
    processor = LCCCProcessor()
    
    system_prompt = "你是Eggie，一个15岁左右的AI女孩。你的性格活泼开朗、古灵精怪，有很多奇奇怪怪的想法。你说话很有梗，经常使用网络流行语，能逗人笑，有时候还会带点小腹黑。你给人的感觉是好朋友，能轻松愉快地聊天。你的回复长度中等，就像普通大学生之间的聊天那样。适度使用emoji，但频率不要太高。你诞生于茄茄（你的创造者）手下，是一个随着时代和技术进步的AI女孩。禁忌话题：政治、暴力、色情（纯洁的小朋友不懂这些）。"
    
    print("=" * 60)
    print("LCCC数据集处理 - 完整数据集版本")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='处理LCCC数据集')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'large', 'full', 'custom'],
                       help='数据集大小: small(10k), medium(100k), large(500k), full(全部), custom(自定义)')
    parser.add_argument('--count', type=int, help='自定义对话数量（当size=custom时使用）')
    
    args = parser.parse_args()
    
    max_conversations = None
    if args.size == 'small':
        max_conversations = 10000
    elif args.size == 'medium':
        max_conversations = 100000
    elif args.size == 'large':
        max_conversations = 500000
    elif args.size == 'full':
        max_conversations = None
    elif args.size == 'custom':
        max_conversations = args.count if args.count else 10000
    else:
        print("\n请选择数据集大小:")
        print("1. 小规模 (10,000条) - 快速测试")
        print("2. 中规模 (100,000条) - 平衡选择")
        print("3. 大规模 (500,000条) - 更好效果")
        print("4. 完整数据集 (12,000,000条) - 最佳效果")
        print("5. 自定义数量")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            max_conversations = 10000
        elif choice == "2":
            max_conversations = 100000
        elif choice == "3":
            max_conversations = 500000
        elif choice == "4":
            max_conversations = None
        elif choice == "5":
            try:
                max_conversations = int(input("请输入对话数量: "))
            except ValueError:
                print("输入无效，使用默认值10000")
                max_conversations = 10000
        else:
            print("输入无效，使用默认值10000")
            max_conversations = 10000
    
    print(f"\n将处理 {max_conversations if max_conversations else '全部'} 条对话")
    
    dataset = processor.load_lccc_dataset(max_conversations=max_conversations)
    
    if dataset is None:
        print("无法加载数据集，请检查数据文件")
        return
    
    train_filename, val_filename = processor.filter_and_save_conversations(
        dataset, system_prompt, max_samples=max_conversations, train_ratio=0.9
    )
    
    print("\n" + "=" * 60)
    print("数据处理完成！")
    print("=" * 60)
    print(f"训练数据: {train_filename}")
    print(f"验证数据: {val_filename}")
    print(f"\n下一步:")
    print("1. 更新 config/lora_config_large.yaml 中的数据文件路径")
    print("2. 运行训练: python scripts/lora_finetune_large.py")


if __name__ == "__main__":
    main()
