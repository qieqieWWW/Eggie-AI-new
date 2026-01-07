"""
LCCC数据集处理脚本
用于从LCCC数据集中筛选和处理符合Eggie性格的对话
"""

from datasets import load_dataset
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import random


class LCCCProcessor:
    def __init__(self, output_dir: str = "data/lccc_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.forbidden_patterns = [
            r'政治|政府|选举|党派',
            r'暴力|杀人|打人|打架|死亡|死',
            r'色情|性|做爱|黄片|AV',
            r'赌博|博彩|彩票'
        ]
        
    def load_lccc_dataset(self, dataset_size: str = "base", max_conversations: int = 10000):
        print(f"正在加载LCCC-{dataset_size}数据集...")
        
        try:
            dataset_path = Path("data/lccc_raw/LCCD.json")
            if not dataset_path.exists():
                raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
                
            print(f"正在读取文件: {dataset_path}")
            print(f"注意：文件较大，将只读取前 {max_conversations} 条对话...")
            
            import json
            conversations = []
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations = data[:max_conversations]
                
            print(f"数据集加载成功！")
            print(f"总对话数: {len(conversations)}")
            
            dataset = {"train": []}
            for conv in conversations:
                dataset["train"].append({"conversation": conv})
                
            return dataset
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("尝试从GitHub下载...")
            return self._download_from_github(dataset_size)
            
    def _download_from_github(self, dataset_size: str):
        print(f"从GitHub下载LCCC-{dataset_size}数据集...")
        print("请手动下载以下文件:")
        if dataset_size == "base":
            print("  - LCCC-base: https://github.com/thu-coai/CDial-GPT")
        else:
            print("  - LCCC-large: https://github.com/thu-coai/CDial-GPT")
        print("下载后放在 data/lccc_raw/ 目录下")
        return None
        
    def is_eggie_style(self, text: str) -> bool:
        if not text or len(text) < 2:
            return False
            
        if len(text) > 100:
            return False
            
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text):
                return False
                
        return True
        
    def filter_conversations(self, dataset, max_samples: int = 10000):
        print(f"正在筛选符合Eggie风格的对话...")
        
        filtered_conversations = []
        
        for idx, conversation in enumerate(dataset['train']):
            if len(filtered_conversations) >= max_samples:
                break
                
            conversation_data = conversation['conversation']
            
            if not conversation_data or len(conversation_data) < 2:
                continue
            
            for i in range(0, len(conversation_data) - 1, 2):
                user_msg = conversation_data[i]
                assistant_msg = conversation_data[i + 1]
                
                if not user_msg or not assistant_msg:
                    continue
                    
                if not self.is_eggie_style(assistant_msg):
                    continue
                    
                filtered_conversations.append({
                    "user": user_msg,
                    "assistant": assistant_msg
                })
                
                if len(filtered_conversations) >= max_samples:
                    break
            
            if len(filtered_conversations) % 1000 == 0:
                print(f"已筛选 {len(filtered_conversations)} 条对话...")
                
        print(f"筛选完成！共 {len(filtered_conversations)} 条对话")
        return filtered_conversations
        
    def format_for_training(self, conversations: List[Dict], system_prompt: str):
        print("正在格式化训练数据...")
        
        formatted_data = []
        
        for conv in conversations:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conv['user']},
                {"role": "assistant", "content": conv['assistant']}
            ]
            
            formatted_data.append({
                "messages": messages
            })
            
        print(f"格式化完成！共 {len(formatted_data)} 条训练样本")
        return formatted_data
        
    def save_data(self, data: List[Dict], filename: str):
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f"数据已保存到: {output_path}")
        
    def split_data(self, data: List[Dict], train_ratio: float = 0.8):
        random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        print(f"训练集: {len(train_data)} 条")
        print(f"验证集: {len(val_data)} 条")
        
        return train_data, val_data
        
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
    
    dataset = processor.load_lccc_dataset(dataset_size="base")
    
    if dataset is None:
        print("无法加载数据集，请检查网络连接或手动下载数据")
        return
    
    filtered_conversations = processor.filter_conversations(dataset, max_samples=10000)
    
    formatted_data = processor.format_for_training(filtered_conversations, system_prompt)
    
    merged_data = processor.merge_with_template(formatted_data, "data/training_data_template.json")
    
    train_data, val_data = processor.split_data(merged_data, train_ratio=0.8)
    
    processor.save_data(train_data, "train_data.jsonl")
    processor.save_data(val_data, "val_data.jsonl")
    
    print("\n数据处理完成！")
    print("下一步可以使用以下命令进行LoRA微调:")
    print("python scripts/lora_finetune.py")


if __name__ == "__main__":
    main()
