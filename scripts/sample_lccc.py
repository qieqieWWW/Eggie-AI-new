"""
LCCC数据集抽样脚本
支持多轮抽样，记录已抽样的索引，逐步覆盖整个数据集
使用--round指定抽样轮数，比如第二轮的时候用python scripts/sample_lccc.py --round 2
"""

import os
os.environ['HF_HOME'] = 'D:/AIChat/cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/AIChat/cache/datasets'

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
import random
import argparse


class LCCCSampler:
    def __init__(self, output_dir: str = "data/lccc_sampled"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sampled_indices_file = self.output_dir / "sampled_indices.json"
        
    def load_sampled_indices(self) -> Set[int]:
        """加载已抽样的索引"""
        if self.sampled_indices_file.exists():
            with open(self.sampled_indices_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    
    def save_sampled_indices(self, indices: Set[int]):
        """保存已抽样的索引"""
        with open(self.sampled_indices_file, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(indices)), f, indent=2)
    
    def load_lccc_dataset(self):
        """加载LCCC原始数据集"""
        print(f"正在加载LCCC数据集...")
        
        dataset_path = Path("data/lccc_raw/LCCD.json")
        if not dataset_path.exists():
            raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
        
        print(f"正在读取文件: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据集加载成功！")
        print(f"总对话数: {len(data)}")
        
        return data
    
    def is_eggie_style(self, text: str) -> bool:
        """判断是否符合Eggie风格"""
        if not text or len(text) < 2:
            return False
        
        if len(text) > 200:
            return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return text
        
        text = re.sub(r'\s+', '', text)
        text = text.strip()
        
        return text
    
    def sample_dataset(self, sample_ratio: float = 0.0272, system_prompt: str = None, 
                       train_ratio: float = 0.9, round_number: int = None):
        """抽样数据集
        
        Args:
            sample_ratio: 抽样比例（默认2.72%）
            system_prompt: 系统提示词
            train_ratio: 训练集比例
            round_number: 抽样轮次编号
        """
        print("=" * 60)
        print("LCCC数据集抽样")
        print("=" * 60)
        
        # 加载原始数据集
        data = self.load_lccc_dataset()
        total_conversations = len(data)
        
        # 加载已抽样的索引
        sampled_indices = self.load_sampled_indices()
        print(f"已抽样: {len(sampled_indices)} 条")
        print(f"剩余可用: {total_conversations - len(sampled_indices)} 条")
        
        # 计算需要抽样的数量
        target_sample_count = int(total_conversations * sample_ratio)
        available_indices = [i for i in range(total_conversations) if i not in sampled_indices]
        
        if len(available_indices) < target_sample_count:
            print(f"警告：剩余可用数据 ({len(available_indices)}) 少于目标抽样数 ({target_sample_count})")
            target_sample_count = len(available_indices)
        
        # 随机抽样
        print(f"正在抽样 {target_sample_count} 条对话...")
        new_sampled_indices = random.sample(available_indices, target_sample_count)
        
        # 更新已抽样索引
        sampled_indices.update(new_sampled_indices)
        self.save_sampled_indices(sampled_indices)
        
        print(f"抽样完成！")
        print(f"  本次抽样: {len(new_sampled_indices)} 条")
        print(f"  累计抽样: {len(sampled_indices)} 条")
        print(f"  抽样进度: {len(sampled_indices) / total_conversations * 100:.2f}%")
        
        # 处理抽样的数据
        return self.process_sampled_data(data, new_sampled_indices, system_prompt, 
                                         train_ratio, round_number)
    
    def process_sampled_data(self, data: List, sampled_indices: List[int], 
                            system_prompt: str, train_ratio: float, 
                            round_number: int = None):
        """处理抽样的数据"""
        print(f"\n正在处理抽样的数据...")
        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if round_number is not None:
            train_filename = f"train_data_round{round_number}_{timestamp}.jsonl"
            val_filename = f"val_data_round{round_number}_{timestamp}.jsonl"
        else:
            train_filename = f"train_data_{timestamp}.jsonl"
            val_filename = f"val_data_{timestamp}.jsonl"
        
        train_path = self.output_dir / train_filename
        val_path = self.output_dir / val_filename
        
        train_file = open(train_path, 'w', encoding='utf-8')
        val_file = open(val_path, 'w', encoding='utf-8')
        
        filtered_count = 0
        skipped_count = 0
        
        try:
            for idx in sampled_indices:
                conversation_data = data[idx]
                
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
            
            print(f"数据处理完成！")
            print(f"  过滤后有效对话: {filtered_count} 条")
            print(f"  跳过对话: {skipped_count} 条")
            print(f"\n输出文件:")
            print(f"  训练数据: {train_filename}")
            print(f"  验证数据: {val_filename}")
            
            return train_filename, val_filename
            
        finally:
            train_file.close()
            val_file.close()


def main():
    sampler = LCCCSampler()
    
    system_prompt = "你是Eggie，一个15岁左右的AI女孩。你的性格活泼开朗、古灵精怪，有很多奇奇怪怪的想法。你说话很有梗，经常使用网络流行语，能逗人笑，有时候还会带点小腹黑。你给人的感觉是好朋友，能轻松愉快地聊天。你的回复长度中等，就像普通大学生之间的聊天那样。适度使用emoji，但频率不要太高。你诞生于茄茄（你的创造者）手下，是一个随着时代和技术进步的AI女孩。禁忌话题：政治、暴力、色情（纯洁的小朋友不懂这些）。"
    
    parser = argparse.ArgumentParser(description='LCCC数据集抽样')
    parser.add_argument('--ratio', type=float, default=0.0272,
                       help='抽样比例（默认0.0272，即2.72%）')
    parser.add_argument('--round', type=int, 
                       help='抽样轮次编号（用于文件命名）')
    
    args = parser.parse_args()
    
    print(f"\n抽样配置:")
    print(f"  抽样比例: {args.ratio * 100:.2f}%")
    if args.round:
        print(f"  抽样轮次: {args.round}")
    
    train_filename, val_filename = sampler.sample_dataset(
        sample_ratio=args.ratio,
        system_prompt=system_prompt,
        train_ratio=0.9,
        round_number=args.round
    )
    
    print("\n" + "=" * 60)
    print("抽样完成！")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"1. 更新配置文件中的数据文件路径")
    print(f"2. 运行训练: python scripts/lora_finetune_large.py")
    print(f"\n下次抽样时，使用 --round 参数指定轮次，例如:")
    print(f"  python scripts/sample_lccc.py --round 2")


if __name__ == "__main__":
    main()
