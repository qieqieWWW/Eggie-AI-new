import json
from pathlib import Path
from typing import List, Dict, Any
import random


class DataPreparer:
    def __init__(self, template_path: str = "data/training_data_template.json"):
        self.template_path = Path(template_path)
        self.system_prompt = ""
        self.conversations = []
        
    def load_template(self):
        with open(self.template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.system_prompt = data['system_prompt']
        self.conversations = data['conversations']
        
    def format_for_training(self, output_path: str = "data/training_data.jsonl"):
        formatted_data = []
        
        for conversation in self.conversations:
            messages = conversation['messages']
            
            formatted_messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            for msg in messages:
                formatted_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            formatted_data.append({
                "messages": formatted_messages
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"训练数据已保存到: {output_path}")
        print(f"共 {len(formatted_data)} 条对话")
        
    def add_conversation(self, user_message: str, assistant_message: str):
        new_conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        self.conversations.append(new_conversation)
        
    def save_template(self, output_path: str = None):
        if output_path is None:
            output_path = self.template_path
        
        data = {
            "system_prompt": self.system_prompt,
            "conversations": self.conversations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"模板已保存到: {output_path}")
        
    def shuffle_data(self):
        random.shuffle(self.conversations)
        print(f"数据已打乱，共 {len(self.conversations)} 条对话")
        
    def split_data(self, train_ratio: float = 0.8):
        split_index = int(len(self.conversations) * train_ratio)
        train_data = self.conversations[:split_index]
        val_data = self.conversations[split_index:]
        
        print(f"训练集: {len(train_data)} 条")
        print(f"验证集: {len(val_data)} 条")
        
        return train_data, val_data


def main():
    preparer = DataPreparer()
    preparer.load_template()
    
    print(f"加载了 {len(preparer.conversations)} 条对话")
    print(f"系统提示词: {preparer.system_prompt[:50]}...")
    
    preparer.format_for_training()
    
    train_data, val_data = preparer.split_data()
    
    print("\n训练数据准备完成！")
    print("接下来可以使用这些数据进行LoRA微调")


if __name__ == "__main__":
    main()
