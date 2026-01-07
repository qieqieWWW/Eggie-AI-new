import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import random


class MemorySystem:
    def __init__(self, memory_dir: str = "data/memories"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.short_term_memory: Dict[str, List[Dict]] = {}
        self.long_term_memory: Dict[str, Dict[str, Any]] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        self.max_short_term_length = 20
        self.max_long_term_length = 100
        
        self.load_memories()
        
    def load_memories(self):
        long_term_file = self.memory_dir / "long_term_memory.json"
        user_profiles_file = self.memory_dir / "user_profiles.json"
        
        if long_term_file.exists():
            with open(long_term_file, 'r', encoding='utf-8') as f:
                self.long_term_memory = json.load(f)
                
        if user_profiles_file.exists():
            with open(user_profiles_file, 'r', encoding='utf-8') as f:
                self.user_profiles = json.load(f)
                
        print(f"加载记忆系统完成")
        print(f"长期记忆: {len(self.long_term_memory)} 条")
        print(f"用户画像: {len(self.user_profiles)} 个")
        
    def save_memories(self):
        long_term_file = self.memory_dir / "long_term_memory.json"
        user_profiles_file = self.memory_dir / "user_profiles.json"
        
        with open(long_term_file, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
            
        with open(user_profiles_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_profiles, f, ensure_ascii=False, indent=2)
            
    def add_to_short_term(self, user_id: str, role: str, content: str):
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = []
            
        self.short_term_memory[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.short_term_memory[user_id]) > self.max_short_term_length:
            self.short_term_memory[user_id] = self.short_term_memory[user_id][-self.max_short_term_length:]
            
    def get_short_term(self, user_id: str) -> List[Dict]:
        return self.short_term_memory.get(user_id, [])
        
    def add_to_long_term(self, user_id: str, key: str, value: str):
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = {}
            
        self.long_term_memory[user_id][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if len(self.long_term_memory[user_id]) > self.max_long_term_length:
            oldest_key = min(self.long_term_memory[user_id].keys(), 
                           key=lambda k: self.long_term_memory[user_id][k]["timestamp"])
            del self.long_term_memory[user_id][oldest_key]
            
    def get_long_term(self, user_id: str) -> Dict[str, Any]:
        return self.long_term_memory.get(user_id, {})
        
    def update_user_profile(self, user_id: str, field: str, value: str):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
            
        self.user_profiles[user_id][field] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_profiles.get(user_id, {})
        
    def extract_user_info(self, user_id: str, message: str):
        info_patterns = {
            "name": ["我叫", "我是", "名字是"],
            "age": ["我今年", "年龄"],
            "hobby": ["喜欢", "爱好", "兴趣"],
            "job": ["工作", "职业", "上班"],
            "location": ["住在", "来自", "在"]
        }
        
        for field, patterns in info_patterns.items():
            for pattern in patterns:
                if pattern in message:
                    value = message.split(pattern)[-1].strip()
                    if len(value) > 0 and len(value) < 50:
                        self.update_user_profile(user_id, field, value)
                        print(f"[记忆系统] 提取到用户信息: {field} = {value}")
                        break
                        
    def get_memory_summary(self, user_id: str) -> str:
        summary_parts = []
        
        profile = self.get_user_profile(user_id)
        if profile:
            profile_str = " | ".join([f"{k}: {v['value']}" for k, v in profile.items()])
            summary_parts.append(f"用户信息: {profile_str}")
            
        long_term = self.get_long_term(user_id)
        if long_term:
            recent_memories = sorted(long_term.items(), 
                                   key=lambda x: x[1]["timestamp"], 
                                   reverse=True)[:5]
            memories_str = " | ".join([f"{k}: {v['value']}" for k, v in recent_memories])
            summary_parts.append(f"重要记忆: {memories_str}")
            
        return " | ".join(summary_parts) if summary_parts else "暂无记忆"
        
    def clear_short_term(self, user_id: str):
        if user_id in self.short_term_memory:
            self.short_term_memory[user_id] = []
            
    def clear_all_short_term(self):
        self.short_term_memory = {}


class QuestionGenerator:
    def __init__(self):
        self.question_templates = [
            "对了，你最近{topic}怎么样呀？",
            "我想问问，你{topic}吗？",
            "话说回来，你{topic}？",
            "突然想到，你{topic}？",
            "好奇一下，你{topic}？"
        ]
        
        self.topics = [
            "在忙什么",
            "有什么有趣的事情",
            "心情如何",
            "最近看了什么电影",
            "有没有什么新发现",
            "周末打算做什么",
            "最近有没有遇到什么困难",
            "有什么想分享的"
        ]
        
        self.follow_up_questions = [
            "真的吗？能多跟我说说吗？",
            "哇，听起来很有意思！",
            "哈哈，那你后来怎么样了？",
            "嗯嗯，我懂了～",
            "还有呢？继续说呀～"
        ]
        
    def generate_question(self, context: Optional[str] = None) -> str:
        template = random.choice(self.question_templates)
        topic = random.choice(self.topics)
        
        question = template.format(topic=topic)
        
        return question
        
    def generate_follow_up(self) -> str:
        return random.choice(self.follow_up_questions)
        
    def should_ask_question(self, conversation_length: int, user_turns: int) -> bool:
        if conversation_length < 3:
            return False
            
        if user_turns > 5 and random.random() < 0.3:
            return True
            
        if random.random() < 0.15:
            return True
            
        return False


class EnhancedEggieModel:
    def __init__(self, eggie_model):
        self.eggie = eggie_model
        self.memory = MemorySystem()
        self.question_generator = QuestionGenerator()
        self.user_turns: Dict[str, int] = {}
        
    def chat(self, user_id: str, user_message: str, context: Optional[str] = None) -> str:
        self.memory.extract_user_info(user_id, user_message)
        
        self.memory.add_to_short_term(user_id, "user", user_message)
        
        if user_id not in self.user_turns:
            self.user_turns[user_id] = 0
        self.user_turns[user_id] += 1
        
        short_term = self.memory.get_short_term(user_id)
        memory_summary = self.memory.get_memory_summary(user_id)
        
        enhanced_system_prompt = self._build_enhanced_system_prompt(memory_summary)
        
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        messages.extend(short_term)
        
        response = self.eggie.chat(user_message, messages)
        
        self.memory.add_to_short_term(user_id, "assistant", response)
        
        if self.question_generator.should_ask_question(len(short_term), self.user_turns[user_id]):
            question = self.question_generator.generate_question()
            response += f"\n\n{question}"
            
        self.memory.save_memories()
        
        return response
        
    def _build_enhanced_system_prompt(self, memory_summary: str) -> str:
        base_prompt = self.eggie.persona_config['persona']['system_prompt']
        
        if memory_summary and memory_summary != "暂无记忆":
            enhanced_prompt = f"{base_prompt}\n\n关于用户的信息：{memory_summary}\n\n请根据这些信息，以更个性化的方式与用户交流。"
        else:
            enhanced_prompt = base_prompt
            
        return enhanced_prompt
        
    def proactive_message(self, user_id: str) -> Optional[str]:
        if random.random() < 0.2:
            return self.question_generator.generate_question()
        return None
        
    def remember_important_info(self, user_id: str, key: str, value: str):
        self.memory.add_to_long_term(user_id, key, value)
        self.memory.save_memories()
        
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        return {
            "profile": self.memory.get_user_profile(user_id),
            "long_term_memory": self.memory.get_long_term(user_id),
            "short_term_memory": self.memory.get_short_term(user_id)
        }
