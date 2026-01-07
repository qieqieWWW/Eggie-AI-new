import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import load_model_config, load_persona_config


class EggieModel:
    def __init__(self):
        self.model_config = load_model_config()
        self.persona_config = load_persona_config()
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def load_model(self):
        print("正在加载 Eggie 的模型...")
        
        quantization_config = None
        if self.model_config['model']['quantization']['enabled']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.model_config['model']['quantization']['load_in_4bit'],
                bnb_4bit_compute_dtype=getattr(torch, self.model_config['model']['quantization']['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=self.model_config['model']['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=self.model_config['model']['quantization']['bnb_4bit_quant_type']
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model']['name'],
            trust_remote_code=self.model_config['model']['trust_remote_code'],
            cache_dir=self.model_config['model']['cache_dir']
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model']['name'],
            quantization_config=quantization_config,
            device_map=self.model_config['model']['device_map'],
            trust_remote_code=self.model_config['model']['trust_remote_code'],
            low_cpu_mem_usage=self.model_config['model']['low_cpu_mem_usage'],
            cache_dir=self.model_config['model']['cache_dir']
        )
        
        self.device = self.model.device
        print(f"模型加载完成！设备: {self.device}")
        
    def chat(self, user_message: str, conversation_history: Optional[list] = None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        system_prompt = self.persona_config['persona']['system_prompt']
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        
        inference_config = self.model_config['inference']
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=inference_config['max_new_tokens'],
                temperature=inference_config['temperature'],
                top_p=inference_config['top_p'],
                top_k=inference_config['top_k'],
                repetition_penalty=inference_config['repetition_penalty'],
                do_sample=inference_config['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def interactive_chat(self):
        print("\n" + "="*50)
        print("Eggie 已启动！输入 'quit' 或 'exit' 退出")
        print("="*50 + "\n")
        
        conversation_history = []
        
        while True:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\nEggie: 拜拜啦！下次再聊～ 😊")
                break
            
            if not user_input:
                continue
            
            try:
                response = self.chat(user_input, conversation_history)
                print(f"\nEggie: {response}\n")
                
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                    
            except Exception as e:
                print(f"\n错误: {e}\n")


if __name__ == "__main__":
    eggie = EggieModel()
    eggie.load_model()
    eggie.interactive_chat()