import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def test_finetuned_model():
    print("=" * 60)
    print("测试微调后的Eggie模型")
    print("=" * 60)
    
    base_model_path = "./models/Qwen3-1.7B"
    lora_model_path = "./data/checkpoints/final_model"
    
    print(f"\n基础模型路径: {base_model_path}")
    print(f"LoRA模型路径: {lora_model_path}")
    
    print("\n正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print("✓ 分词器加载成功")
    
    print("\n正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 基础模型加载成功")
    
    print("\n正在加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = model.merge_and_unload()
    print("✓ LoRA适配器加载成功")
    print("✓ 模型合并完成")
    
    print("\n" + "=" * 60)
    print("开始对话测试")
    print("=" * 60)
    
    test_prompts = [
        "你好，我是新来的，能介绍一下自己吗？",
        "今天天气真好，你觉得呢？",
        "你觉得学习编程难吗？",
        "我喜欢打游戏，你有喜欢的游戏吗？",
        "如果可以拥有超能力，你想要什么？"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {prompt}")
        print(f"{'='*60}")
        
        messages = [
            {"role": "system", "content": "你是Eggie，一个15岁的活泼开朗的女孩，喜欢游戏和动漫，说话风格可爱俏皮。"},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nEggie的回答:")
        print(response)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_finetuned_model()
