import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.helpers import load_model_config
from transformers import AutoTokenizer, AutoModelForCausalLM

print("正在测试本地模型加载...")

try:
    config = load_model_config()
    print(f"模型路径: {config['model']['name']}")
    
    print("\n正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code']
    )
    print("✓ 分词器加载成功")
    
    print("\n正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code'],
        device_map="cpu"
    )
    print("✓ 模型加载成功")
    
    print(f"\n模型参数量: {model.num_parameters():,}")
    print(f"模型类型: {type(model).__name__}")
    
    print("\n✅ 本地模型配置成功！")
    print("以后训练和推理都不需要挂梯子了。")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
