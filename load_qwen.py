from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 定义要加载的大语言模型名称（Qwen3-1.7B，通义千问3系列，1.7B参数，非量化版本）
MODEL_NAME = "Qwen/Qwen3-1.7B"

# 配置4bit量化参数
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,        # 启用4bit量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16精度
    bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步减少显存占用
    bnb_4bit_quant_type="nf4"  # 使用NF4量化类型（NormalFloat4），量化效果最好
)

# 1. 加载分词器（Tokenizer）：负责将自然语言文本转换为模型可理解的数字token
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,                # 指定要加载的模型对应的分词器名称
    trust_remote_code=True,    # 信任远程代码（Qwen模型需要自定义代码来实现分词/模型逻辑，必须开启）
    cache_dir=None             # 不指定缓存目录，使用transformers默认的缓存路径（通常是~/.cache/huggingface/）
)

# 2. 加载因果语言模型（AutoModelForCausalLM：适用于文本生成的自回归模型）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,                # 指定要加载的模型名称
    quantization_config=quantization_config,  # 使用BitsAndBytesConfig进行4bit量化
    device_map="auto",         # 自动分配模型到可用设备（优先GPU，无GPU则使用CPU）
    trust_remote_code=True,    # 信任远程代码（Qwen模型非HuggingFace原生架构，需要自定义代码加载，必须开启）
    low_cpu_mem_usage=True,    # 启用低CPU内存占用模式，加载模型时减少CPU内存消耗
    cache_dir=None             # 不指定缓存目录，使用transformers默认缓存路径
)

# 3. 定义对话消息列表（遵循ChatML格式，包含角色和对应的内容）
# role可选值：user（用户提问）、assistant（模型回复）、system（系统预设）
messages = [
    {"role": "user", "content": "Who are you?"},  # 用户向模型提出问题："你是谁？"
]

# 4. 对对话消息进行预处理，转换为模型可直接输入的格式
inputs = tokenizer.apply_chat_template(
    messages,                  # 传入定义好的对话消息列表
    add_generation_prompt=True,# 添加生成提示词（Qwen模型需要该参数来触发正确的回复生成，标记用户输入结束，模型开始回复）
    tokenize=True,             # 启用分词功能，将文本转换为模型可识别的数字token（input_ids等）
    return_dict=True,          # 返回字典格式的结果，方便通过键名访问数据（如inputs["input_ids"]）
    return_tensors="pt",       # 返回PyTorch张量（tensor）格式，适配PyTorch框架的模型
).to(model.device)             # 将预处理后的输入数据移动到模型所在的设备（GPU/CPU），保证数据与模型设备一致

# 5. 使用模型进行文本生成，得到输出结果
outputs = model.generate(
    **inputs,                  # 解包预处理后的输入数据（传入input_ids、attention_mask等必要参数）
    max_new_tokens=40          # 限制模型生成的新token数量最多为40个，避免生成过长文本
)

# 6. 解码模型输出结果，转换为人类可阅读的自然语言并打印
# 切片说明：outputs[0]取第一个生成结果（批量大小为1），[inputs["input_ids"].shape[-1]:] 跳过输入部分的token，只取模型新生成的内容
# 这样可以避免重复打印用户的提问，只显示模型的回复
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))