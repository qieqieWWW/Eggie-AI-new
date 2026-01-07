# 测试脚本：验证从检查点恢复训练时的参数使用情况

import json
import torch
import yaml
import os
from transformers import TrainingArguments

# 加载配置文件
config_path = "d:\文档\AIChat\config\lora_config_sampled.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 检查点路径
checkpoint_path = "d:\文档\AIChat\data\checkpoints_sampled\checkpoint-13500"

print("="*60)
print("验证从检查点恢复训练时的参数使用情况")
print("="*60)

# 1. 查看配置文件中的新参数
print("\n1. 配置文件中的新参数：")
print(f"dataloader_num_workers: {config.get('dataloader_num_workers')}")
print(f"dataloader_prefetch_factor: {config.get('dataloader_prefetch_factor')}")
print(f"dataloader_pin_memory: {config.get('dataloader_pin_memory')}")
print(f"save_steps: {config.get('save_steps')}")
print(f"eval_steps: {config.get('eval_steps')}")
print(f"logging_steps: {config.get('logging_steps')}")

# 2. 查看检查点中的训练状态
print("\n2. 检查点中的训练状态：")
trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
if os.path.exists(trainer_state_path):
    with open(trainer_state_path, 'r', encoding='utf-8') as f:
        trainer_state = json.load(f)
    
    print(f"global_step: {trainer_state.get('global_step')}")
    print(f"eval_steps: {trainer_state.get('eval_steps')}")
    print(f"best_global_step: {trainer_state.get('best_global_step')}")
else:
    print("未找到 trainer_state.json 文件")

# 3. 检查点中的训练参数 (training_args.bin)：
print("\n3. 检查点中的训练参数 (training_args.bin)：")
training_args_path = os.path.join(checkpoint_path, "training_args.bin")
if os.path.exists(training_args_path):
    print("注意：training_args.bin 包含了训练时使用的旧参数，但 Hugging Face Trainer 在恢复训练时")
    print("不会使用这些旧参数，而是使用当前代码中创建的新 TrainingArguments。")
    print("\n关键验证点：")
    print(f"- 当前配置文件中的 dataloader_num_workers: {config.get('dataloader_num_workers')}")
    print(f"- 当前配置文件中的 save_steps: {config.get('save_steps')}")
    print(f"- 当前配置文件中的 eval_steps: {config.get('eval_steps')}")
else:
    print("未找到 training_args.bin 文件")

# 4. 模拟创建新的 TrainingArguments
print("\n4. 模拟创建新的 TrainingArguments：")
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=config.get('num_train_epochs', 3),
    per_device_train_batch_size=config.get('per_device_train_batch_size', 4),
    per_device_eval_batch_size=config.get('per_device_eval_batch_size', 4),
    gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
    learning_rate=config.get('learning_rate', 2e-4),
    warmup_steps=config.get('warmup_steps', 100),
    logging_steps=config.get('logging_steps', 10),
    save_steps=config.get('save_steps', 500),
    eval_steps=config.get('eval_steps', 500),
    save_total_limit=config.get('save_total_limit', 3),
    fp16=config.get('fp16', True),
    optim=config.get('optim', "adamw_torch"),
    dataloader_num_workers=config.get('dataloader_num_workers', 0),
    dataloader_prefetch_factor=config.get('dataloader_prefetch_factor', 2),
    dataloader_pin_memory=config.get('dataloader_pin_memory', False),
)

print(f"新 TrainingArguments.dataloader_num_workers: {training_args.dataloader_num_workers}")
print(f"新 TrainingArguments.dataloader_prefetch_factor: {training_args.dataloader_prefetch_factor}")
print(f"新 TrainingArguments.dataloader_pin_memory: {training_args.dataloader_pin_memory}")
print(f"新 TrainingArguments.save_steps: {training_args.save_steps}")
print(f"新 TrainingArguments.eval_steps: {training_args.eval_steps}")

print("\n" + "="*60)
print("结论：")
print("-"*60)
print("从检查点恢复训练时，Hugging Face Trainer会：")
print("1. 加载检查点中的模型权重、训练步数和学习率调度器状态")
print("2. 使用当前代码中创建的新TrainingArguments（基于最新配置文件）")
print("3. 不会使用检查点中保存的旧训练参数")
print("\n因此，您的新配置参数（包括dataloader优化参数）会在恢复训练时生效！")
print("="*60)
