"""
LoRA微调脚本 - 大数据集版本
支持大规模数据集训练，具有更好的进度显示和checkpoint管理
"""

import os
os.environ['HF_HOME'] = 'D:/AIChat/cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/AIChat/cache/datasets'

import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys





def load_config(config_path: str = "config/lora_config_sampled.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def list_checkpoints(output_dir: str):
    """列出所有可用的checkpoints"""
    checkpoints = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return checkpoints
    
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            try:
                step_num = int(item.name.replace('checkpoint-', ''))
                checkpoints.append((step_num, item))
            except ValueError:
                continue
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def select_training_mode(config):
    """让用户选择训练模式"""
    output_dir = config['output_dir']
    checkpoints = list_checkpoints(output_dir)
    base_model = config.get('base_model', './models/Qwen3-1.7B')
    
    print("\n" + "=" * 80)
    print("训练模式选择")
    print("=" * 80)
    
    print(f"\n基础模型: {base_model}")
    
    if checkpoints:
        print(f"\n可用的Checkpoints:")
        for i, (step_num, path) in enumerate(checkpoints, 1):
            print(f"  {i}. checkpoint-{step_num} (路径: {path})")
    else:
        print(f"\n未找到任何checkpoint")
    
    print("\n请选择训练模式:")
    print("  1. 从头开始训练（使用基础模型）")
    
    if checkpoints:
        print("  2. 从最新checkpoint继续训练")
        print("  3. 选择特定checkpoint继续训练")
    else:
        print("  2. 从最新checkpoint继续训练（无可用checkpoint）")
        print("  3. 选择特定checkpoint继续训练（无可用checkpoint）")
    
    resume_from_checkpoint = None
    
    while True:
        choice = input("\n请输入选项 (1-3): ").strip()
        
        if choice == '1':
            print(f"\n✓ 将从头开始训练")
            print(f"  基础模型: {base_model}")
            config['model_name_or_path'] = base_model
            config['overwrite_output_dir'] = True
            resume_from_checkpoint = False
            return config, resume_from_checkpoint
        
        elif choice == '2':
            if not checkpoints:
                print("\n❌ 错误: 没有可用的checkpoint")
                continue
            
            latest_step, latest_path = checkpoints[-1]
            print(f"\n✓ 将从最新checkpoint继续训练")
            print(f"  Checkpoint: checkpoint-{latest_step}")
            print(f"  路径: {latest_path}")
            config['model_name_or_path'] = str(latest_path)
            config['overwrite_output_dir'] = False
            resume_from_checkpoint = True
            return config, resume_from_checkpoint
        
        elif choice == '3':
            if not checkpoints:
                print("\n❌ 错误: 没有可用的checkpoint")
                continue
            
            print("\n可用的Checkpoints:")
            for i, (step_num, path) in enumerate(checkpoints, 1):
                print(f"  {i}. checkpoint-{step_num}")
            
            while True:
                try:
                    checkpoint_choice = input(f"\n请输入checkpoint编号 (1-{len(checkpoints)}): ").strip()
                    idx = int(checkpoint_choice) - 1
                    
                    if 0 <= idx < len(checkpoints):
                        step_num, path = checkpoints[idx]
                        print(f"\n✓ 将从checkpoint-{step_num}继续训练")
                        print(f"  路径: {path}")
                        config['model_name_or_path'] = str(path)
                        config['overwrite_output_dir'] = False
                        resume_from_checkpoint = str(path)
                        return config, resume_from_checkpoint
                    else:
                        print(f"❌ 无效的编号，请输入 1-{len(checkpoints)}")
                except ValueError:
                    print("❌ 请输入有效的数字")
        
        else:
            print("❌ 无效的选项，请输入 1-3")


def load_and_preprocess_data(config):
    print(f"正在加载数据集...")
    print(f"训练文件: {config['dataset_path']}/{config['train_file']}")
    print(f"验证文件: {config['dataset_path']}/{config['validation_file']}")
    
    start_time = time.time()
    
    train_path = Path(config['dataset_path']) / config['train_file']
    val_path = Path(config['dataset_path']) / config['validation_file']
    
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"验证数据文件不存在: {val_path}")
    
    # 检查是否使用完整加载模式
    use_full_load = config.get('load_full_to_memory', False)
    
    load_start = time.time()
    if use_full_load:
        print("使用完整加载模式（数据加载到内存）")
        train_data = load_dataset('json', data_files=str(train_path), split='train', keep_in_memory=True)
        val_data = load_dataset('json', data_files=str(val_path), split='train', keep_in_memory=True)
    else:
        print("使用流式传输模式（节省内存）")
        train_data = load_dataset('json', data_files=str(train_path), split='train')
        val_data = load_dataset('json', data_files=str(val_path), split='train')
    load_time = time.time() - load_start
    print(f"数据加载耗时: {load_time:.2f}秒")
    
    print(f"训练数据: {len(train_data)} 条")
    print(f"验证数据: {len(val_data)} 条")
    
    total_time = time.time() - start_time
    print(f"数据加载和预处理总耗时: {total_time:.2f}秒")
    
    return train_data, val_data


def format_messages(messages, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text


def preprocess_function(examples, tokenizer, system_prompt):
    messages_batch = examples['messages']
    
    # 准备所有需要处理的消息列表
    processed_messages = []
    for messages in messages_batch:
        if messages[0]['role'] != 'system':
            # 直接创建新列表而不是拼接，更高效
            processed_msg = [{"role": "system", "content": system_prompt}]
            processed_msg.extend(messages)
            processed_messages.append(processed_msg)
        else:
            processed_messages.append(messages)
    
    # 批量应用chat template
    formatted_texts = tokenizer.apply_chat_template(
        processed_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_texts}





def main():
    print("=" * 80)
    print("LoRA微调 - 大数据集版本")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    config = load_config()
    
    # 让用户选择训练模式
    config, resume_from_checkpoint = select_training_mode(config)
    
    print("\n配置信息:")
    print(f"  模型: {config['model_name_or_path']}")
    print(f"  LoRA rank: {config['lora_r']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  训练轮数: {config['num_train_epochs']}")
    print(f"  批次大小: {config['per_device_train_batch_size']}")
    print(f"  梯度累积: {config['gradient_accumulation_steps']}")
    print(f"  学习率: {config['learning_rate']}")
    print()
    
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name_or_path'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ 分词器加载成功")
    
    print("正在加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    print("✓ 模型加载成功")
    
    print("正在配置LoRA...")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✓ LoRA配置完成")
    
    print("正在加载和预处理数据...")
    train_data, val_data = load_and_preprocess_data(config)
    
    # 格式化训练数据计时
    format_start = time.time()
    print("正在格式化训练数据...")
    train_dataset = train_data.map(
        lambda examples: preprocess_function(examples, tokenizer, config['system_prompt']),
        batched=True,
        num_proc=16,
        remove_columns=['messages']
    )
    val_dataset = val_data.map(
        lambda examples: preprocess_function(examples, tokenizer, config['system_prompt']),
        batched=True,
        num_proc=8,
        remove_columns=['messages']
    )
    format_time = time.time() - format_start
    print(f"数据格式化耗时: {format_time:.2f}秒")
    
    # 分词计时
    tokenize_start = time.time()
    print("正在分词...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    tokenize_time = time.time() - tokenize_start
    print(f"分词耗时: {tokenize_time:.2f}秒")
    
    print("✓ 数据预处理完成")
    
    # 初始化标准DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    output_dir = config['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lora_finetune_{timestamp}"
    
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": config['num_train_epochs'],
        "per_device_train_batch_size": config['per_device_train_batch_size'],
        "per_device_eval_batch_size": config['per_device_eval_batch_size'],
        "gradient_accumulation_steps": config['gradient_accumulation_steps'],
        "learning_rate": config['learning_rate'],
        "warmup_steps": config['warmup_steps'],
        "logging_steps": config['logging_steps'],
        "save_steps": config['save_steps'],
        "eval_steps": config['eval_steps'],
        "save_total_limit": config['save_total_limit'],
        "fp16": config['fp16'],
        "optim": config['optim'],
        "dataloader_num_workers": config.get('dataloader_num_workers', 0),
        "dataloader_prefetch_factor": config.get('dataloader_prefetch_factor', 2),
        "dataloader_pin_memory": config.get('dataloader_pin_memory', True),
        "logging_dir": config['logging_dir'],
        "overwrite_output_dir": config['overwrite_output_dir'],
        "save_strategy": config['save_strategy'],
        "eval_strategy": config['eval_strategy'],
        "load_best_model_at_end": config['load_best_model_at_end'],
        "metric_for_best_model": config['metric_for_best_model'],
        "greater_is_better": config['greater_is_better'],
        "report_to": config['report_to'],
        "remove_unused_columns": config['remove_unused_columns'],
        "run_name": run_name,
        "save_safetensors": True,
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
    }
    
    if config.get('max_steps') is not None:
        training_args_dict['max_steps'] = config['max_steps']
    
    training_args = TrainingArguments(**training_args_dict)
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"训练参数:")
    print(f"  有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  总训练步数: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    print(f"  保存间隔: 每 {training_args.save_steps} 步")
    print(f"  评估间隔: 每 {training_args.eval_steps} 步")
    print(f"  日志间隔: 每 {training_args.logging_steps} 步")
    print()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    try:
        if resume_from_checkpoint is not None and resume_from_checkpoint is not False:
            print(f"\n从checkpoint恢复训练: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        
        print("正在保存最终模型...")
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✓ 模型已保存到: {final_model_path}")
        
        print("\n训练总结:")
        print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  训练步数: {trainer.state.global_step}")
        print(f"  最佳模型: {trainer.state.best_model_checkpoint}")
        print(f"  最佳指标: {trainer.state.best_metric}")
        
        print("\n下一步:")
        print("  1. 使用 tensorboard --logdir=./data/logs 查看训练曲线")
        print("  2. 使用 python scripts/test_eggie.py 测试模型")
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("训练被中断")
        print("=" * 80)
        print(f"当前checkpoint已保存")
        print(f"可以使用以下命令继续训练:")
        print(f"  python scripts/lora_finetune_large.py")
        print(f"Trainer会自动从最新的checkpoint恢复")
        sys.exit(0)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
