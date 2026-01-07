#!/usr/bin/env python
"""
自动从最新checkpoint继续训练的脚本
绕过交互式选择，直接选择选项2
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lora_finetune_large import (
    load_config, list_checkpoints, load_and_preprocess_data, 
    preprocess_function, format_messages
)
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from datetime import datetime
import numpy as np

def auto_resume_from_latest():
    """自动从最新checkpoint继续训练"""
    print("=" * 80)
    print("LoRA微调 - 大数据集版本 (自动恢复模式)")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载配置
    config = load_config()
    
    # 自动选择最新checkpoint
    base_model = config['base_model']
    checkpoints = list_checkpoints(config['output_dir'])
    
    if not checkpoints:
        print("❌ 错误: 没有可用的checkpoint")
        print("将从头开始训练")
        config['model_name_or_path'] = base_model
        config['overwrite_output_dir'] = True
        resume_from_checkpoint = False
    else:
        # 从最新checkpoint继续
        latest_step, latest_path = checkpoints[-1]
        print(f"✓ 自动选择最新checkpoint继续训练")
        print(f"  Checkpoint: checkpoint-{latest_step}")
        print(f"  路径: {latest_path}")
        config['model_name_or_path'] = str(latest_path)
        config['overwrite_output_dir'] = False
        resume_from_checkpoint = str(latest_path)
    
    print("\n配置信息:")
    print(f"  模型: {config['model_name_or_path']}")
    print(f"  LoRA rank: {config['lora_r']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  训练轮数: {config['num_train_epochs']}")
    print(f"  批次大小: {config['per_device_train_batch_size']}")
    print(f"  梯度累积: {config['gradient_accumulation_steps']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  数据加载器: {config.get('dataloader_num_workers', 0)} workers, prefetch={config.get('dataloader_prefetch_factor', 2)}")
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
    format_start = datetime.now()
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
    format_time = (datetime.now() - format_start).total_seconds()
    print(f"✓ 数据格式化完成，耗时: {format_time:.2f}秒")
    
    # 分词计时
    tokenize_start = datetime.now()
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
    tokenize_time = (datetime.now() - tokenize_start).total_seconds()
    print(f"✓ 分词完成，耗时: {tokenize_time:.2f}秒")
    
    # 初始化DataCollator
    data_collator_base = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 使用标准DataCollator
    data_collator = data_collator_base
    
    output_dir = config['output_dir']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'lora_finetune_{timestamp}'
    
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
    
    print('\n' + '=' * 80)
    print('开始训练')
    print('=' * 80)
    print(f'训练参数:')
    print(f'  有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}')
    print(f'  总训练步数: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}')
    print(f'  保存间隔: 每 {training_args.save_steps} 步')
    print(f'  评估间隔: 每 {training_args.eval_steps} 步')
    print(f'  日志间隔: 每 {training_args.logging_steps} 步')
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
        if resume_from_checkpoint:
            print(f"\n从checkpoint恢复训练: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            print("\n从头开始训练")
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
        print(f"  python scripts/resume_from_latest_checkpoint.py")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    auto_resume_from_latest()
