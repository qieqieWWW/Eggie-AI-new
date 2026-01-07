import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.helpers import load_model_config, load_lora_config


class LoRATrainer:
    def __init__(self):
        self.model_config = load_model_config() #加载模型配置
        self.lora_config = load_lora_config()#加载LoRA配置
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        print("正在加载模型和分词器...")
        
        quantization_config = None
        if self.model_config['model']['quantization']['enabled']:
            from transformers import BitsAndBytesConfig
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
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model']['name'],
            quantization_config=quantization_config,
            device_map=self.model_config['model']['device_map'],
            trust_remote_code=self.model_config['model']['trust_remote_code'],
            low_cpu_mem_usage=self.model_config['model']['low_cpu_mem_usage'],
            cache_dir=self.model_config['model']['cache_dir']
        )
        
        print("模型和分词器加载完成！")
        
    def prepare_model_for_lora(self):
        print("正在准备模型进行LoRA微调...")
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.lora_config['lora']['r'],
            lora_alpha=self.lora_config['lora']['lora_alpha'],
            target_modules=self.lora_config['lora']['target_modules'],
            lora_dropout=self.lora_config['lora']['lora_dropout'],
            bias=self.lora_config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        self.model.print_trainable_parameters()
        
        print("LoRA配置完成！")
        
    def load_dataset(self, train_path: str, val_path: str = None):
        print(f"正在加载数据集...")
        
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        print(f"训练集加载完成，共 {len(train_dataset)} 条样本")
        
        val_dataset = None
        if val_path and Path(val_path).exists():
            val_dataset = load_dataset("json", data_files=val_path, split="train")
            print(f"验证集加载完成，共 {len(val_dataset)} 条样本")
        
        return train_dataset, val_dataset
        
    def preprocess_function(self, examples):
        texts = []
        
        for messages in examples['messages']:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        tokenized_inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.lora_config['training']['max_seq_length'],
            padding=False,
            return_tensors=None
        )
        
        return tokenized_inputs
        
    def prepare_dataset(self, dataset):
        print("正在预处理数据...")
        
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        print("数据预处理完成！")
        
        return tokenized_dataset
        
    def setup_trainer(self, train_dataset, eval_dataset=None):
        print("正在设置训练器...")
        
        training_args = TrainingArguments(
            output_dir=self.lora_config['training']['output_dir'],
            num_train_epochs=self.lora_config['training']['num_train_epochs'],
            per_device_train_batch_size=self.lora_config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.lora_config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.lora_config['training']['gradient_accumulation_steps'],
            learning_rate=self.lora_config['training']['learning_rate'],
            weight_decay=self.lora_config['training']['weight_decay'],
            warmup_steps=self.lora_config['training']['warmup_steps'],
            logging_steps=self.lora_config['training']['logging_steps'],
            save_steps=self.lora_config['training']['save_steps'],
            eval_steps=self.lora_config['training']['eval_steps'],
            fp16=self.lora_config['training']['fp16'],
            max_grad_norm=self.lora_config['training']['max_grad_norm'],
            optim=self.lora_config['training']['optim'],
            lr_scheduler_type=self.lora_config['training']['lr_scheduler_type'],
            eval_strategy=self.lora_config['training']['eval_strategy'],
            save_strategy=self.lora_config['training']['save_strategy'],
            load_best_model_at_end=self.lora_config['training']['load_best_model_at_end'],
            report_to=self.lora_config['training']['report_to'],
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("训练器设置完成！")
        
    def train(self):
        print("开始训练...")
        
        self.trainer.train()
        
        print("训练完成！")
        
    def save_model(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.lora_config['training']['output_dir'] + "/final_model"
        
        print(f"正在保存模型到: {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("模型保存完成！")


def main():
    trainer = LoRATrainer()
    
    trainer.load_model_and_tokenizer()
    trainer.prepare_model_for_lora()
    
    train_dataset, val_dataset = trainer.load_dataset(
        "data/lccc_processed/train_data.jsonl",
        "data/lccc_processed/val_data.jsonl"
    )
    
    tokenized_train_dataset = trainer.prepare_dataset(train_dataset)
    tokenized_val_dataset = trainer.prepare_dataset(val_dataset) if val_dataset else None
    
    trainer.setup_trainer(tokenized_train_dataset, tokenized_val_dataset)
    trainer.train()
    trainer.save_model()
    
    print("\nLoRA微调完成！模型已保存。")


if __name__ == "__main__":
    main()
