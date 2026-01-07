# Eggie-AI

尝试构建一个能在微信里聊天、能点赞评论朋友圈、能查看屏幕状态陪你打游戏工作聊天的有独特性格的ai

## 项目简介
基于 Qwen3-1.7B 模型，通过 LoRA 微调创建具有鲜明个性和自主发问能力的 AI，接入微信进行自然对话。

## 项目结构

```
AIChat/
├── src/                      # 源代码目录
│   ├── models/              # 模型相关代码
│   │   ├── __init__.py
│   │   ├── model_loader.py  # 模型加载和推理
│   │   └── lora_trainer.py  # LoRA 微调训练
│   ├── wechat/              # 微信接入
│   │   ├── __init__.py
│   │   └── wechat_bot.py    # 微信机器人
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   └── helpers.py       # 辅助函数
│   └── memory/              # 记忆系统
│       ├── __init__.py
│       └── memory_system.py  # 对话记忆管理
├── data/                    # 数据目录
│   ├── training_data/       # 训练数据
│   │   └── conversations.jsonl
│   └── checkpoints/         # 模型检查点
├── config/                  # 配置文件
│   ├── model_config.yaml    # 模型配置
│   ├── lora_config.yaml     # LoRA 配置
│   └── persona_config.yaml  # AI 人设配置
├── logs/                    # 日志文件
├── load_qwen.py            # 模型加载测试脚本
└── requirements.txt         # 依赖包列表
```

## 开发阶段

1. ✅ 创建项目结构
2. ✅ 设计 AI 人设
3. ✅ 实现基础对话功能
4. ✅ 准备训练数据
5. ✅ LoRA 微调
6. ⏳ 微信接入
7. ⏳ 高级功能（记忆系统、自主发问）

## 技术栈

- **基础模型**: Qwen3-1.7B (4bit 量化)
- **微调框架**: PEFT (LoRA)
- **微信接入**: itchat / wechaty
- **记忆系统**: 向量数据库 (可选)
- **训练框架**: Transformers + PyTorch
