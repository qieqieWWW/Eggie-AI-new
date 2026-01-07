# 使用LCCC数据集训练Eggie

## 概述

LCCC (Large-scale Cleaned Chinese Conversation) 是清华大学开发的大规模中文对话数据集，包含超过1200万条高质量对话。这些对话来自微博、豆瓣、贴吧等多个平台，经过严格的数据清洗，非常适合用于训练像Eggie这样的中文对话AI。

## 数据集特点

- **LCCC-base**: 约670万条对话，清洗更严格，质量更高
- **LCCC-large**: 约1450万条对话，规模更大
- 数据来源：微博、豆瓣、贴吧、字幕、电商对话等
- 已过滤：脏字脏词、特殊字符、颜表情、语法不通、上下文不相关的对话

## 训练流程

### 步骤1: 安装依赖

```bash
pip install datasets
```

### 步骤2: 处理LCCC数据集

运行数据处理脚本，从LCCC数据集中筛选符合Eggie风格的对话：

```bash
python scripts/process_lccc.py
```

这个脚本会：
1. 自动从Hugging Face下载LCCC-base数据集
2. 筛选符合Eggie活泼开朗、有梗风格的对话
3. 过滤掉政治、暴力、色情等禁忌话题
4. 合并模板数据（15条Eggie人设对话）
5. 分割训练集和验证集（80:20）
6. 保存为JSONL格式

**输出文件**:
- `data/lccc_processed/train_data.jsonl` - 训练数据
- `data/lccc_processed/val_data.jsonl` - 验证数据

### 步骤3: 配置训练参数

编辑 `config/lora_config.yaml`，根据需要调整训练参数：

```yaml
training:
  num_train_epochs: 3              # 训练轮数
  per_device_train_batch_size: 4   # 批次大小
  learning_rate: 2e-4              # 学习率
  max_seq_length: 512              # 最大序列长度
  output_dir: "checkpoints/lora_eggie"  # 输出目录
```

### 步骤4: 开始LoRA微调

```bash
python scripts/lora_finetune.py
```

训练过程会：
1. 加载Qwen3-1.7B模型（4bit量化）
2. 配置LoRA参数
3. 加载处理好的LCCC数据
4. 进行微调训练
5. 保存训练好的模型

**输出目录**: `checkpoints/lora_eggie/final_model`

### 步骤5: 使用微调后的模型

修改 `config/model_config.yaml`，使用微调后的模型：

```yaml
model:
  name: "checkpoints/loca_eggie/final_model"  # 改为微调后的模型路径
```

然后运行：

```bash
python src/main.py
```

## 数据筛选策略

数据处理脚本使用以下策略筛选对话：

### 符合Eggie风格的关键词
- 语气词：哈哈、嘿嘿、嘻嘻、哎呀、哇、嗯、嗯嗯
- 语气助词：嘛、呢、呀、哦、诶
- 情感词：开心、有趣、好玩、厉害、棒、喜欢、爱
- 标点符号：～、!、?

### 过滤规则
- 回复长度：2-100字符
- 禁止话题：政治、暴力、色情、赌博
- 必须包含至少一个Eggie风格关键词

## 手动下载数据集

如果自动下载失败，可以手动下载：

1. 访问 https://github.com/thu-coai/CDial-GPT
2. 下载LCCC-base数据集
3. 将文件放在 `data/lccc_raw/` 目录下
4. 修改 `scripts/process_lccc.py` 中的加载逻辑

## 训练建议

### 数据量
- 初次训练：建议使用1000-5000条数据
- 正式训练：可以使用10000条或更多
- 数据越多，效果越好，但训练时间也越长

### 训练轮数
- 初次训练：1-2个epoch
- 观察验证集loss，选择最佳checkpoint
- 避免过拟合

### 批次大小
- 根据GPU显存调整
- 4bit量化下，通常可以设置4-8
- 如果显存不足，可以降低批次大小

### 学习率
- LoRA微调推荐：2e-4 到 5e-4
- 可以尝试不同的学习率
- 使用学习率调度器（如cosine）

## 监控训练

训练过程中会输出：
- 训练loss
- 验证loss
- 学习率变化
- 训练进度

建议：
- 观察loss曲线，确保正常下降
- 如果验证集loss上升，可能过拟合
- 保存多个checkpoint，选择最佳模型

## 评估模型

训练完成后，可以通过以下方式评估：

1. **交互式测试**:
   ```bash
   python src/main.py
   ```

2. **微信测试**:
   ```bash
   python src/wechat/wechat_bot.py
   ```

3. **对比测试**:
   - 对比微调前后的回复质量
   - 检查是否符合Eggie的人设
   - 评估回复的有趣程度和梗的使用

## 常见问题

### Q: 训练很慢怎么办？
A: 
- 减少训练数据量
- 降低批次大小
- 使用更少的epoch
- 使用更快的GPU

### Q: 显存不足？
A:
- 降低 `per_device_train_batch_size`
- 降低 `max_seq_length`
- 使用梯度累积

### Q: 效果不好？
A:
- 增加训练数据量
- 调整学习率
- 增加训练轮数
- 检查数据质量

### Q: 如何继续训练？
A:
- 使用 `resume_from_checkpoint` 参数
- 指定之前的checkpoint路径

## 下一步

训练完成后，可以：

1. **添加更多数据**:
   - 收集更多符合Eggie风格的对话
   - 使用其他数据集（如ShareGPT、Alpaca）
   - 手动创建更多训练样本

2. **优化模型**:
   - 调整LoRA参数（r、alpha）
   - 尝试不同的量化配置
   - 使用更大的模型

3. **部署应用**:
   - 接入微信
   - 开发Web界面
   - 部署到服务器

## 参考资料

- LCCC论文: https://arxiv.org/abs/2008.03946
- LCCC GitHub: https://github.com/thu-coai/CDial-GPT
- Hugging Face Datasets: https://huggingface.co/datasets/lccc
- LoRA论文: https://arxiv.org/abs/2106.09685
