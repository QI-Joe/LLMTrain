# Llama3 情感分类微调项目

针对EmpatheticDialogues数据集的Llama3模型情感分类微调框架

## 📋 项目概述

本项目提供了一个完整的骨架架构，用于使用Llama3模型对EmpatheticDialogues数据集进行32类情感分类微调。

**核心特性**:
- ✅ 支持Dialogue Window机制 (IAMM论文)
- ✅ Few-shot / Semi-supervised学习支持
- ✅ 完整的结果记录和可视化 (TensorboardX)
- ✅ 灵活的配置管理
- ✅ 模块化设计，易于扩展

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n llama3_finetune python=3.10
conda activate llama3_finetune

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes
pip install datasets scikit-learn pandas numpy
pip install tensorboardX tqdm
```

### 2. 数据准备

```bash
# 首先需要使用IAMM项目生成预处理数据
cd src/utils/data
python loader.py  # 生成 data/dataset_preproc.p

# 测试数据加载
cd ../../..
python data_loader_llama3.py
```

**输出示例**:
```
==================================================
测试数据加载
==================================================
加载数据: ./data/dataset_preproc.p
训练集: 19533 样本
验证集: 2770 样本
测试集: 2547 样本

--- Few-shot Sampling ---
Few-shot采样: 512 样本 (16 shots/class)

--- Semi-supervised Sampling ---
Semi-supervised采样: 1953 labeled, 17580 unlabeled

--- Dialogue Window ---
原始对话轮数: 8
裁剪后轮数: 3
```

### 3. 开始训练

```bash
# 基础训练
python train_llama3.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --dialogue_window 3

# Few-shot训练 (16-shot)
python train_llama3.py \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 20 \
    --dialogue_window 3 \
    --shots_per_class 16
```

### 4. 查看结果

```bash
# 启动TensorBoard
tensorboard --logdir=./outputs/Llama3_Day --port=6006

# 浏览器访问 http://localhost:6006
```

## 📁 项目结构

```
Llama_train/
├── train_llama3.py              # 训练脚本骨架
├── config_llama3.py             # 配置文件
├── data_loader_llama3.py        # 数据加载器
├── utils_llama3.py              # 工具函数 (日志、TensorBoard)
├── data/                        # 数据目录
│   └── empatheticdialogues/
├── outputs/                     # 输出目录
│   └── Llama3_Day/
│       └── {param1}_{param2}_{param3}/
│           ├── TensorBoard日志
│           ├── 配置备份
│           ├── 训练日志
│           ├── 最终结果
│           └── checkpoints/
└── markdown/
    └── Llama3微调指南.md        # 详细指南
```

## 📖 核心文件说明

### 1. config_llama3.py - 配置文件

管理所有训练超参数，包括：
- 模型参数 (模型名称、类别数)
- 数据参数 (dialogue_window、batch_size)
- 训练参数 (学习率、epoch数)
- Few-shot设置
- 日志和检查点配置

**关键配置**:
```python
config = TrainingConfig(
    model_name="meta-llama/Meta-Llama-3-8B",
    num_emotions=32,
    dialogue_window=3,  # IAMM dialogue window
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=10
)
```

### 2. 数据加载器 - 对齐IAMM格式

**核心改进**:
- ✅ 直接加载 `src/utils/data/loader.py` 生成的 `dataset_preproc.p`
- ✅ 对齐原有IAMM数据格式
- ✅ 精简到~200行代码
- ✅ 专注于采样和窗口裁剪

**实现功能**:

1. **数据加载** - `load_empathetic_data()`
   ```python
   # 加载预处理数据
   data_tra, data_val, data_tst, vocab = load_empathetic_data("./data")
   ```

2. **Few-shot采样** - `sample_few_shot()`
   ```python
   # 每个类别采样16个样本
   indices = sample_few_shot(data_tra, shots_per_class=16)
   ```

3. **Semi-supervised采样** - `sample_semi_supervised()`
   ```python
   # 10%有标签，90%无标签
   labeled, unlabeled = sample_semi_supervised(data_tra, labeled_ratio=0.1)
   ```

4. **Dialogue Window裁剪** - `clip_dialogue_window()`
   ```python
   # 提取最后3轮对话
   windowed = clip_dialogue_window(context, window_size=3)
   ```

**关键特性**:
- 数据维度: `[batch, seq_length]` (多轮对话已拼接)
- 自动应用dialogue window机制
- Few-shot采样在DataLoader创建时自动应用
- 兼容标准Transformer输入格式

### 3. train_llama3.py - 训练脚本

骨架架构，包含：
- 模型加载与初始化 (TODO)
- 训练循环框架
- 评估流程
- 检查点保存
- 结果记录集成

**设计理念**: 提供清晰的结构，具体实现留给用户自定义

### 4. utils_llama3.py - 工具函数

提供：
- `ResultRecorder`: TensorboardX集成
- `setup_logger`: 日志配置
- `set_seed`: 随机种子
- 其他辅助函数

**TensorBoard记录**:
- Loss (每100步)
- 评估指标 (每1000步)
- 学习率曲线
- 自动组织实验文件夹

## 🎯 核心功能

### Dialogue Window机制

来自IAMM论文的关键创新：

```python
# 原始对话: 8轮
context = [
    ["Hi", "how", "are", "you"],
    ["I'm", "fine"],
    ["That's", "good"],
    ["What", "about", "you"],
    ["I'm", "feeling", "great"],
    ["Sounds", "wonderful"],
    ["Yes", "I'm", "very", "happy"],
    ["That's", "awesome"]
]

# dialogue_window=3: 提取最后3轮
windowed = clip_dialogue_window(context, window_size=3)
# 结果: [
#   ["Yes", "I'm", "very", "happy"],
#   ["That's", "awesome"],
# ]

# 拼接后送入模型: [batch, seq_length]
# "Yes I'm very happy That's awesome"
```

**优势**:
- ✅ 减少计算量（不处理完整历史）
- ✅ 聚焦于最近上下文（最相关的信息）
- ✅ 避免信息过载
- ✅ 直接输出 `[batch, seq_length]` 格式

### Few-shot Learning

支持每个类别固定样本数的训练：

```python
config.few_shot = True
config.shots_per_class = 16  # 每个情感类别16个样本

# 总训练样本: 32 * 16 = 512
```

### 实验管理

自动组织实验文件夹：

```
outputs/Llama3_Day/
├── lr_1e-5_bs_8_ws_3/     # 实验1
├── lr_2e-5_bs_8_ws_3/     # 实验2
└── lr_2e-5_bs_16_ws_5/    # 实验3
```

每个实验包含：
- TensorBoard日志
- 配置备份 (config.json)
- 训练日志 (training.log)
- 最终结果 (final_results.json)
- 检查点文件夹

## 📊 评估指标

自动计算和记录：

| 指标 | 说明 |
|-----|------|
| **Accuracy** | 整体准确率 |
| **F1 (Macro)** | 宏平均F1 (主要指标) |
| **F1 (Weighted)** | 加权F1 |
| **Precision** | 查准率 |
| **Recall** | 查全率 |

## 🔧 推荐微调方法

详见 [markdown/Llama3微调指南.md](markdown/Llama3微调指南.md) 第5节

### LoRA (推荐)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    task_type="SEQ_CLS"
)

model = get_peft_model(base_model, lora_config)
```

**优势**:
- 显存占用低 (~18GB for Llama3-8B)
- 训练速度快
- 效果接近全量微调

### QLoRA (显存受限时)

添加4bit量化，进一步降低显存需求到 ~12GB

## 📚 详细文档

完整指南请参考: **[markdown/Llama3微调指南.md](markdown/Llama3微调指南.md)**

包含：
1. 项目概述
2. 环境准备
3. 数据准备详解
4. 模型加载与初始化
5. **微调方法对比** (LoRA, QLoRA, Full Fine-tuning等)
6. 训练流程详解
7. 评估与指标
8. 结果记录与可视化
9. 实验管理
10. 常见问题解答

## ⚡ 使用建议

### 1. 开始前

1. **阅读详细指南**: [markdown/Llama3微调指南.md](markdown/Llama3微调指南.md)
2. **生成预处理数据**: 运行 `src/utils/data/loader.py` 生成 `dataset_preproc.p`
3. **理解Dialogue Window机制**: 参考IAMM论文和DataLoader详细解析
4. **配置环境**: 确保GPU显存足够

### 2. 实现顺序

建议按以下顺序填充TODO部分：

1. **测试数据加载** (data_loader_llama3.py)
   ```bash
   python data_loader_llama3.py
   ```
   - 验证数据格式正确
   - 查看采样结果
   - 确认dialogue window工作

2. **模型加载** (train_llama3.py)
   - 实现 `load_model()` 函数
   - 配置LoRA (如果使用)
   - 测试模型前向传播

3. **训练循环** (train_llama3.py)
   - 实现 `train_epoch()` 函数
   - 添加梯度累积、混合精度等优化
   - 测试一个epoch

4. **评估** (train_llama3.py)
   - 实现 `evaluate()` 函数
   - 集成TensorBoard记录
   - 验证指标计算

### 3. 调试技巧

```python
# 使用小数据测试
config.batch_size = 2
train_loader = DataLoader(dataset[:10])  # 只用10个样本

# 快速验证
config.log_interval = 1
config.eval_interval = 10

# 过拟合测试 (检查模型capacity)
# 使用10个样本，训练到loss接近0
```

## 🤝 自定义扩展

骨架设计允许轻松扩展：

### 添加新指标

```python
# 在 evaluate() 中添加
from sklearn.metrics import matthews_corrcoef

metrics["mcc"] = matthews_corrcoef(all_labels, all_preds)
```

### 使用不同模型

```python
# 修改 config_llama3.py
config.model_name = "meta-llama/Meta-Llama-3-70B"
# 或
config.model_name = "./local_model_path"
```

### 添加数据增强

```python
# 在 Dataset.__getitem__() 中
if self.augmentation:
    utterance = back_translate(utterance)
```

## 📝 引用

如果使用本框架，请引用相关论文：

```bibtex
@article{llama3,
  title={Llama 3 Model Card},
  author={Meta AI},
  year={2024}
}

@inproceedings{rashkin2019towards,
  title={Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset},
  author={Rashkin, Hannah and Smith, Eric Michael and Li, Margaret and Boureau, Y-Lan},
  booktitle={ACL},
  year={2019}
}

@inproceedings{iamm,
  title={An Iterative Associative Memory Model for Empathetic Response Generation},
  author={...},
  booktitle={...},
  year={...}
}
```

## 🐛 故障排除

常见问题及解决方案请查看指南第10节

**快速检查清单**:
- [ ] CUDA可用且版本兼容
- [ ] 数据路径正确
- [ ] Tokenizer加载成功
- [ ] 模型类别数 = 32
- [ ] Batch输入维度正确: [batch, turn, seq_length]

## 📧 联系方式

如有问题，请参考：
1. [markdown/Llama3微调指南.md](markdown/Llama3微调指南.md) - 详细文档
2. [markdown/DataLoader详细解析.md](markdown/DataLoader详细解析.md) - 数据加载机制
3. GitHub Issues (如果开源)

---

**版本**: v1.0  
**日期**: 2026-01-22  
**许可**: MIT License

祝训练顺利! 🎉
