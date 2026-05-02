"""
Cendre QLoRA Fine-tuning with Unsloth
RTX 5090 (32GB VRAM) | qwen3:32b
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers.utils import is_torch_bf16_greater_than_zero

# ============ 配置 ============
MODEL_NAME = "qwen3:32b"           # Ollama 模型路径（ollama list 里的名字）
MAX_SEQ_LENGTH = 4096              # 最大上下文长度
LOAD_IN_4BIT = True                # 4bit 量化加载（省显存）
USE_BF16 = is_torch_bf16_greater_than_zero()

TRAINING_OUTPUT_DIR = "./cendre_finetuned"
TRAINING_DATA_PATH = "./cendre_training_data.jsonl"

# 训练参数（RTX 5090 32GB 优化）
PER_DEVICE_BATCH_SIZE = 1          # 每 GPU 批次大小（32GB 卡可以开 2 但保险起见）
GRADIENT_ACCUMULATION_STEPS = 16   # 梯度累积 = 有效批次 16
LEARNING_RATE = 2e-4                # 学习率
EPOCHS = 3                         # 训练轮数
WARMUP_STEPS = 50                 # 预热步数
LORA_R = 64                        # LoRA rank
LORA_ALPHA = 128                   # LoRA alpha
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FPS = 10                           # 训练过程保存频率（每多少步保存一次）

# ============ 加载模型 ============
print("=" * 50)
print("加载模型中...")
print(f"BF16 支持: {USE_BF16}")
print("=" * 50)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# 添加 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 梯度检查点省显存
)

print("模型 + LoRA 加载完成")
print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

# ============ 加载训练数据 ============
print("\n" + "=" * 50)
print("加载训练数据...")
print("=" * 50)

def formatting_prompts_func(examples):
    """将数据格式化成 Qwen3/ChatML 格式"""
    EOS_TOKEN = tokenizer.eos_token
    
    texts = []
    for messages in examples['messages']:
        # 构建对话格式
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                text += f"<|im_start|>assistant\n{content}{EOS_TOKEN}<|im_end|>\n"
        
        if text.endswith("\n"):
            text = text[:-1]  # 去掉最后一个换行
        
        # 包装成对话格式
        texts.append(text)
    
    return {"text": texts}

# 加载 JSONL 数据集
dataset = load_dataset("json", data_files=TRAINING_DATA_PATH, split="train")
print(f"数据集大小: {len(dataset)} 条")

# 格式化
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names,
    desc="格式化数据"
)

print(f"格式化完成: {len(dataset)} 条")
print(f"示例文本(前200字):\n{dataset[0]['text'][:200]}...")

# ============ 开始训练 ============
print("\n" + "=" * 50)
print("开始 QLoRA 微调！")
print(f"设备: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"有效批次大小: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print("=" * 50)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    
    args=TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not USE_BF16,
        bf16=USE_BF16,
        
        logging_steps=10,
        save_steps=FPS,
        save_total_limit=3,
        
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        
        report_to="none",
        remove_unused_columns=False,
    ),
)

# 开始训练
trainer_stats = trainer.train()
print("\n训练完成！")
print(f"总训练时间: {trainer_stats.metrics['train_runtime']:.1f} 秒")
print(f"总步数: {trainer_stats.metrics['train_steps']}")

# ============ 保存适配器 ============
print("\n" + "=" * 50)
print("保存 LoRA 适配器...")
print("=" * 50)

ADAPTER_PATH = f"{TRAINING_OUTPUT_DIR}/cendre_lora_adapter"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"适配器已保存到: {ADAPTER_PATH}")

print("\n" + "=" * 50)
print("下一步操作:")
print("1. 合并适配器到基础模型")
print("2. 转换为 Ollama 模型格式")
print("=" * 50)