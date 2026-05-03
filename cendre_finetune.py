"""
Cendre QLoRA Fine-tuning with Unsloth
RTX 5090 | Qwen3-32B (local) | 494 training samples
"""
import os, torch, json, gc
torch.cuda.empty_cache()
from datasets import Dataset

# ===== 显存优化 =====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

# ===== 配置 =====
MODEL_NAME = "unsloth/qwen3-32b-bnb-4bit"  # ModelScope 上的预量化版
MAX_SEQ_LENGTH = 1024                        # 聊天数据单轮对话通常很短
LOAD_IN_4BIT = True
TRAINING_OUTPUT_DIR = "./cendre_finetuned"
TRAINING_DATA_PATH = "./cendre_training_data.jsonl"
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_STEPS = 50
LORA_R = 16                                   # 494条数据，rank 16 足够
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 只训 attention，跳过 MLP（SwiGLU 激活值最耗显存）
SAVE_STEPS = 50
USE_BF16 = True if torch.cuda.get_device_capability()[0] >= 8 else False

print("=" * 60)
print(f"模型: {MODEL_NAME}")
print(f"BF16: {USE_BF16}")
print(f"训练数据: {TRAINING_DATA_PATH}")
print(f"训练轮数: {EPOCHS}")
print(f"学习率: {LEARNING_RATE}")
print(f"LoRA rank: {LORA_R}")
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}, 显存: {gpu.total_memory/1024**3:.1f}GB")
print("=" * 60)

# ===== 加载训练数据 =====
with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]
dataset = Dataset.from_list(raw_data)
print(f"加载 {len(dataset)} 条训练数据")

# ===== 加载模型 =====
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
    # 从 ModelScope 镜像站下载（原脚本已在顶部设置 HF_ENDPOINT）
)

# ===== 添加 LoRA =====
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=MAX_SEQ_LENGTH,
)

print(f"模型加载完成，显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

# ===== 格式化函数（OpenAI chat format） =====
def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat)
print(f"数据格式化完成，样例：{dataset[0]['text'][:200]}...")

# ===== 训练 =====
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not USE_BF16,
        bf16=USE_BF16,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        output_dir=TRAINING_OUTPUT_DIR,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
        optim="adamw_8bit",           # 8-bit Adam 省 2-3GB
        ddp_find_unused_parameters=False,
    ),
)

# ===== 开始训练 =====
gc.collect()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
print(f"训练前显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
print("开始训练...")
trainer.train()
print("训练完成")

# ===== 保存 LoRA 权重 =====
model.save_pretrained(TRAINING_OUTPUT_DIR)
tokenizer.save_pretrained(TRAINING_OUTPUT_DIR)
print(f"LoRA 权重保存到: {TRAINING_OUTPUT_DIR}")
print(f"训练后显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
