"""
Cendre QLoRA Fine-tuning with Unsloth
RTX 5090 (32GB VRAM) | Qwen3-32B

鐢ㄦ硶: python cendre_finetune.py
"""

import os
import sys
import torch

# ============ 閰嶇疆 ============
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
USE_BF16 = True  # RTX 5090 + PyTorch 2.8 鏀寔 bf16

TRAINING_OUTPUT_DIR = "./cendre_finetuned"
TRAINING_DATA_PATH = "./cendre_training_data.jsonl"

# 璁粌鍙傛暟锛圧TX 5090 32GB 浼樺寲锛?PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_STEPS = 50
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SAVE_STEPS = 100

# ============ 鍔犺浇妯″瀷 ============
print("=" * 50)
print("鍔犺浇妯″瀷涓?..")
print(f"妯″瀷: {MODEL_NAME}")
print(f"BF16 鏀寔: {USE_BF16}")
print("=" * 50)

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# 娣诲姞 LoRA 閫傞厤鍣?model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print("妯″瀷 + LoRA 鍔犺浇瀹屾垚")
print(f"鏄惧瓨鍗犵敤: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

# ============ 鍔犺浇璁粌鏁版嵁 ============
from datasets import load_dataset

print("\n" + "=" * 50)
print("鍔犺浇璁粌鏁版嵁...")
print("=" * 50)

def formatting_prompts_func(examples):
    EOS_TOKEN = tokenizer.eos_token
    texts = []
    for messages in examples['messages']:
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
            text = text[:-1]
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=TRAINING_DATA_PATH, split="train")
print(f"鏁版嵁闆嗗ぇ灏? {len(dataset)} 鏉?)

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names,
    desc="鏍煎紡鍖栨暟鎹?
)

print(f"鏍煎紡鍖栧畬鎴? {len(dataset)} 鏉?)
print(f"绀轰緥鏂囨湰(鍓?00瀛?:\n{dataset[0]['text'][:200]}...")

# ============ 寮€濮嬭缁?============
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

print("\n" + "=" * 50)
print("寮€濮?QLoRA 寰皟锛?)
print(f"璁惧: {torch.cuda.get_device_name(0)}")
print(f"鏄惧瓨: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"鏈夋晥鎵规澶у皬: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
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
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    ),
)

trainer_stats = trainer.train()
print("\n璁粌瀹屾垚锛?)
print(f"鎬昏缁冩椂闂? {trainer_stats.metrics['train_runtime']:.1f} 绉?)
print(f"鎬绘鏁? {trainer_stats.metrics['train_steps']}")

# ============ 淇濆瓨閫傞厤鍣?============
print("\n" + "=" * 50)
print("淇濆瓨 LoRA 閫傞厤鍣?..")
print("=" * 50)

ADAPTER_PATH = f"{TRAINING_OUTPUT_DIR}/cendre_lora_adapter"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"閫傞厤鍣ㄥ凡淇濆瓨鍒? {ADAPTER_PATH}")

print("\n" + "=" * 50)
print("涓嬩竴姝? python cendre_merge.py")
print("=" * 50)