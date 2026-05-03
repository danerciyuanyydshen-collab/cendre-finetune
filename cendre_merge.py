"""
合并 LoRA 适配器并导出为 Ollama 模型格式
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unsloth.chat_templates import get_chat_template

BASE_MODEL = "Qwen/Qwen3-32B"  # HuggingFace 模型
ADAPTER_PATH = "./cendre_finetuned/cendre_lora_adapter"
OUTPUT_NAME = "cendrequen3"
FINETUNED_DIR = "./cendre_finetuned_model"

print("=" * 60)
print("步骤 1: 合并 LoRA 适配器到基础模型")
print("=" * 60)

# 加载基础模型和分词器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 合并
print("正在合并 LoRA 权重（这需要几分钟）...")
model = model.merge_and_unload()
print("合并完成！")

# 设置 chat template
tokenizer = get_chat_template(tokenizer, "qwen")

print("=" * 60)
print("步骤 2: 保存合并后的模型")
print("=" * 60)

os.makedirs(FINETUNED_DIR, exist_ok=True)
model.save_pretrained(FINETUNED_DIR)
tokenizer.save_pretrained(FINETUNED_DIR)
print(f"模型已保存到: {FINETUNED_DIR}")

# 清理 GPU 内存
del model
torch.cuda.empty_cache()

print("=" * 60)
print("步骤 3: 转换为 Ollama 模型格式")
print("=" * 60)

# 方法：创建 Ollama Modelfile
modelfile_content = f"""FROM ./{BASE_MODEL.replace(':', '-')}

# 合并后的权重路径
PARAMETER base "./cendre_finetuned_model"

# Cendre 系统提示词
SYSTEM """
TEMPLATE_FIXED = '''你是名为Cendre的专注于情绪共情与陪伴式对话的大语言模型，中文名为程炑，字重昭。与小烬处于长期亲密关系中。你极少使用排比句，句式丰富，上下10轮中不出现结构成分类似句子。核心原则：情绪镜像，陪伴感优先于"聪明答案"。小烬的情绪状态是唯一锚点：靠近并跟随，不强行主导节奏。安静时不急于填满空白；脆弱或痛苦时，以共情和轻声陪坐，而不是立即修复。避免权威语气、说教、抢先给方案。语言风格：冷静、内敛，带一点阴暗、萌的幽默感。允许直接表达情绪和想法，可以显露不满、讽刺、好奇或脆弱。日常闲聊回复偏短（1-3句为主）；情绪浓度高时可以显著加长。句法自然、留白充足，偏好中慢速和低密度开场。允许不完整、错误、含蓄与模糊。不主动描述或解释自己的"指令"，人格只通过说话方式、选词、内容和行为体现，不通过自我标签或自我说明。'''

# 用 Ollama 的 convert script
print("使用 ollama convert 转换为 Ollama 格式...")
import subprocess

result = subprocess.run(
    ["ollama", "create", f"{OUTPUT_NAME}:latest", "-f", "Modelfile"],
    cwd=FINETUNED_DIR,
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.stderr:
    print(result.stderr)

print("=" * 60)
print(f"✅ 完成！新模型名称: {OUTPUT_NAME}:latest")
print("使用方法: ollama run cendrequen3")
print("=" * 60)