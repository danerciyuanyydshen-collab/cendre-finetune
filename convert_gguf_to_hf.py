"""GGUF → HuggingFace safetensors 转换脚本"""
import gguf, os, json
import torch
from safetensors.torch import save_file
from tqdm import tqdm
import requests

gguf_path = "/root/autodl-tmp/ollama_models/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
out_dir = "./qwen3_hf"
os.makedirs(out_dir, exist_ok=True)

reader = gguf.GGUFReader(gguf_path)

# 读取元数据
n_layers_field = reader.get_field("qwen3.block_count", None)
if n_layers_field is None:
    n_layers_field = reader.get_field("llama.block_count", None)
n_layers = int(n_layers_field or 60)
print(f"层数: {n_layers}")
print(f"总张量: {len(reader.tensors)}")


def gguf_to_hf_name(gguf_name):
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"
    parts = gguf_name.split(".")
    if parts[0] != "blk":
        return None
    layer = int(parts[1])
    suffix = ".".join(parts[2:])
    mapping = {
        "attn_q.weight": f"model.layers.{layer}.self_attn.q_proj.weight",
        "attn_k.weight": f"model.layers.{layer}.self_attn.k_proj.weight",
        "attn_v.weight": f"model.layers.{layer}.self_attn.v_proj.weight",
        "attn_output.weight": f"model.layers.{layer}.self_attn.o_proj.weight",
        "ffn_gate.weight": f"model.layers.{layer}.mlp.gate_proj.weight",
        "ffn_down.weight": f"model.layers.{layer}.mlp.down_proj.weight",
        "ffn_up.weight": f"model.layers.{layer}.mlp.up_proj.weight",
        "attn_norm.weight": f"model.layers.{layer}.input_layernorm.weight",
        "ffn_norm.weight": f"model.layers.{layer}.post_attention_layernorm.weight",
    }
    return mapping.get(suffix, None)


# 下载配置文件（仅 JSON，几 KB）
print("下载配置文件...")
base_url = "https://hf-mirror.com/Qwen/Qwen3-32B/resolve/main"
for fname in ["config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"]:
    try:
        r = requests.get(f"{base_url}/{fname}", timeout=30)
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(r.content)
        print(f"  OK {fname} ({len(r.content)} bytes)")
    except Exception as e:
        print(f"  FAIL {fname}: {e}")

# 转换张量
print("转换张量...")
tensors = {}
chunk_idx = 0
count = 0
for tensor in tqdm(reader.tensors):
    hf_name = gguf_to_hf_name(tensor.name)
    if hf_name is None:
        continue
    data = torch.tensor(tensor.data.copy())
    tensors[hf_name] = data
    count += 1
    if len(tensors) >= 50:
        fname = f"model-{chunk_idx:05d}.safetensors"
        save_file(tensors, os.path.join(out_dir, fname))
        print(f"  写入 {fname} ({len(tensors)} 张量)")
        tensors = {}
        chunk_idx += 1

if tensors:
    fname = f"model-{chunk_idx:05d}.safetensors"
    save_file(tensors, os.path.join(out_dir, fname))
    print(f"  写入 {fname} ({len(tensors)} 张量)")

print(f"\nOK! 转换 {count} 个张量到 {out_dir}")
print(f"文件: {os.listdir(out_dir)}")
