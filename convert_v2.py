"""GGUF → HuggingFace safetensors 转换脚本 (v2)"""
import gguf, os, torch
from safetensors.torch import save_file
from tqdm import tqdm
import requests

gguf_path = "/root/autodl-tmp/ollama_models/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
out_dir = "./qwen3_hf"
os.makedirs(out_dir, exist_ok=True)

reader = gguf.GGUFReader(gguf_path)

# Get layer count
try:
    n_layers = int(reader.get_field("qwen3.block_count"))
except (TypeError, KeyError):
    try:
        n_layers = int(reader.get_field("llama.block_count"))
    except (TypeError, KeyError):
        n_layers = 60

print(f"层数: {n_layers}, 总张量: {len(reader.tensors)}")

def hf_name(n):
    if n == "token_embd.weight": return "model.embed_tokens.weight"
    if n == "output.weight": return "lm_head.weight"
    if n == "output_norm.weight": return "model.norm.weight"
    p = n.split(".")
    if p[0] != "blk": return None
    l = int(p[1])
    s = ".".join(p[2:])
    m = {
        "attn_q.weight": f"model.layers.{l}.self_attn.q_proj.weight",
        "attn_k.weight": f"model.layers.{l}.self_attn.k_proj.weight",
        "attn_v.weight": f"model.layers.{l}.self_attn.v_proj.weight",
        "attn_output.weight": f"model.layers.{l}.self_attn.o_proj.weight",
        "ffn_gate.weight": f"model.layers.{l}.mlp.gate_proj.weight",
        "ffn_down.weight": f"model.layers.{l}.mlp.down_proj.weight",
        "ffn_up.weight": f"model.layers.{l}.mlp.up_proj.weight",
        "attn_norm.weight": f"model.layers.{l}.input_layernorm.weight",
        "ffn_norm.weight": f"model.layers.{l}.post_attention_layernorm.weight",
    }
    return m.get(s)

print("下载配置文件...")
for f in ["config.json","tokenizer_config.json","tokenizer.json","generation_config.json"]:
    r = requests.get(f"https://hf-mirror.com/Qwen/Qwen3-32B/resolve/main/{f}", timeout=30)
    with open(os.path.join(out_dir,f),"wb") as fp:
        fp.write(r.content)
    print(f"  {f} ({len(r.content)}b)")

print("转换张量...")
tensors={}
ci=0
count=0
for t in tqdm(reader.tensors):
    hn = hf_name(t.name)
    if hn is None: continue
    tensors[hn] = torch.tensor(t.data.copy())
    count+=1
    if len(tensors)>=50:
        save_file(tensors, os.path.join(out_dir,f"model-{ci:05d}.safetensors"))
        tensors={}
        ci+=1
if tensors:
    save_file(tensors, os.path.join(out_dir,f"model-{ci:05d}.safetensors"))
print(f"OK! {count}张量 -> {out_dir}")
