"""生成 model.safetensors.index.json（在 AutoDL 上跑）"""
import json, os, glob

out_dir = "/root/autodl-tmp/qwen3_hf"

# 读取 config.json 获取 hidden_size
with open(os.path.join(out_dir, "config.json")) as f:
    config = json.load(f)

# 找到所有 safetensors shard 文件
shard_files = sorted(glob.glob(os.path.join(out_dir, "model-*.safetensors")))
shards = []
total_size = 0
for sf in shard_files:
    size = os.path.getsize(sf)
    shards.append(os.path.basename(sf))
    total_size += size

print(f"找到 {len(shards)} 个 shard，总大小 {total_size/1024**3:.1f}GB")

# 对于 Qwen3-32B，weight_map 的生成需要知道每个 shard 包含哪些张量
# 从 safetensors 文件头读取张量名
from safetensors import safe_open

weight_map = {}
for shard_file in shard_files:
    shard_basename = os.path.basename(shard_file)
    with safe_open(shard_file, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            weight_map[key] = shard_basename

index = {
    "metadata": {"total_size": total_size},
    "weight_map": weight_map
}

with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"索引文件已生成，包含 {len(weight_map)} 个张量映射")
print(f"文件: {os.path.join(out_dir, 'model.safetensors.index.json')}")
