import sys
import time
from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(
    repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
    allow_patterns=["UD-Q3_K_XL/*"],
    local_dir=r"D:\AI_Bundle\qwen38next",
    max_workers=4,
)
print(f"DONE in {time.time()-t0:.0f}s -> {path}", flush=True)
