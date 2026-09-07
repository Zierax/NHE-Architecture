import os

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

path = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir="D:/hf_cache",
    resume_download=True,
)
print("QWEN DONE:", path, flush=True)
