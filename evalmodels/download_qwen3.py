from huggingface_hub import snapshot_download
import os

model_names = [
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-14B",
    # "Qwen/Qwen3-32B"
    "Qwen/Qwen3-4B-Base",
    # "Qwen/Qwen3-4B-Base"
]

base_dir = "models/"

for model in model_names:
    model_id = model.split("/")[-1]
    local_dir = os.path.join(base_dir, model_id)
    print(f"Downloading {model} to {local_dir}...")
    snapshot_download(
        repo_id=model,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
