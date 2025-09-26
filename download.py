from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="tencent/Hunyuan3D-2.1",
    repo_type="model",
    local_dir="checkpoint/Paint",
    allow_patterns="hunyuan3d-paintpbr-v2-1/*"
)
print(f"✅ Done! Saved Paint model to checkpoint/Paint")

snapshot_download(
    repo_id="tencent/Hunyuan3D-2.1",
    repo_type="model",
    local_dir="checkpoint/Shape",
    allow_patterns="hunyuan3d-dit-v2-1/*"
)
print(f"✅ Done! Saved Shape model to checkpoint/Shape")

snapshot_download(
    repo_id="tomjackson2023/rembg",
    repo_type="model",
    local_dir="checkpoint/Rembg",
    allow_patterns="u2net.onnx"
)

print("✅ Done! Saved Rembg model to checkpoint/Rembg")