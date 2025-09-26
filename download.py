from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="tencent/Hunyuan3D-2.1",
    repo_type="model",
    local_dir="checkpoint/Paint",
    allow_patterns="hunyuan3d-paintpbr-v2-1/*"
)

snapshot_download(
    repo_id="tencent/Hunyuan3D-2.1",
    repo_type="model",
    local_dir="checkpoint/Shape",
    allow_patterns="hunyuan3d-dit-v2-1/*"
)