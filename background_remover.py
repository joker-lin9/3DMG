import os
from typing import Optional, List
from PIL import Image
from rembg import remove, new_session as _rembg_new_session


def new_local_session(
    model_name: str = "u2net",
    model_dir: Optional[str] = None,
    use_gpu: bool = False,
    extra_providers: Optional[List[str]] = None,
    **kwargs
):
    """
    创建 rembg 会话，但强制模型从本地目录 checkpoint/Rembg 加载/缓存。

    参数:
        model_name: 模型名（默认 "u2net"）
        model_dir: 自定义模型目录，默认使用 "checkpoint/Rembg"
        use_gpu: 是否优先使用 CUDAExecutionProvider
        extra_providers: 追加的 onnxruntime providers 列表（可选）
        **kwargs: 透传给 rembg.new_session 的其他参数
    """
    # 自定义模型保存/查找目录
    model_dir = model_dir or os.path.join("checkpoint", "Rembg")
    os.makedirs(model_dir, exist_ok=True)

    # 让 rembg / onnxruntime 只在该目录找模型（等价于覆盖 ~/.u2net）
    os.environ["U2NET_HOME"] = model_dir
    os.environ["REMBG_HOME"] = model_dir

    # 设备 providers（可根据环境装了 onnxruntime-gpu 与否决定是否真的走 CUDA）
    providers = ["CPUExecutionProvider"]
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if extra_providers:
        # 允许调用方追加/覆盖
        providers = extra_providers + [p for p in providers if p not in extra_providers]

    # 创建并返回官方会话对象
    return _rembg_new_session(model_name, providers=providers, **kwargs)


class BackgroundRemover:
    def __init__(self, model_name: str = "u2net", model_dir: Optional[str] = None, use_gpu: bool = False, **kwargs):
        # 这里会把模型下载/读取到 checkpoint/Rembg/<model_name>.onnx
        self.session = new_local_session(model_name=model_name, model_dir=model_dir, use_gpu=use_gpu, **kwargs)

    def __call__(self, image: Image.Image):
        # 与原逻辑一致：输出保持 RGBA，背景透明
        return remove(image, session=self.session, bgcolor=[255, 255, 255, 0])


# 示例
if __name__ == "__main__":
    # 首次运行会将 u2net.onnx 放到 checkpoint/Rembg 下；之后离线可直接读取
    br = BackgroundRemover(model_name="u2net", use_gpu=False)
    img = Image.open("input.jpg")
    out = br(img)
    out.save("output.png")
