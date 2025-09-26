from flask import Flask, request
import json
from flask_cors import CORS
import base64
import io

import sys
sys.path.insert(0, './hunyuan3d-2.1/hy3dshape')
sys.path.insert(0, './hunyuan3d-2.1/hy3dpaint')

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
import torch
import os

# ======================
# 模型初始化（全局只加载一次）
# ======================
model_path = 'checkpoint/Shape'
subfolder = 'hunyuan3d-dit-v2-1'

kwargs = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
use_safetensors = False
variant = 'fp16'

extension = 'ckpt' if not use_safetensors else 'safetensors'
variant_suffix = '' if variant is None else f'.{variant}'
ckpt_name = f'model{variant_suffix}.{extension}'
model_path = os.path.join(model_path, subfolder)
ckpt_path = os.path.join(model_path, ckpt_name)
config_path = os.path.join(model_path, 'config.yaml')

pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
    ckpt_path,
    config_path,
    device=device,
    dtype=dtype,
    use_safetensors=use_safetensors,
    **kwargs
)

# ======================
# Flask app
# ======================
app = Flask(__name__)
CORS(app, resources=r'/*')


@app.route('/shape/', methods=['POST'])
def shape_API():
    try:
        # 读取 POST JSON
        data = request.get_data() 
        data = json.loads(data)
        image_b64 = data.get("image")
        num_inference_steps = int(data.get("num_inference_steps", 50))

        # base64 解码为 PIL.Image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)

        # 运行生成
        with torch.inference_mode():
            mesh = pipeline_shapegen(image=image, num_inference_steps=num_inference_steps)[0]

        # 导出 GLB 文件
        out_path = "demo.glb"
        mesh.export(out_path)

        # 再次读出并转为 base64 作为返回
        with open(out_path, "rb") as f:
            glb_bytes = f.read()
        glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

        result = {
            "code": 1,
            "glb": glb_b64,
        }
    except Exception as e:
        result = {
            "code": 0,
            "error": str(e),
        }

    return json.dumps(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
