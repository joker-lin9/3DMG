# import sys
# sys.path.insert(0, './hunyuan3d-2.1/hy3dshape')
# sys.path.insert(0, './hunyuan3d-2.1/hy3dpaint')
# from textureGenPipeline import Hunyuan3DPaintPipeline
# from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
# from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
# import torch

# # let's generate a mesh first
# shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('checkpoint', subfolder='Shape')
# mesh_untextured = shape_pipeline(image='hunyuan3d-2.1/assets/demo.png')[0]

# # paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
# # mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')

# model_path = 'checkpoint'
# subfolder = 'Shape'
# kwargs = {}
# device='cuda',
# dtype=torch.float16,
# use_safetensors=False,
# variant='fp16',
# kwargs['from_pretrained_kwargs'] = dict(
#     model_path=model_path,
#     subfolder=subfolder,
#     use_safetensors=use_safetensors,
#     variant=variant,
#     dtype=dtype,
#     device=device,
# )

import sys
sys.path.insert(0, './hunyuan3d-2.1/hy3dshape')
sys.path.insert(0, './hunyuan3d-2.1/hy3dpaint')

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
import torch
import os

###可更换参数###
image_path = './hunyuan3d-2.1/assets/demo.png'
num_inference_steps = 50

###############

model_path = 'checkpoint/Shape'
subfolder = 'hunyuan3d-dit-v2-1'
image = Image.open(image_path).convert("RGBA")

kwargs = {}
device='cuda'
dtype= torch.float16
use_safetensors=False
variant='fp16'

extension = 'ckpt' if not use_safetensors else 'safetensors'
variant = '' if variant is None else f'.{variant}'
ckpt_name = f'model{variant}.{extension}'
model_path = os.path.join(model_path, subfolder)
ckpt_path = os.path.join(model_path, ckpt_name)
config_path = os.path.join(model_path, 'config.yaml')
ckpt_path = os.path.join(model_path, ckpt_name)

pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

mesh = pipeline_shapegen(image=image, num_inference_steps=num_inference_steps)
mesh = mesh[0]
mesh.export('demo.glb')

# paint
# max_num_view = 6  # can be 6 to 9
# resolution = 512  # can be 768 or 512
# conf = Hunyuan3DPaintConfig(max_num_view, resolution)
# conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
# conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
# conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
# paint_pipeline = Hunyuan3DPaintPipeline(conf)

# output_mesh_path = 'demo_textured.glb'
# output_mesh_path = paint_pipeline(
#     mesh_path = "demo.glb", 
#     image_path = 'assets/demo.png',
#     output_mesh_path = output_mesh_path
# )
