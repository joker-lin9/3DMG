## 一. 演示视频：
[./demo.mp4](./demo.mp4)

<video width="800" controls>
  <source src="./demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 二. 运行说明：

### 1. 环境搭建（Windows）
#### 环境创建
````
conda create -n Hunyuan3D python=3.10
conda activate Hunyuan3D
````

#### 环境安装
````
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/tencent-hunyuan/hunyuan3d-2.1
pip install -r requirements.txt
pip install cupy-cuda12x==13.4.1
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install deepspeed --prefer-binary
pip install waitress flask_cors
````

#### 编译文件替换
##### powershell
````
Copy-Item .\custom_rasterizer_setup.py .\hunyuan3d-2.1\hy3dpaint\custom_rasterizer\setup.py -Force
Copy-Item .\DifferentiableRenderer_setup.py .\hunyuan3d-2.1\hy3dpaint\DifferentiableRenderer\setup.py
````

##### bash
````
cp -f custom_rasterizer_setup.py hunyuan3d-2.1/hy3dpaint/custom_rasterizer/setup.py
cp -f DifferentiableRenderer_setup.py hunyuan3d-2.1/hy3dpaint/DifferentiableRenderer/setup.py
cd hunyuan3d-2.1\hy3dpaint\custom_rasterizer
pip install -e .
cd ..\DifferentiableRenderer
pip install -e .
````

#### 模型下载
````
cd ../.. 
mkdir hy3dpaint\ckpt
python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 'hy3dpaint\ckpt\RealESRGAN_x4plus.pth')"
cd ..
mkdir checkpoint
python download.py
````

### 2. 后端启动

```bash
run.bat
```

### 3. 打开 Web
文生图
* [index.html](index.html) 

模型生成
* [shape.html](shape.html)

### 三. 架构设计文档

[./架构设计文档.md](./架构设计文档.md)

### 四. 产品设计文档（包括议题答复）

[./产品设计文档.md](./产品设计文档.md)
