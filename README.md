conda create -n Hunyuan3D python=3.10
conda activate Hunyuan3D

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
<!-- pip install termcolor transformers -->

git clone https://github.com/tencent-hunyuan/hunyuan3d-2.1

<!-- cd Hunyuan3D-2.1 -->
pip install -r requirements.txt
pip install cupy-cuda12x==13.4.1
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install deepspeed --prefer-binary

<!-- cd .. -->
powershell
Copy-Item .\custom_rasterizer_setup.py .\hunyuan3d-2.1\hy3dpaint\custom_rasterizer\setup.py -Force
Copy-Item .\DifferentiableRenderer_setup.py .\hunyuan3d-2.1\hy3dpaint\DifferentiableRenderer\setup.py

cmd
cp -f custom_rasterizer_setup.py hunyuan3d-2.1/hy3dpaint/custom_rasterizer/setup.py
cp -f DifferentiableRenderer_setup.py hunyuan3d-2.1/hy3dpaint/DifferentiableRenderer/setup.py

cd hunyuan3d-2.1\hy3dpaint\custom_rasterizer
pip install -e .

cd ..\DifferentiableRenderer
pip install -e .

cd ../.. 
mkdir hy3dpaint\ckpt
python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 'hy3dpaint\ckpt\RealESRGAN_x4plus.pth')"

<!-- wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt -->

<!-- cd ../.. -->
<!-- cd hy3dpaint/DifferentiableRenderer -->
<!-- bash compile_mesh_painter.sh -->
<!-- cd ../.. -->

<!-- wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt -->


<!-- conda activate Hunyuan3D
cd D:\model\3DMG\hunyuan3d-2.1\hy3dpaint\custom_rasterizer
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
python setup.py install -->