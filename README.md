conda create -n Hunyuan3D python=3.12
conda activate Hunyuan3D

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
<!-- pip install termcolor transformers -->

git clone https://github.com/tencent-hunyuan/hunyuan3d-2.1

cd Hunyuan3D-2.1
pip install -r requirements.txt

cd hy3dpaint/custom_rasterizer
pip install -e .
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..

wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt