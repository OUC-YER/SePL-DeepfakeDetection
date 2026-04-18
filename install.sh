#!/bin/bash

echo "Installing dependencies..."

# PyTorch (CUDA 11.3)
#pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
#--extra-index-url https://download.pytorch.org/whl/cu113

pip install numpy==1.26.4
pip install pandas==2.3.1
pip install Pillow==12.0.0
pip install imgaug==0.4.0
pip install tqdm==4.67.1
pip install scipy==1.15.3

pip install opencv-python==4.11.0.86
pip install scikit-image==0.25.2
pip install scikit-learn==1.7.2

pip install albumentations==1.3.0
pip install efficientnet-pytorch==0.7.1
pip install timm==0.6.12
pip install segmentation-models-pytorch==0.3.2

pip install tensorboard==2.20.0
pip install einops==0.8.1
pip install transformers==4.46.2
pip install kornia==0.8.1

pip install simplejson==3.20.2
pip install lmdb==2.2.0
pip install fvcore==0.1.5.post20221221
pip install iopath==0.1.10
pip install portalocker==3.2.0
pip install tabulate==0.10.0
pip install termcolor==3.3.0
pip install yacs==0.1.8

echo "Installation completed!"