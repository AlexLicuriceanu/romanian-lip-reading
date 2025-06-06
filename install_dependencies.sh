#!/bin/bash

sudo apt update
sudo apt install ffmpeg
sudo apt install nvidia-cuda-toolkit
sudo apt-get update
sudo apt-get install libcudnn9 libcudnn9-dev
sudo apt install libturbojpeg0-dev

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

