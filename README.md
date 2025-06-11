# Romanian Lip Reading Dataset

## Requirements
- Git
- Docker (Optional, but recommended)
- TensorRT-compatible GPU (Optional, but speeds up inference). See [compatibility matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

## Setup
### Option 1: Docker
```sh
git clone https://github.com/AlexLicuriceanu/romanian-lip-reading.git
docker build -t romanian-lip-reading .
docker run --gpus all -it romanian-lip-reading
```

### Option 2: Install locally
```sh
git clone https://github.com/AlexLicuriceanu/romanian-lip-reading.git
```