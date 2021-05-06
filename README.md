# Installation notes for tensorrt, cuda, pytorch, tensorflow, torch2trt in ubuntu 18.04

## Table  of content

1. [Install nvidia driver 450](#install-nvidia-driver-450)
2. [Install cuda-10.2 and cudnn 8.0.2](#install-cuda-and-cudnn)
3. [Install Anaconda and Create Environment](#install-anaconda-and-create-environment)
4. [Install TensorRT-7.1.3.4](#install-tensorrt)
5. [Install Torch2trt and Trt_pose](#install-torch2trt-and-trt_pose)

## Install Nvidia Driver 450
  install nvidia-drivers for ubuntu 18.04
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
# exact version 450.102.04
sudo apt-get install nvidia-driver-450
```
After this reboot pc
- init 6

and check nvidia driver
- nvidia-smi

----
## Install Cuda and Cudnn
- Download cuda toolkit 10.2 deb file from [nivida developer website](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)
- Install cuda 10.2 deb file by following their instructions.
- Then add these lines in `~/.bashrc` file.
    - export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
    - export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
- Then ```source ~/.zshrc or ~/.bashrc```
- Downloads cudnn 8.0.2 for cuda 10.2 from [nvidia developer website](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse802-102)
- Install cudnn 8.0.2 with these command
```bash
sudo dpkg -i libcudnn8_8.0.2.39-1+cuda10.2_amd64.deb
libcudnn8-dev_8.0.2.39-1+cuda10.2_amd64.deb       
libcudnn8-doc_8.0.2.39-1+cuda10.2_amd64.deb                 
```

---
## Install Anaconda and Create Environment
- Install Anaconda
- Create virtual environment and install packages
```bash
conda create -n dev python=3.7
conda activate dev
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
conda install tensorflow-gpu=2.2.0
pip install Cython
pip install pycocotools
# install other computervision packages
pip install -r requirements.txt
```
---
## Install TensorRT
- Download and install TensorRT tar file.
- Then add these lines in `~/.bashrc` file.
    - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zmh/hdd/backups/Downloads/tensor-rt+cuda-10.2/TensorRT-7.1.3.4/lib
    - export PATH=$PATH://home/zmh/hdd/backups/Downloads/tensor-rt+cuda-10.2/TensorRT-7.1.3.4/bin
- Then ```source ~/.bashrc```.
- Install python packages from extracted tensorrt tar file.
```bash
 cd TensorRT-7.1.3.4
 pip install $tenosrrt_dir/python/tensorrt-7.1.3.4-cp(python version).whl
 pip install $tensorrt_dir/graphsurgeon/*.whl
 pip install $tensorrt_dir/uff/*.whl
```
---

## Install Torch2trt and Trt_pose
- Install Torch2trt
```bash
sudo apt-get install libprotobuf* protobuf-compiler ninja-build
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd Torch2trt
python setup.py install --plugins
```
- Install Trt_pose
```bash
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python setup.py install
```
