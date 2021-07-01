# Installation Notes for Tensorrt, Cuda, Cudnn, Anaconda, Pytorch, Tensorflow, Torch2trt in Ubuntu 18.04

## Table  of content

1. [Install nvidia driver 450](#install-nvidia-driver-450)
2. [Install cuda-10.2 and cudnn 8.0.5](#install-cuda-and-cudnn)
3. [Install Anaconda and Create Environment](#install-anaconda-and-create-environment)
4. [Install TensorRT-7.2.3](#install-tensorrt)
5. [Install Torch2trt](#install-torch2trt)

---

## Install Nvidia Driver 450
 Run below command to install nvidia driver
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
# exact version 450.102.04
sudo apt-get install nvidia-driver-450
```

Then reboot and check the nvidia driver
```bash
init 6
nvidia-smi
```

----

## Install Cuda and Cudnn

### Step 1.  CUDA 10.2 Deb File

- Download cuda toolkit 10.2 deb file from [nivida developer website](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)

- Then run below commands to install cuda 10.2.
```bash
cd (cuda download directory)
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
- Then add these cuda environment variables in `~/.bashrc` file.
```bash
nano ~/.bashrc
# add these variables in the bottom of ~/.bashrc file
export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- Then activate these variable with this command
```bash
source ~/.bashrc
```

### Step 2. CUDNN 8.0.5 Deb Files

- Downloads cudnn 8.0.5 deb files for cuda 10.2 from [nvidia developer website](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse805-102)
- Run these command to install cudnn 8.0.5
```bash
sudo dpkg -i libcudnn8_8.0.5.39-1+cuda10.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.5.39-1+cuda10.2_amd64.deb       
sudo dpkg -i libcudnn8-samples_8.0.5.39-1+cuda10.2_amd64.deb      
```

---

## Install Anaconda and Create Environment

- Download and install [anaconda](https://www.anaconda.com/products/individual#Downloads)
- Then create virtual environment
```bash
conda create -n dev python=3.7
```

- Install [pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/)
```bash
conda activate dev
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
python -c "import torch; print('Cuda:', torch.cuda.is_available())"
```

- Install [tensorflow-gpu](https://www.tensorflow.org/install/gpu)
```bash
conda activate dev
conda install tensorflow-gpu=2.2.0
python -c "import tensorflow as tf; print('Cuda:', tf.test.is_gpu_available())"
```
-Install other python computer vision packages
```bash
pip install Cython
pip install pycocotools
pip install -r ~/requirements.txt
```

---

## Install TensorRT

- Download [TensorRT `7.2.3`](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/tars/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz) tar file  for cuda 10.2 and cudnn8.1.
```bash
# extract tensorrt tar file
tar xzvf <downloaded TensorRT tar file>
```
__Note__* You can check official tensorrt installation instruction from [here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html).

- Then add these tensorrt environment variables in `~/.bashrc` file.
```bash
nano ~/.bashrc
# change your tensorrt extracted folder path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your tensorrt extracted folder>/lib
export PATH=$PATH:<your tensorrt extracted folder>/bin
```
- Then activate these variables with this command
```bash
source ~/.bashrc
```

- Install python packages from your tenssorrt extracted folder.
```bash
 cd <your tensorrt extracted folder>
 pip install python/tensorrt-7.1.3.4-cp<your python version>.whl
 pip install graphsurgeon/*.whl
 pip install uff/*.whl
```

---

## Install Torch2trt

- Install Torch2trt
```bash
sudo apt-get install libprotobuf* protobuf-compiler ninja-build
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd Torch2trt
python setup.py install --plugins
```
___

Bravo!!!
ENJOY your deep learning journey.

<img src=https://media.giphy.com/media/XRB1uf2F9bGOA/giphy.gif width="832" height="480"/> |
