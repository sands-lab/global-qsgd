# Global-QSGD: Allreduce-Compatible Quantization for Distributed Learning 

[![License: MIT](https://img.shields.io/badge/License-MIT-Green.svg)](https://opensource.org/licenses/MIT)
[![Paper 2025](https://img.shields.io/badge/Paper-ECAI'25-blue.svg)](https://ecai2025.eu/)
[![Docker](https://img.shields.io/docker/pulls/myuser/myimage)](https://hub.docker.com/r/myuser/myimage)

Global-QSGD provides an easy-to-use Python library that accelerates distributed deep learning training through gradient quantization with global information. Our approach significantly reduces communication overhead while maintaining training convergence and model accuracy, enabling efficient scaling across multiple nodes.
We evaluated our approach on different models, including CNNs, Transformers, and Recommendation Models. The code is tested on ASUS ESC N4A-E11 server equipped with 4 NVIDIA A100 GPUs, which runs Ubuntu 22.04 with CUDA 11.6 and PyTorch 1.13.0.

## 🎯 Key Contributions
- **Global Normalization**: Gradient quantization with global norm 
- **Exponential Dithering**: Ensures convergence
- **Hardware-Optimized**: Efficient CUDA kernels for exponential encoding/decoding
- **Easy-to-use**: Seamless PyTorch DDP integration


## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: 1.13.0+
- **CUDA**: 11.6+
- **Hardware**: Tested on NVIDIA A100 GPUs
- **OS**: Ubuntu 22.04 (recommended)

## 🐳 Quick Start with Docker (Recommended)

The fastest way to get started is using our pre-configured Docker environment:

```bash
# Pull the official Docker image
docker pull messagebuffer/global-qsgd:latest

# Run with GPU support
docker run --ipc=host --net=host --gpus=all \
           --ulimit memlock=-1:-1 \
           --name GlobalQSGD \
           -it messagebuffer/global-qsgd:latest bash
```

## 🔧 Installation from Source

### Option 1: Quick Installation
```bash
cd ~
git clone git@github.com:sands-lab/global-qsgd.git
cd global-qsgd
python3 setup.py install
```

### Option 2: Development Installation
```bash
cd ~
rm -r Global-QSGD
git clone git@github.com:sands-lab/global-qsgd.git
cd global-qsgd
pip3 install -e .
```

### Verify Installation
```python
# Installation Check
python3
>>> import torch
>>> import gqsgd
>>> from gqsgd.ddphook import *
>>> from gqsgd import lgreco_hook, powerSGD_hook

# Run simple test for distributed communication
python3 test/testddp.py
```

## 💡 Usage

### Basic Integration

Global-QSGD seamlessly integrates with PyTorch's DistributedDataParallel (DDP):

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gqsgd.ddphook import standard_dithering_hook, exponential_dithering_hook

# Initialize your model
model = YourModel()
ddp_model = DDP(model, device_ids=[local_rank])

# Register Global-QSGD communication hook
ddp_model.register_comm_hook(None, exponential_dithering_hook)

# Training proceeds normally
for batch in dataloader:
    optimizer.zero_grad()
    output = ddp_model(batch)
    loss = criterion(output, target)
    loss.backward()  # Gradients are automatically quantized
    optimizer.step()
```

## 🚀 Supported Quantization Methods

| Method | Description | Hook Name | Note |
|--------|-------------|-----------|----------|
| **Global-QSGD Standard Dithering** | Linear quantization with global norm | `standard_dithering_hook` | Best speed-up |
| **Global-QSGD Exponential Dithering** | Exponential quantization with global norm | `exponential_dithering_hook` | Best convergence |
| **THC** | Quantization with global norm | `thc_hook` | Baseline for Allreduce compatible quantization |
| **PowerSGD** | Low-rank matrix approximation | `powerSGD_hook` | Baseline for Allreduce compatible decomposition |
| **QSGD** | Quantized SGD with stochastic rounding | `qsgd_hook` | Baseline for Allgather based quantization |
| **Default (No Quantization)** | No quantization, standard DDP communication | `default_hook` | Baseline, no quantization |

## 🧪 Experimental Validation

Our framework has been extensively validated across three diverse domains:

### Recommendation Systems: DeepLight on Criteo

```bash
cd /root/global-qsgd/models/DeepLight  
bash ./launch.sh
```

Each experiment includes comprehensive comparisons across all quantization methods with detailed performance metrics.

### Natural Language Processing: TransformerXL on WikiText-103  

```bash
# Execute from host: Copy data inside docker
# dataset can be obtained from https://www.kaggle.com/datasets/dekomposition/wikitext103
# Rename files under wikitext-103 to: test.txt, train.txt, valid.txt
docker cp <path to wikitext> GlobalQSGD:/root/global-qsgd/models/TransformerXL/pytorch
# Execute inside docker
cd /root/global-qsgd/models/TransformerXL/pytorch
bash ./launch.sh
```

### Computer Vision: ResNet101 on ImageNet
```bash
# Execute from host: Copy data inside docker
# dataset can be obtained from https://www.kaggle.com/datasets/zcyzhchyu/mini-imagenet
# miniimagenet should  contains the train and val folders
# Inside the train and val folders, there are many subfolders that contain JPEG pictures
docker cp <path to miniimagenet> GlobalQSGD:/root/miniimagenet
# Execute inside docker
cd /root/global-qsgd/models/ResNet101
bash ./launch.sh
```

## 🏗️ Architecture Overview

```
Global-QSGD/
├── gqsgd/                    # Core quantization library
│   ├── ddphook.py           # PyTorch DDP integration hooks
│   ├── allreduce.py         # Distributed communication primitives  
│   ├── powerSGD_hook.py     # PowerSGD implementation
│   └── lgreco_hook.py       # LGreco adaptive compression
├── models/                   # Experimental validation
│   ├── ResNet101/           # Computer vision experiments
│   ├── TransformerXL/       # NLP experiments  
│   └── DeepLight/           # Recommendation system experiments
├── gqsgd_cuda.cu            # CUDA kernels for quantization
└── setup.py                 # Package installation
```

## 📄 Citation

If you use Global-QSGD in your research, please cite our ECAI 2025 paper:

```bibtex
@inproceedings{global-qsgd-ecai2025,
  title={Global-QSGD: Allreduce-Compatible Quantization for Distributed Learning with Theoretical Guarantees},
  author={Jihao Xin and Marco Canini and Peter Richtárik and Samuel Horváth},
  booktitle={Proceedings of the European Conference on Artificial Intelligence (ECAI)},
  year={2025},
  publisher={IOS Press}
}
```
---
<div align="center">
  Made with ❤️ by the Global-QSGD Team from KAUST & MBZUAI
</div>