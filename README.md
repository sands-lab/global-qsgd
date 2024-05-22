# Global-QSGD
Global-QSGD provides an easy-of-use python library to speedup deep learning training by quantize with global information.

### Prerequisite

Global-QSGD requires the installation of PyTorch and CUDA.

The code is tested on ASUS ESC N4A-E11 server equipped with 4 NVIDIA A100 GPUs, which runs. Ubuntu 22.04 with CUDA 11.6, and PyTorch 1.13.0.

### Run with Docker (Recommended)
The recommended way to config our project is to use Docker, where we have installed Global-QSGD and all the dependencies to reproduce our experiment.
```shell
docker pull messagebuffer/global-qsgd:latest
docker run --ipc=host --net=host --gpus=all --ulimit memlock=-1:-1 -v <mount path> --name GlobalQSGD -it messagebuffer/global-qsgd:latest bash
```


### Compile from scratch
User can also compile the source code to a python package by yourself.
In case to do so, we suggest you also to run inside the provided Docker container to avoid version imcompatibility.
We have packed Global-QSGD as a python package which can be simply use with pip.
* **Install.** 
    ```shell
    cd Global-QSGD
    python3 setup.py install
    ```
* **Uninstall** 
    ```shell
    pip uninstall gqsgd
    ```
* **Check Installation**
    ```shell
    python3
    import torch
    import gqsgd
    from gqsgd.ddphook import *
    from gqsgd import lgreco_hook, powerSGD_hook
    ```
* **Sanity Check**
    ```shell
    # cd test
    python3 testddp.py
    ```
### Usage
Users can simply use Global-QSGD by registering the hook after wrap the model by DDP.

Hook can choose from [default_hook, standard_dithering_hook, exponential_dithering_hook].
```python
# wrap up the model with Python DDP 
from gqsgd.ddphook import *
ddp_model.register_comm_hook(None, exponential_dithering_hook)
```

### Examples
We provide experiment with 3 models:
* DeepLight
* ResNet101
* TransformerXL
We give specific user guide on each model in the `models` folder.