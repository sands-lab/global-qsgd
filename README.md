# Global-QSGD
Global-QSGD provides an easy-of-use python library to speedup deep learning training by quantize with global information.

### Prerequisite

Global-QSGD requires the installation of PyTorch and CUDA.

The code is tested on ASUS ESC N4A-E11 server equipped with 4 NVIDIA A100 GPUs, which runs. Ubuntu 22.04 with CUDA 11.6, and we use PyTorch 1.13.0.

We provides an `environment.yaml` file to quickly install the code.

We also give user guide on each model in `models` folder.
### Installation
* Install
    ```shell
    python setup.py install
    ```
* Uninstall
    ```shell
    pip uninstall gqsgd
    ```
* Check Installation
    ```shell
    python
    import torch
    import gqsgd
    ```
* Run a test
    ```shell
    # cd test
    python testddp.py
    ```
### Usage
Users can simply use Global-QSGD by registering the hook after wrap the model by DDP. Hook can choose from [default_hook, standard_dithering_hook, exponential_dithering_hook].
```python
from gqsgd.ddphook import *
model.register_comm_hook(None, exponential_dithering_hook)
```