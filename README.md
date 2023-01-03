# Global-Quantization

* Install

    ```shell
    python setup.py install
    ```
* Uninstall
    ```shell
    pip uninstall gqsgd gqsgd_cuda
    ```
* Check Installation
    ```shell
    python
    import torch
    import gqsgd
    import gqsgd_cuda
    ```
* Usage
    ```python
    from gqsgd.ddphook import *
    model.register_comm_hook(None, exponential_dithering_hook)# Register right after calling model.ddp()
    ```