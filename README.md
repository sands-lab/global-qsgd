# gqsgd

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
    import gqsgd
    ```
* Usage
    ```python
    from gqsgd.ddphook import *
    model.register_comm_hook(None, exponential_dithering_hook)# [default_hook, standard_dithering_hook, exponential_dithering_hook]
    ```