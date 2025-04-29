from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gqsgd',
    ext_modules=[
        CUDAExtension('gqsgd_cuda', [
            'gqsgd_cuda_wrapper.cpp',
            'gqsgd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
)