from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gqsgd',
    ext_modules=[
        CUDAExtension('gqsgd_cuda', [
            'gqsgd.cpp',
            'gqsgd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
)