from setuptools import setup, Extension
from torch.utils import cpp_extension

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    
    name='prosparsity_engine',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('prosparsity_engine', [
            'pybind.cpp',
            'prosparsity_cuda.cu',
        ]),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)