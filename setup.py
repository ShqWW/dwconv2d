from setuptools import setup
from torch.utils import cpp_extension

from Cython.Build import cythonize
setup(
    name ='dwconv2d',   #编译后的链接名称
    ext_modules=[cpp_extension.CUDAExtension('dwconv2d', 
    ['dwconv2d.cpp', 
    'depthwise_fwd/launch.cu',
    'depthwise_bwd_wgrad/launch.cu',
    'depthwise_bwd_dgrad/launch.cu',
    ],)
    ],   #待编译文件，以及编译函数
    cmdclass={'build_ext': cpp_extension.BuildExtension}, #执行编译命令设置
    packages=['Dwconv']
)

# CUDAExtension
# CppExtension