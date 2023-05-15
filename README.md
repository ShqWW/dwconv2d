# Efficient Implementation of 2D Depthwise Convolution for Large Kernel Based on Pytorch and Nvidia CUDA

This is an efficient cuda implementation of 2D depthwise convolution for large kernel, it can be used in Pytorch deep learning framework.


## Feature
* Only support for fp32 precision （fp16 will be added in the future).

* Support for forward and backward mode.

* Support for kernel size range from 3 to 31, and the kernel width and height can be different.

* Support for bias.

* Faster training and testing speed with large kernel size than `nn.conv2d` in Pytorch.

## Install
The recommended environment is Ubuntu22.04+cuda11.8+cudnn8.x+Pytorch2.0.

Running `python setup.py install` under your environment to compile and install the package, it may be time-consuming depended on your hardware.


## Usage
Trying the demo code below.
```python
import torch
from Dwconv.dwconv_layer import *

layer = DwConv2d(num_channel=384, kernel_size=(5, 7), padding=(2, 3), bias=True).cuda()
input = torch.randn(64, 384, 32, 32).cuda()
output = layer(input)
print(output.shape)
```
You can also try [test.py](https://github.com/ShqWW/dwconv2d/blob/master/test.py) to test the speed of depthwise conv with different kernel sizes and input sizes. Here is the testing result on Geforce RTX 4090:

    **************new dwconv2d****************
    kernel size 3 iter time 68.85363801848143 ms
    kernel size 5 iter time 99.33419805020094 ms
    kernel size 7 iter time 101.52079409454018 ms
    kernel size 9 iter time 104.76248001214117 ms
    kernel size 11 iter time 113.58783196192235 ms
    kernel size 13 iter time 131.15966296754777 ms
    kernel size 15 iter time 148.0217109201476 ms
    kernel size 17 iter time 186.7981820832938 ms
    kernel size 19 iter time 213.8294338947162 ms
    kernel size 21 iter time 254.66446299105883 ms
    kernel size 23 iter time 275.64177196472883 ms
    kernel size 25 iter time 316.41299498733133 ms
    kernel size 27 iter time 413.0465209018439 ms
    kernel size 29 iter time 453.05013796314597 ms
    kernel size 31 iter time 459.79193004313856 ms
    *******************************************
    **************pytorch dwconv2d****************
    kernel size 3 iter time 84.65386694297194 ms
    kernel size 5 iter time 198.80758307408541 ms
    kernel size 7 iter time 351.4602029463276 ms
    kernel size 9 iter time 543.8372779171914 ms
    kernel size 11 iter time 784.702883916907 ms
    kernel size 13 iter time 1091.992371948436 ms
    kernel size 15 iter time 1493.0288789328188 ms
    kernel size 17 iter time 1915.5742489965633 ms
    kernel size 19 iter time 2397.6722250226885 ms
    kernel size 21 iter time 2966.810050071217 ms
    kernel size 23 iter time 3575.081075890921 ms
    kernel size 25 iter time 4168.36630506441 ms
    kernel size 27 iter time 4771.884319023229 ms
    kernel size 29 iter time 5629.526844015345 ms
    kernel size 31 iter time 6489.46202499792 ms
    *******************************************

## Thanks
This project is motivated by ["Megengin"](https://github.com/MegEngine/MegEngine) and its paper "[RepLKNet](https://arxiv.org/abs/2203.06717)"。




