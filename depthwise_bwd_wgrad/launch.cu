#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <iostream>

#include "cuda.h"

#include "kernel.cuh"
#include "para.cuh"



using IntArrayRef = c10::ArrayRef<int64_t>;

at::Tensor Dwconv2dLauncherFP32Wgrad(const at::Tensor input, const at::Tensor ograd, const int padding_h, const int padding_w)
{
    const int N = input.size(0);
    const int C = input.size(1);
    const int iH = input.size(2);
    const int iW = input.size(3);

    const int oH = ograd.size(2);
    const int oW = ograd.size(3);


    const int kH = iH - oH + 1 + padding_h * 2;
    const int kW = iW - oW + 1 + padding_w * 2;

    Param param = {N, C, iH, iW, kH, kW, oH, oW, padding_h, padding_w};

    at::Tensor dst = at::zeros({C, 1, kH, kW}, input.options());
    const float *src_ = input.data_ptr<float>();
    const float *ograd_ = ograd.data_ptr<float>();
    float *dst_ = dst.data_ptr<float>();

    INSTANCE
    return dst;
}

at::Tensor Dwconv2dLauncherFP32Bgrad(const at::Tensor ograd)
{
    at::Tensor dst = ograd.sum(IntArrayRef{0, 2, 3});
    return dst;
}