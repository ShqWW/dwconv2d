#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <iostream>

#include "cuda.h"
#include "kernel.cuh"
#include "para.cuh"



using IntArrayRef = c10::ArrayRef<int64_t>;

at::Tensor Dwconv2dLauncherFP32Dgrad(const at::Tensor Ograd, const at::Tensor weight_inv, const int padding_h, const int padding_w)
{
    const int N = Ograd.size(0);
    const int C = Ograd.size(1);
    const int OgradH = Ograd.size(2);
    const int OgradW = Ograd.size(3);

    const int kH = weight_inv.size(2);
    const int kW = weight_inv.size(3);

    const int IgradH = OgradH + kH - 1 - padding_h * 2;
    const int IgradW = OgradW + kW - 1 - padding_w * 2;
    const int padding_ho = kH - 1 - padding_h;
    const int padding_wo = kW - 1 - padding_w;


    if (kH==3&&kW==3&&padding_h==1&&padding_w==1) 
    {
        // IntArrayRef kernel_size = {kH, kW};
        at::NoGradGuard no_grad;
        at::Tensor dst = at::cudnn_convolution(Ograd, weight_inv, {1, 1}, {1, 1}, {1, 1}, C, 1, 1, 1);
        return dst;
    }
    else
    {
        Param param = {N, C, OgradH, OgradW, kH, kW, IgradH, IgradW, padding_ho, padding_wo};
        at::Tensor dst = at::zeros({N, C, IgradH, IgradW}, Ograd.options());
        const float *src_ = Ograd.data_ptr<float>();
        const float *flt_ = weight_inv.data_ptr<float>();
        float *dst_ = dst.data_ptr<float>();
        INSTANCE
        return dst;
    } 
}

